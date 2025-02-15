import fastapi
from fastapi import FastAPI, HTTPException
import websockets
import json
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timezone
from collections import deque, defaultdict
import torch
from chronos import BaseChronosPipeline
import logging
from typing import Dict, Any, Tuple, Optional, List
from contextlib import asynccontextmanager

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PredictionCache:
    def __init__(self):
        self.predictions: Dict[str, Dict[str, Any]] = {}
        self.last_updated: Dict[str, datetime] = {}
        self.collection_start: Dict[str, datetime] = {}
        self.is_collecting: Dict[str, bool] = {}

    def start_collection(self, symbol: str):
        self.collection_start[symbol] = datetime.now(timezone.utc)
        self.is_collecting[symbol] = True
        logger.info(f"Started collecting data for {symbol}")

    def update_prediction(self, symbol: str, prediction: Dict[str, Any]):
        logger.debug(f"Updating prediction for {symbol}: {prediction}")
        self.predictions[symbol] = prediction
        self.last_updated[symbol] = datetime.now(timezone.utc)
        self.is_collecting[symbol] = False

    def get_prediction(self, symbol: str) -> Optional[Dict[str, Any]]:
        return self.predictions.get(symbol)

    def should_predict(self, symbol: str) -> bool:
        if not self.is_collecting.get(symbol):
            return False
            
        if symbol not in self.collection_start:
            return False
            
        collection_time = (datetime.now(timezone.utc) - self.collection_start[symbol]).total_seconds()
        return collection_time >= 60

class PricePredictor:
    def __init__(self, websocket_url: str, api_key: str, symbols: List[str]):
        self.websocket_url = websocket_url
        self.api_key = api_key
        self.symbols = symbols
        self.price_caches = {symbol: deque() for symbol in symbols}  # 30 mins of data at 5s intervals
        self.websocket = None
        self.prediction_cache = PredictionCache()
        self.is_connected = False
        
        try:
            self.pipeline = BaseChronosPipeline.from_pretrained(
                "amazon/chronos-bolt-base",
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.bfloat16,
            )
            logger.info("Successfully initialized Chronos pipeline")
        except Exception as e:
            logger.error(f"Failed to initialize Chronos pipeline: {e}")
            raise

    async def connect_websocket(self) -> None:
        try:
            if self.websocket:
                await self.websocket.close()
            
            self.websocket = await websockets.connect(self.websocket_url)
            
            # Subscribe to the prices topic
            subscribe_message = {
                "topic": "prices",
                "event": "phx_join",
                "ref": None,
                "payload": {"api_key": self.api_key}
            }
            await self.websocket.send(json.dumps(subscribe_message))
            response = await self.websocket.recv()
            logger.debug(f"Subscription response: {response}")
            
            self.is_connected = True
            logger.info(f"WebSocket connected and subscribed to prices topic")
            
            # Start collecting data for all symbols
            for symbol in self.symbols:
                self.prediction_cache.start_collection(symbol)
                
        except Exception as e:
            self.is_connected = False
            logger.error(f"WebSocket connection failed: {e}")
            raise

    def process_price_update(self, data: Dict[str, Any]) -> Tuple[str, pd.Timestamp, float]:
        try:
            logger.debug(f"Processing price update: {data}")
            timestamp = pd.Timestamp(data['payload']['timestamp'])
            price = float(data['payload']['price'])
            symbol = data['payload']['symbol']
            return symbol, timestamp, price
        except Exception as e:
            logger.error(f"Error processing price update: {e}")
            raise

    def prepare_forecast_data(self, symbol: str) -> pd.DataFrame:
        if len(self.price_caches[symbol]) < 2:
            raise ValueError(f"Insufficient price data for forecasting {symbol}")
            
        df = pd.DataFrame(list(self.price_caches[symbol]), columns=['timestamp', 'price'])
        df.set_index('timestamp', inplace=True)
        df = df.sort_index()
        return df

    def predict_direction(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        try:
            # Generate forecast for next 60 steps (assuming 5-second intervals)
            quantiles, _ = self.pipeline.predict_quantiles(
                context=torch.tensor(df['price'].values, dtype=torch.float32),
                prediction_length=60,
                quantile_levels=[0.1, 0.5, 0.9]
            )
            
            # Get median predictions
            predictions = quantiles[0, :, 1].numpy()
            
            # Compare last actual price with last predicted price
            last_actual_price = df['price'].iloc[-1]
            last_predicted_price = predictions[-1]
            
            price_direction = "up" if last_predicted_price > last_actual_price else "down"
            price_change_pct = abs((last_predicted_price - last_actual_price) / last_actual_price * 100)
            
            confidence_boost = min(len(self.price_caches[symbol]) / 60.0, 1.0)
            adjusted_confidence = price_change_pct * confidence_boost

            prediction = {
                "direction": price_direction,
                "confidence": round(float(adjusted_confidence), 2),
                "current_price": float(last_actual_price),
                "predicted_price": float(last_predicted_price),
                "timestamp": df.index[-1].isoformat(),
                "data_points": len(self.price_caches[symbol]),
                "price_change_pct": round(float(price_change_pct), 2)
            }
            
            logger.debug(f"Generated prediction for {symbol}: {prediction}")
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}")
            return {"error": str(e)}

    async def process_websocket_messages(self):
        while True:
            try:
                if not self.is_connected:
                    logger.warning("WebSocket disconnected, attempting to reconnect...")
                    await self.connect_websocket()
                    continue

                message = await self.websocket.recv()
                logger.debug(f"Received WebSocket message: {message}")
                data = json.loads(message)
                
                if data.get('event') == 'price_update':
                    symbol, timestamp, price = self.process_price_update(data)
                    
                    if symbol in self.symbols:
                        self.price_caches[symbol].append((timestamp, price))
                        logger.debug(f"Added price point for {symbol}: {price} at {timestamp}")
                        
                        # Check if we should make a prediction
                        if self.prediction_cache.should_predict(symbol):
                            logger.info(f"30 seconds elapsed for {symbol}, generating prediction")
                            df = self.prepare_forecast_data(symbol)
                            prediction = self.predict_direction(df, symbol)
                            self.prediction_cache.update_prediction(symbol, prediction)
                            # Start collecting new data for the next prediction
                            self.prediction_cache.start_collection(symbol)
                    
            except websockets.exceptions.ConnectionClosed:
                logger.error("WebSocket connection closed")
                self.is_connected = False
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error in websocket message processing: {e}")
                await asyncio.sleep(1)
                continue

    async def maintain_connection(self):
        while True:
            try:
                if not self.is_connected:
                    await self.connect_websocket()
                await self.process_websocket_messages()
            except Exception as e:
                logger.error(f"Connection maintenance error: {e}")
                self.is_connected = False
                await asyncio.sleep(5)

# FastAPI app setup
WEBSOCKET_URL = "ws://localhost:4000/ws/websocket"
API_KEY = "" # Funstash API Key
SYMBOLS = ["pepe", "shib", "doge", "bonk", "wif", "floki", "popcat", "pnut", "turbo", "pengu"]

predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global predictor
    predictor = PricePredictor(WEBSOCKET_URL, API_KEY, SYMBOLS)
    asyncio.create_task(predictor.maintain_connection())
    yield
    # Shutdown
    if predictor and predictor.websocket:
        await predictor.websocket.close()

app = FastAPI(lifespan=lifespan)

@app.get("/api/token/{token_symbol}")
async def get_token_prediction(token_symbol: str):
    if token_symbol not in SYMBOLS:
        raise HTTPException(status_code=404, detail="Token not found")
        
    prediction = predictor.prediction_cache.get_prediction(token_symbol)
    if not prediction:
        if predictor.prediction_cache.is_collecting.get(token_symbol, False):
            raise HTTPException(
                status_code=404, 
                detail="Still collecting data for prediction"
            )
        raise HTTPException(
            status_code=404, 
            detail="No prediction available yet"
        )
        
    return prediction

@app.get("/api/status")
async def get_status():
    """Get the status of all token predictions"""
    if not predictor:
        return {"status": "initializing"}
        
    return {
        "status": "connected" if predictor.is_connected else "disconnected",
        "symbols": SYMBOLS,
        "predictions_available": list(predictor.prediction_cache.predictions.keys()),
        "currently_collecting": {
            symbol: predictor.prediction_cache.is_collecting.get(symbol, False)
            for symbol in SYMBOLS
        },
        "collection_started": {
            symbol: predictor.prediction_cache.collection_start[symbol].isoformat()
            if symbol in predictor.prediction_cache.collection_start else None
            for symbol in SYMBOLS
        },
        "cache_sizes": {
            symbol: len(predictor.price_caches[symbol]) 
            for symbol in SYMBOLS
        },
        "last_updated": {
            symbol: predictor.prediction_cache.last_updated[symbol].isoformat()
            for symbol in predictor.prediction_cache.predictions.keys()
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
