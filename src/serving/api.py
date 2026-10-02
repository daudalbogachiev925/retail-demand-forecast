"""FastAPI serving endpoint."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from pathlib import Path
from src.models.lgbm_model import DemandLGBM

app = FastAPI(title="Demand Forecast API", version="0.1.0")

MODEL_PATH = Path("models/lgbm_demand.txt")
model: DemandLGBM | None = None


class ForecastRequest(BaseModel):
    sku_id: str
    features: List[float]


class ForecastResponse(BaseModel):
    sku_id: str
    forecast: float
    model_version: str = "0.1.0"


@app.on_event("startup")
def load_model():
    global model
    if MODEL_PATH.exists():
        model = DemandLGBM.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"WARNING: no model at {MODEL_PATH}")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    x = np.array([req.features])
    pred = model.predict(x)
    return ForecastResponse(sku_id=req.sku_id, forecast=float(pred[0]))
