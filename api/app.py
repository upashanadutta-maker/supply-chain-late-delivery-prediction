import json
import time
import logging

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)

model = xgb.Booster()
model.load_model("supply_chain_model.json")

with open("feature_names.json") as f:
    FEATURE_NAMES = json.load(f)

app = FastAPI(title="Late Delivery Risk API")


class PredictionRequest(BaseModel):
    features: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok", "n_features": len(FEATURE_NAMES)}


@app.post("/predict")
def predict(request: PredictionRequest):
    start = time.time()

    missing = [name for name in FEATURE_NAMES if name not in request.features]
    if missing:
        return {"error": "missing features", "missing": missing}

    row = pd.DataFrame([request.features])[FEATURE_NAMES]
    probability = float(model.predict(xgb.DMatrix(row))[0])

    latency_ms = (time.time() - start) * 1000
    logging.info(f"prediction={probability:.4f} latency_ms={latency_ms:.2f}")

    return {
        "late_delivery_probability": probability,
        "predicted_class": int(probability > 0.5),
        "latency_ms": round(latency_ms, 2),
    }
