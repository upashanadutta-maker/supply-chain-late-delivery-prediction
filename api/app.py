from typing import Any

import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.predict import (
    load_model_bundle,
    predict,
)


app = FastAPI(
    title="Late Delivery Risk API",
    description=(
        "Predicts the probability that a supply-chain "
        "order will be delivered late."
    ),
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    records: list[dict[str, Any]]


model_bundle = None


def get_model_bundle():
    """
    Load the trained model bundle once
    and reuse it for future predictions.
    """

    global model_bundle

    if model_bundle is None:
        model_bundle = load_model_bundle()

    return model_bundle


@app.get("/")
def root():
    return {
        "message": "Late Delivery Risk API is running."
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def make_prediction(
    request: PredictionRequest
):
    """
    Predict late-delivery risk for one or
    more raw order records.
    """

    try:
        data = pd.DataFrame(
            request.records
        )

        bundle = get_model_bundle()

        predictions = predict(
            data,
            bundle
        )

        return {
            "predictions": (
                predictions
                .to_dict(
                    orient="records"
                )
            )
        }

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )
