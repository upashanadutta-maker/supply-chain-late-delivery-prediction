import os
from pathlib import Path

import joblib
import pandas as pd

from src.preprocessing import transform_features


DEFAULT_MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "artifacts/model_bundle.joblib",
)

def load_model_bundle(model_path=DEFAULT_MODEL_PATH):
    """
    Load the trained model and preprocessing artifacts.
    """

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model bundle not found: {model_path}"
        )

    return joblib.load(model_path)


def predict(data, model_bundle):
    """
    Predict late-delivery risk for new raw records.
    """

    model = model_bundle["model"]

    preprocessing_artifacts = (
        model_bundle["preprocessing_artifacts"]
    )

    threshold = model_bundle["threshold"]

    processed_data = transform_features(
        data,
        preprocessing_artifacts
    )

    probabilities = model.predict_proba(
        processed_data
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    results = pd.DataFrame({
        "late_delivery_probability": probabilities,
        "late_delivery_prediction": predictions,
    })

    return results


def predict_from_csv(
    csv_path,
    model_path=DEFAULT_MODEL_PATH
):
    """
    Load raw records from a CSV and return predictions.
    """

    data = pd.read_csv(
        csv_path,
        encoding="latin-1"
    )

    model_bundle = load_model_bundle(
        model_path
    )

    return predict(
        data,
        model_bundle
    )
