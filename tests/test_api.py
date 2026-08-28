import pandas as pd

from fastapi.testclient import TestClient

import api.app as api_module


client = TestClient(api_module.app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy"
    }


def test_predict_endpoint(monkeypatch):

    def fake_get_model_bundle():
        return {"fake": "bundle"}

    def fake_predict(data, model_bundle):
        return pd.DataFrame({
            "late_delivery_probability": [0.75],
            "late_delivery_prediction": [1],
        })

    monkeypatch.setattr(
        api_module,
        "get_model_bundle",
        fake_get_model_bundle,
    )

    monkeypatch.setattr(
        api_module,
        "predict",
        fake_predict,
    )

    response = client.post(
        "/predict",
        json={
            "records": [
                {
                    "Type": "DEBIT"
                }
            ]
        },
    )

    assert response.status_code == 200

    assert response.json() == {
        "predictions": [
            {
                "late_delivery_probability": 0.75,
                "late_delivery_prediction": 1,
            }
        ]
    }
