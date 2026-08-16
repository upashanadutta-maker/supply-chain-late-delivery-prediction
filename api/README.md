# Late Delivery Risk API

Serves the XGBoost model from this repo behind an HTTP endpoint.

## Endpoints
- `GET /health` - service status and expected feature count
- `POST /predict` - accepts 31 features, returns probability, class, and latency

## Run
    pip install -r requirements.txt
    uvicorn app:app --reload

Interactive docs at http://127.0.0.1:8000/docs

## Design notes
- Model loads once at startup, not per request
- Incoming features are reordered to match training column order before inference
- Requests missing any of the 31 features return a list of what is absent
- Per-request latency is logged
