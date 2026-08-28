# Late Delivery Risk API

FastAPI serving layer for the supply-chain late-delivery prediction model.

The API accepts raw order records, applies the same preprocessing rules used during training, loads the saved Gradient Boosting model bundle, and returns a late-delivery probability and binary prediction.

## Architecture

Raw order data

→ FastAPI endpoint

→ `src/predict.py`

→ `src/preprocessing.py`

→ saved `model_bundle.joblib`

→ prediction

## Model

The production model is a tuned `GradientBoostingClassifier`.

Parameters:

- `n_estimators = 100`
- `learning_rate = 0.05`
- `max_depth = 3`
- decision threshold = `0.40`

Final test performance:

- ROC-AUC: `0.7710`
- Accuracy: `0.6800`
- Precision: `0.7076`
- Recall: `0.7058`
- F1-score: `0.7067`

## Endpoints

### Health Check

`GET /health`

Example response:

```json
{
  "status": "healthy"
}
