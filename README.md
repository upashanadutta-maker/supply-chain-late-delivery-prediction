# Late Delivery Risk Prediction in Smart Supply Chains

An end-to-end machine learning project for predicting whether a supply-chain order is at risk of late delivery using information available around the time the order is placed.

The project goes beyond notebook-based modeling by implementing reusable preprocessing, reproducible training, persisted model artifacts, batch inference, a FastAPI serving layer, automated tests, Docker containerization, and GitHub Actions CI.

## Project Overview

Late deliveries can affect customer satisfaction, operational planning, inventory decisions, and supply-chain efficiency.

The objective of this project is to predict:

- `1` — Late delivery
- `0` — Not late

The analysis uses the DataCo Smart Supply Chain dataset containing approximately:

- 180,000 line-item records
- 65,000+ unique orders

Because a single order can contain multiple line-item records, special care was taken to prevent records belonging to the same `Order Id` from appearing across training, validation, and test sets.

---

## Final Model Performance

The selected model is a tuned `GradientBoostingClassifier`.

| Metric | Validation | Final Test |
|---|---:|---:|
| ROC-AUC | 0.7764 | 0.7710 |
| Accuracy | 0.6904 | 0.6800 |
| Precision | 0.7221 | 0.7076 |
| Recall | 0.7047 | 0.7058 |
| F1-score | 0.7133 | 0.7067 |

A secondary order-level evaluation was also performed by averaging line-item probabilities within each `Order Id`.

### Order-Level Test Performance

| Metric | Score |
|---|---:|
| ROC-AUC | 0.7728 |
| Accuracy | 0.6822 |
| Precision | 0.7114 |
| Recall | 0.7071 |
| F1-score | 0.7092 |

The similarity between row-level and order-level results suggests that repeated line items from larger orders are not materially driving the model's performance.

---

## Model Configuration

The final Gradient Boosting model uses:

```python
GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
```

The classification threshold is:

```text
0.40
```

The threshold was selected using validation data to provide a reasonable precision-recall trade-off in the absence of a defined business cost function.

The untouched test set was not used for model selection, hyperparameter tuning, or threshold selection.

---

## Leakage-Safe Experimental Design

A major focus of the project was preventing optimistic evaluation caused by leakage.

### Grouped Train / Validation / Test Split

The data was split at the unique `Order Id` level before modeling:

```text
60% Training
20% Validation
20% Test
```

This ensures that line items belonging to the same order cannot appear in multiple partitions.

Final split sizes:

| Split | Rows | Unique Orders |
|---|---:|---:|
| Train | 108,215 | 39,451 |
| Validation | 35,933 | 13,150 |
| Test | 36,371 | 13,151 |

Order overlap between all three sets is zero.

---

## Target Leakage Audit

Several variables were excluded because they reveal information that would not be appropriate for an order-placement / pre-dispatch prediction system.

### Leakage Columns

```text
Delivery Status
Days for shipping (real)
Order Status
shipping date (DateOrders)
```

For example, `Days for shipping (real)` directly reflects the realized shipping outcome and is therefore unavailable at prediction time.

### Identifier and Personal Columns

Customer identifiers, order identifiers, product identifiers, and personal information were removed from model inputs.

Examples include:

```text
Customer Email
Customer Fname
Customer Lname
Customer Password
Customer Street
Customer Id
Order Id
Order Customer Id
Order Item Id
Product Card Id
Category Id
Department Id
Product Category Id
Product Image
Customer Zipcode
```

`Order Id` is used for grouped splitting but is not used as a predictive feature.

---

## Feature Engineering

The raw order timestamp:

```text
order date (DateOrders)
```

is transformed into:

```text
order_month
order_dayofweek
order_hour
```

The original timestamp is then removed.

After preprocessing and constant-feature removal, the final model contains:

```text
31 features
```

---

## Categorical Encoding

The project uses count encoding for categorical variables.

Count mappings are learned from the **training data only**:

```text
Training data
    ↓
learn category counts
    ↓
save mappings
```

The same mappings are reused for validation, test, and production inference.

An unseen category receives:

```text
0
```

This prevents validation/test information from influencing training-time preprocessing and avoids training-serving skew.

---

## Model Development

Several classification algorithms were compared using validation ROC-AUC.

| Model | Validation ROC-AUC |
|---|---:|
| Logistic Regression | 0.6862 |
| XGBoost | 0.7618 |
| Random Forest | 0.7705 |
| Gradient Boosting | 0.7752 |
| Tuned Gradient Boosting | **0.7764** |

Gradient Boosting was selected based on validation ROC-AUC.

Hyperparameter tuning produced only a modest improvement over the default Gradient Boosting model, which is reported transparently rather than overstated.

---

## Model Interpretation

SHAP was used to interpret the final Gradient Boosting model.

The strongest model signals included:

1. Shipping service configuration
2. Scheduled shipment duration
3. Payment transaction type
4. Order hour

`Shipping Mode` and `Days for shipment (scheduled)` are strongly related in this dataset:

```text
Same Day       → 0 scheduled days
First Class    → 1 scheduled day
Second Class   → 2 scheduled days
Standard Class → 4 scheduled days
```

They should therefore be interpreted as related representations of the same shipping-service configuration rather than independent causal drivers.

SHAP values are interpreted as model associations, not causal effects.

---

# ML Engineering Architecture

The original data-science workflow was converted into reusable production-oriented components.

```text
                 Raw Supply-Chain Data
                          │
                          ▼
                 Group-Safe Splitting
                          │
                          ▼
                src/preprocessing.py
                          │
                          ▼
                    src/train.py
                          │
                          ▼
              artifacts/model_bundle.joblib
                          │
                          ▼
                   src/predict.py
                          │
                          ▼
                     FastAPI
                          │
                          ▼
                   POST /predict
```

The saved model bundle contains:

```text
trained Gradient Boosting model
preprocessing artifacts
training count mappings
constant-feature definitions
final feature order
decision threshold
model parameters
```

This ensures that inference uses exactly the same preprocessing rules that were learned during training.

---

## Project Structure

```text
supply-chain-late-delivery-prediction/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── api/
│   ├── app.py
│   ├── README.md
│   └── requirements.txt
│
├── artifacts/
│   └── .gitkeep
│
├── data/
│   └── .gitkeep
│
├── src/
│   ├── __init__.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
│
├── tests/
│   ├── test_api.py
│   └── test_preprocessing.py
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
├── README.md
└── supply_chain_late_delivery.ipynb
```

---

# Reproducible Training

The dataset is not committed to this repository.

The project was trained using the DataCo Smart Supply Chain dataset attached directly in Kaggle.

Clone the repository inside Kaggle:

```bash
git clone https://github.com/upashanadutta-maker/supply-chain-late-delivery-prediction.git
```

Move into the repository:

```bash
cd supply-chain-late-delivery-prediction
```

Run the training pipeline:

```bash
python -m src.train \
    --data-path "/kaggle/input/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis/DataCoSupplyChainDataset.csv"
```

The script performs:

```text
load dataset
    ↓
group-safe 60/20/20 split
    ↓
fit preprocessing on training data
    ↓
transform validation and test data
    ↓
train Gradient Boosting
    ↓
evaluate validation set
    ↓
evaluate untouched test set
    ↓
perform order-level robustness evaluation
    ↓
save model bundle
```

The generated artifact is:

```text
artifacts/model_bundle.joblib
```

Generated model binaries are intentionally excluded from Git.

---

# Inference

`src/predict.py` separates inference from training.

A saved model can therefore be loaded and used without retraining:

```python
from src.predict import load_model_bundle, predict

model_bundle = load_model_bundle()

predictions = predict(
    new_data,
    model_bundle
)
```

Example output:

```text
late_delivery_probability    late_delivery_prediction
0.4035                       1
```

Since the selected decision threshold is `0.40`, a probability greater than or equal to `0.40` receives prediction `1`.

---

# FastAPI Serving Layer

The model is exposed through a FastAPI application.

Start the API with:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

## Health Endpoint

```text
GET /health
```

Response:

```json
{
  "status": "healthy"
}
```

## Prediction Endpoint

```text
POST /predict
```

The endpoint accepts one or more raw order records and returns:

```json
{
  "predictions": [
    {
      "late_delivery_probability": 0.4035,
      "late_delivery_prediction": 1
    }
  ]
}
```

FastAPI automatically provides interactive API documentation at:

```text
/docs
```

---

# Model Artifact Management

The trained `model_bundle.joblib` is intentionally not committed to GitHub.

By default, inference expects:

```text
artifacts/model_bundle.joblib
```

The location can also be configured using the environment variable:

```text
MODEL_PATH
```

For example:

```bash
MODEL_PATH=/models/model_bundle.joblib
```

This keeps generated model artifacts separate from application source code.

---

# Docker

The FastAPI serving application is containerized using Docker.

Build the image:

```bash
docker build -t supply-chain-late-delivery-api .
```

Because the model artifact is external, mount the artifact directory when running the container:

```bash
docker run \
  -p 8000:8000 \
  -v "$(pwd)/artifacts:/app/artifacts" \
  supply-chain-late-delivery-api
```

The API will then be available on port:

```text
8000
```

---

# Automated Testing

The project includes tests for both preprocessing and API behavior.

Run:

```bash
python -m pytest -q
```

Current test suite:

```text
4 passed
```

The tests verify:

- leakage and identifier columns are removed
- order-date features are created
- constant features are removed
- unseen categorical values are handled correctly
- `/health` returns a successful response
- `/predict` follows the expected request/response contract

The API tests mock the trained model artifact so that automated CI does not require a large binary model file.

---

# Continuous Integration

GitHub Actions runs automatically on pushes and pull requests.

The CI workflow performs two independent jobs:

```text
test
 └── install dependencies
     └── run pytest

docker-build
 └── build Docker image
```

This helps detect:

- preprocessing regressions
- API regressions
- broken dependencies
- Docker build failures

before changes are integrated further.

---

# Data Science vs ML Engineering

This repository intentionally separates experimentation from production logic.

### Notebook

The notebook contains:

- exploratory analysis
- leakage investigation
- model comparison
- hyperparameter tuning
- threshold selection
- final test evaluation
- SHAP interpretation
- robustness analysis

### `src/`

The `src` package contains reusable application logic:

```text
preprocessing.py
train.py
predict.py
```

This prevents the production system from depending on notebook execution state.

---

# Key Engineering Principles

### Training / Serving Consistency

Preprocessing learned during training is persisted and reused during inference.

### Leakage Prevention

Future-information variables and direct target proxies are excluded before modeling.

### Group-Safe Evaluation

Orders, rather than individual line-item rows, determine dataset partition membership.

### Reproducibility

Training parameters, preprocessing rules, feature ordering, and prediction threshold are persisted with the model.

### Separation of Concerns

Training, preprocessing, inference, API serving, testing, and containerization are implemented as separate components.

---

# Limitations

### Random grouped split

The evaluation uses a random grouped split by `Order Id`, not a chronological split.

The test results therefore estimate performance on unseen orders drawn from a similar mixed-time population. They should not be interpreted as evidence of robustness to future temporal drift.

### Row-level model

The model is trained on line-item records.

A secondary order-level evaluation is performed by averaging line-item probabilities within each order.

### Feature availability

Although known leakage variables were removed, deployment in a real operational system would require verifying that every retained feature is actually available at the intended scoring moment.

### Correlated shipping features

`Shipping Mode` and `Days for shipment (scheduled)` encode closely related information, so feature-importance and SHAP attribution should not be treated as independent effects.

### Interpretation is not causality

SHAP describes how the trained model uses features. It does not establish that those variables cause late deliveries.

---

# Technology Stack

```text
Python
Pandas
Scikit-learn
Joblib
SHAP
FastAPI
Pydantic
Uvicorn
Pytest
Docker
GitHub Actions
Kaggle
```

---

# Project Outcome

This project demonstrates the complete transition from exploratory machine learning to a reproducible serving system:

```text
business problem
      ↓
leakage-safe experimentation
      ↓
group-safe evaluation
      ↓
model selection
      ↓
reusable preprocessing
      ↓
reproducible training
      ↓
model persistence
      ↓
inference module
      ↓
REST API
      ↓
automated tests
      ↓
Docker
      ↓
continuous integration
```

The final model achieves an untouched test ROC-AUC of **0.7710** and an order-level ROC-AUC of **0.7728**, while maintaining strict separation between training, validation, testing, and production inference.
