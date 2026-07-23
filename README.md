# Late Delivery Risk Prediction in Smart Supply Chains

Predicting whether an e-commerce order will be delivered late before it is dispatched, using the DataCo Smart Supply Chain dataset (180,519 orders, 53 features).

## Problem

Late deliveries are costly and hard to anticipate. The goal was to build a classifier that flags high-risk orders *before dispatch*, so operational teams can intervene early. The core modeling challenge: **data leakage**: many columns in the raw data only become known *after* delivery, and using them inflates metrics while making the model useless in production.

## Dataset

- **180,519 orders**, **53 features** across order, shipment, customer, product, and location attributes
- Target: `Late_delivery_risk` (binary — late vs. on-time)
- Source: [DataCo Smart Supply Chain (Kaggle)](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)

## Approach

**Data preprocessing**
- Dropped near-empty columns (`Product Description` 100% null, `Order Zipcode` 86% null)
- Removed **leakage columns** : anything unknowable at order time: `Delivery Status`, `Days for shipping (real)`, `Order Status`, and the raw shipping date
- Removed pure identifiers (customer names, emails, IDs, image URLs)
- Rule applied throughout: *"Would I have this value at prediction time?"* If no → dropped.

**Feature engineering**
- Converted date strings to datetime; extracted `order_dayofweek` and `order_month` from the order date (known at prediction time)
- **Frequency encoding** for all 14 categorical columns which replaced each category with its occurrence count to handle high-cardinality features (e.g. `Order City`: 3,597 unique values) without one-hot explosion

**Modeling**
- 80:20 train/test split
- Compared XGBoost against Logistic Regression (SGD and standard solvers), with `StandardScaler` applied for the linear models (tree models don't require scaling; linear models do; unscaled SGD scored 61.7%, scaled 68.7%)

## Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| **XGBoost** | **72.9%** | **0.813** |
| Logistic Regression (scaled) | 68.7% | 0.687 |
| Logistic Regression SGD (scaled) | 68.7% | 0.688 |

Tree-based boosting clearly outperformed linear models, indicating the drivers of late delivery are **non-linear and interaction-heavy**.

## Feature Importance (SHAP)

- **`Days for shipment (scheduled)`** dominates: orders with *tight* scheduled windows are the ones that go late — short promises get missed, generous ones get met
- Location features (latitude/longitude, order city/state/country) contribute moderately
- Price and discount features contribute little, validating the strategy of keeping plausible features initially and letting SHAP judge, rather than dropping on intuition

## Tech Stack

Python · pandas · scikit-learn · XGBoost · SHAP

---

*Originally developed as a graduate project at the University of Texas at Arlington; rebuilt end-to-end with a focus on leakage-free feature design.*
