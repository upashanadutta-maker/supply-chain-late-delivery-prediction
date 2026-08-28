import argparse
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.preprocessing import (
    fit_preprocessor,
    transform_features,
)


TARGET_COLUMN = "Late_delivery_risk"
GROUP_COLUMN = "Order Id"

SELECTED_THRESHOLD = 0.40

MODEL_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.05,
    "max_depth": 3,
    "random_state": 42,
}


def load_data(data_path):
    """
    Load the raw supply-chain dataset.
    """

    print(f"Loading data from: {data_path}")

    data = pd.read_csv(
        data_path,
        encoding="latin-1"
    )

    print(f"Dataset shape: {data.shape}")

    return data


def split_by_order(data):
    """
    Create 60/20/20 train, validation, and test
    partitions using Order Id as the grouping unit.
    """

    order_level = (
        data[[GROUP_COLUMN, TARGET_COLUMN]]
        .drop_duplicates()
    )

    train_orders, temp_orders = train_test_split(
        order_level,
        test_size=0.40,
        random_state=42,
        stratify=order_level[TARGET_COLUMN],
    )

    val_orders, test_orders = train_test_split(
        temp_orders,
        test_size=0.50,
        random_state=42,
        stratify=temp_orders[TARGET_COLUMN],
    )

    train_data = data[
        data[GROUP_COLUMN].isin(
            train_orders[GROUP_COLUMN]
        )
    ].copy()

    val_data = data[
        data[GROUP_COLUMN].isin(
            val_orders[GROUP_COLUMN]
        )
    ].copy()

    test_data = data[
        data[GROUP_COLUMN].isin(
            test_orders[GROUP_COLUMN]
        )
    ].copy()

    return train_data, val_data, test_data


def separate_features_and_target(data):
    """
    Separate target from raw model features.
    """

    target = data[TARGET_COLUMN].copy()

    features = data.drop(
        columns=[TARGET_COLUMN]
    ).copy()

    return features, target


def evaluate_model(
    model,
    features,
    target,
    threshold
):
    """
    Evaluate model probabilities and predictions.
    """

    probabilities = model.predict_proba(
        features
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(
            target,
            probabilities
        ),
        "accuracy": accuracy_score(
            target,
            predictions
        ),
        "precision": precision_score(
            target,
            predictions
        ),
        "recall": recall_score(
            target,
            predictions
        ),
        "f1": f1_score(
            target,
            predictions
        ),
    }

    return metrics, probabilities


def print_metrics(name, metrics):
    """
    Print evaluation metrics.
    """

    print(f"\n{name}")
    print("-" * len(name))

    print(
        f"ROC-AUC:   {metrics['roc_auc']:.4f}"
    )
    print(
        f"Accuracy:  {metrics['accuracy']:.4f}"
    )
    print(
        f"Precision: {metrics['precision']:.4f}"
    )
    print(
        f"Recall:    {metrics['recall']:.4f}"
    )
    print(
        f"F1-score:  {metrics['f1']:.4f}"
    )


def evaluate_order_level(
    test_data,
    probabilities,
    threshold
):
    """
    Perform secondary evaluation with one
    prediction per Order Id.
    """

    order_results = pd.DataFrame({
        GROUP_COLUMN: test_data[GROUP_COLUMN].values,
        "actual": test_data[TARGET_COLUMN].values,
        "probability": probabilities,
    })

    order_results = (
        order_results
        .groupby(GROUP_COLUMN)
        .agg(
            actual=("actual", "first"),
            probability=("probability", "mean"),
        )
        .reset_index()
    )

    order_results["prediction"] = (
        order_results["probability"]
        >= threshold
    ).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(
            order_results["actual"],
            order_results["probability"]
        ),
        "accuracy": accuracy_score(
            order_results["actual"],
            order_results["prediction"]
        ),
        "precision": precision_score(
            order_results["actual"],
            order_results["prediction"]
        ),
        "recall": recall_score(
            order_results["actual"],
            order_results["prediction"]
        ),
        "f1": f1_score(
            order_results["actual"],
            order_results["prediction"]
        ),
    }

    return metrics


def save_model_bundle(
    model,
    preprocessing_artifacts,
    artifact_path,
):
    """
    Save model and preprocessing rules together.
    """

    bundle = {
        "model": model,
        "preprocessing_artifacts": (
            preprocessing_artifacts
        ),
        "threshold": SELECTED_THRESHOLD,
        "model_params": MODEL_PARAMS,
        "target_column": TARGET_COLUMN,
    }

    artifact_path = Path(artifact_path)

    artifact_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        bundle,
        artifact_path
    )

    print(
        f"\nModel bundle saved to: "
        f"{artifact_path}"
    )


def main(data_path, artifact_path):
    """
    Run the complete training pipeline.
    """

    data = load_data(data_path)

    train_data, val_data, test_data = (
        split_by_order(data)
    )

    print("\nSplit sizes:")
    print(f"Train rows:      {len(train_data)}")
    print(f"Validation rows: {len(val_data)}")
    print(f"Test rows:       {len(test_data)}")

    X_train_raw, y_train = (
        separate_features_and_target(
            train_data
        )
    )

    X_val_raw, y_val = (
        separate_features_and_target(
            val_data
        )
    )

    X_test_raw, y_test = (
        separate_features_and_target(
            test_data
        )
    )

    X_train, preprocessing_artifacts = (
        fit_preprocessor(
            X_train_raw
        )
    )

    X_val = transform_features(
        X_val_raw,
        preprocessing_artifacts
    )

    X_test = transform_features(
        X_test_raw,
        preprocessing_artifacts
    )

    print("\nProcessed feature shapes:")
    print(f"Train:      {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test:       {X_test.shape}")

    model = GradientBoostingClassifier(
        **MODEL_PARAMS
    )

    print("\nTraining Gradient Boosting model...")

    model.fit(
        X_train,
        y_train
    )

    validation_metrics, _ = evaluate_model(
        model,
        X_val,
        y_val,
        SELECTED_THRESHOLD
    )

    print_metrics(
        "VALIDATION RESULTS",
        validation_metrics
    )

    test_metrics, test_probabilities = (
        evaluate_model(
            model,
            X_test,
            y_test,
            SELECTED_THRESHOLD
        )
    )

    print_metrics(
        "FINAL TEST RESULTS",
        test_metrics
    )

    order_metrics = evaluate_order_level(
        test_data,
        test_probabilities,
        SELECTED_THRESHOLD
    )

    print_metrics(
        "ORDER-LEVEL TEST RESULTS",
        order_metrics
    )

    save_model_bundle(
        model,
        preprocessing_artifacts,
        artifact_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Train the late-delivery risk model."
        )
    )

    parser.add_argument(
        "--data-path",
        default="data/DataCoSupplyChainDataset.csv",
        help="Path to the raw CSV dataset.",
    )

    parser.add_argument(
        "--artifact-path",
        default="artifacts/model_bundle.joblib",
        help="Location for the saved model bundle.",
    )

    args = parser.parse_args()

    main(
        data_path=args.data_path,
        artifact_path=args.artifact_path,
    )
