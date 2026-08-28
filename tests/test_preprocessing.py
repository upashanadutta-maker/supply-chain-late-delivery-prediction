import pandas as pd

from src.preprocessing import (
    fit_preprocessor,
    transform_features,
)


def test_preprocessing_pipeline():

    train_data = pd.DataFrame({
        "Type": [
            "DEBIT",
            "CASH",
            "DEBIT",
        ],
        "Shipping Mode": [
            "Standard Class",
            "First Class",
            "Standard Class",
        ],
        "order date (DateOrders)": [
            "1/1/2018 10:00",
            "2/2/2018 15:00",
            "3/3/2018 20:00",
        ],
        "Delivery Status": [
            "Late delivery",
            "Advance shipping",
            "Late delivery",
        ],
        "Order Id": [1, 2, 3],
        "Product Status": [0, 0, 0],
    })

    processed_train, artifacts = fit_preprocessor(
        train_data
    )

    assert "Delivery Status" not in processed_train.columns
    assert "Order Id" not in processed_train.columns
    assert "order date (DateOrders)" not in processed_train.columns

    assert "order_month" in processed_train.columns
    assert "order_dayofweek" in processed_train.columns
    assert "order_hour" in processed_train.columns

    assert "Product Status" not in processed_train.columns


def test_unseen_category():

    train_data = pd.DataFrame({
        "Type": [
            "DEBIT",
            "DEBIT",
            "CASH",
        ],
        "Shipping Mode": [
            "Standard Class",
            "Standard Class",
            "First Class",
        ],
        "order date (DateOrders)": [
            "1/1/2018 10:00",
            "2/2/2018 15:00",
            "3/3/2018 20:00",
        ],
    })

    new_data = pd.DataFrame({
        "Type": ["TRANSFER"],
        "Shipping Mode": ["Same Day"],
        "order date (DateOrders)": [
            "4/4/2018 12:00"
        ],
    })

    _, artifacts = fit_preprocessor(
        train_data
    )

    processed_new = transform_features(
        new_data,
        artifacts
    )

    assert processed_new["Type"].iloc[0] == 0
    assert processed_new["Shipping Mode"].iloc[0] == 0
