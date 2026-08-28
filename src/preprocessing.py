import pandas as pd


LEAKAGE_COLUMNS = [
    "Delivery Status",
    "Days for shipping (real)",
    "Order Status",
    "shipping date (DateOrders)",
]


UNUSABLE_COLUMNS = [
    "Product Description",
    "Order Zipcode",
]


IDENTIFIER_COLUMNS = [
    "Customer Email",
    "Customer Fname",
    "Customer Lname",
    "Customer Password",
    "Customer Street",
    "Customer Id",
    "Order Id",
    "Order Customer Id",
    "Order Item Id",
    "Order Item Cardprod Id",
    "Product Card Id",
    "Category Id",
    "Department Id",
    "Product Category Id",
    "Product Image",
    "Customer Zipcode",
]


DROP_COLUMNS = (
    LEAKAGE_COLUMNS
    + UNUSABLE_COLUMNS
    + IDENTIFIER_COLUMNS
)


ORDER_DATE_COLUMN = "order date (DateOrders)"


def remove_unusable_columns(data):
    """
    Remove leakage, unusable, and identifier columns.
    """

    processed_data = data.copy()

    processed_data = processed_data.drop(
        columns=DROP_COLUMNS,
        errors="ignore"
    )

    return processed_data


def add_order_date_features(data):
    """
    Create month, day-of-week, and hour features
    from the order timestamp.
    """

    processed_data = data.copy()

    if ORDER_DATE_COLUMN not in processed_data.columns:
        raise ValueError(
            f"Missing required column: {ORDER_DATE_COLUMN}"
        )

    processed_data[ORDER_DATE_COLUMN] = pd.to_datetime(
        processed_data[ORDER_DATE_COLUMN]
    )

    processed_data["order_month"] = (
        processed_data[ORDER_DATE_COLUMN].dt.month
    )

    processed_data["order_dayofweek"] = (
        processed_data[ORDER_DATE_COLUMN].dt.dayofweek
    )

    processed_data["order_hour"] = (
        processed_data[ORDER_DATE_COLUMN].dt.hour
    )

    processed_data = processed_data.drop(
        columns=[ORDER_DATE_COLUMN]
    )

    return processed_data


def get_categorical_columns(data):
    """
    Return categorical feature names.
    """

    return (
        data
        .select_dtypes(include=["object", "string"])
        .columns
        .tolist()
    )


def fit_count_maps(data, categorical_columns):
    """
    Learn category-count mappings from training data only.
    """

    count_maps = {}

    for col in categorical_columns:
        count_maps[col] = (
            data[col]
            .value_counts()
            .to_dict()
        )

    return count_maps


def apply_count_encoding(data, count_maps):
    """
    Apply previously learned count mappings.

    Unseen categories are encoded as zero.
    """

    processed_data = data.copy()

    for col, mapping in count_maps.items():

        if col not in processed_data.columns:
            raise ValueError(
                f"Missing categorical column: {col}"
            )

        processed_data[col] = (
            processed_data[col]
            .map(mapping)
            .fillna(0)
        )

    return processed_data


def find_constant_features(data):
    """
    Find features with one or fewer unique values.
    """

    constant_features = [
        col
        for col in data.columns
        if data[col].nunique() <= 1
    ]

    return constant_features


def remove_constant_features(data, constant_features):
    """
    Remove constant features identified from training data.
    """

    processed_data = data.copy()

    processed_data = processed_data.drop(
        columns=constant_features,
        errors="ignore"
    )

    return processed_data


def fit_preprocessor(data):
    """
    Fit preprocessing rules using training data.

    Returns:
        processed_data
        preprocessing_artifacts
    """

    processed_data = remove_unusable_columns(data)

    processed_data = add_order_date_features(
        processed_data
    )

    categorical_columns = get_categorical_columns(
        processed_data
    )

    count_maps = fit_count_maps(
        processed_data,
        categorical_columns
    )

    processed_data = apply_count_encoding(
        processed_data,
        count_maps
    )

    constant_features = find_constant_features(
        processed_data
    )

    processed_data = remove_constant_features(
        processed_data,
        constant_features
    )

    feature_columns = processed_data.columns.tolist()

    preprocessing_artifacts = {
        "categorical_columns": categorical_columns,
        "count_maps": count_maps,
        "constant_features": constant_features,
        "feature_columns": feature_columns,
    }

    return processed_data, preprocessing_artifacts


def transform_features(data, preprocessing_artifacts):
    """
    Transform validation, test, or new production data
    using preprocessing rules learned from training data.
    """

    processed_data = remove_unusable_columns(data)

    processed_data = add_order_date_features(
        processed_data
    )

    processed_data = apply_count_encoding(
        processed_data,
        preprocessing_artifacts["count_maps"]
    )

    processed_data = remove_constant_features(
        processed_data,
        preprocessing_artifacts["constant_features"]
    )

    expected_columns = preprocessing_artifacts[
        "feature_columns"
    ]

    missing_columns = [
        col
        for col in expected_columns
        if col not in processed_data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required model features: "
            f"{missing_columns}"
        )

    processed_data = processed_data[
        expected_columns
    ]

    return processed_data
