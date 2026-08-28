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
