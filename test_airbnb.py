



import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def test_data_loading():
    df = pd.read_csv("airbnbListingsData.csv")
    assert not df.empty

def test_model_training():
    df = pd.read_csv("airbnbListingsData.csv")
    df = df.select_dtypes(include=[np.number])

    X = df.drop(columns=["price"])
    y = df["price"]

    X = X.fillna(0)
    y = y.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

def test_missing_values_handling():
    df = pd.read_csv("airbnbListingsData.csv")
    df = df.select_dtypes(include=[np.number])

    # Preprocessing
    X = df.drop(columns=["price"]).fillna(0)

    # Check no missing values
    assert not X.isnull().values.any(), "There are still missing values in features!"

def test_feature_types():
    df = pd.read_csv("airbnbListingsData.csv")
    df = df.select_dtypes(include=[np.number])

    X = df.drop(columns=["price"]).fillna(0)

    # Check all columns are numeric
    assert all(np.issubdtype(dtype, np.number) for dtype in X.dtypes), "Not all features are numeric!"

def test_no_leakage():
    df = pd.read_csv("airbnbListingsData.csv")
    df = df.select_dtypes(include=[np.number])

    X = df.drop(columns=["price"])

    # Check that 'price' is not mistakenly included in features
    assert "price" not in X.columns, "'price' found in features! Data leakage!"

# # pytest test_airbnb.py --verbose