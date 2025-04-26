# test_random_forest.py

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def test_data_loading():
    df = pd.read_csv("airbnbListingsData.csv")
    assert not df.empty

def test_model_training():
    df = pd.read_csv("airbnbListingsData.csv")

    # Only keep numeric columns
    df = df.select_dtypes(include=[np.number])

    X = df.drop(columns=["price"])
    y = df["price"]

    X = X.fillna(0)
    y = y.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)


    preds = model.predict(X_test)
    score = r2_score(y_test, preds)
    
    assert score > 0, f"Model performance too low! R2: {score}"

if __name__ == "__main__":
    test_data_loading()
    test_model_training()
    print("All tests passed!")

# pytest test_airbnb.py --verbose
