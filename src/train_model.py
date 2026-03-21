import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def train_models(X, y):
    # split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training models...")

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # XGBoost
    xgb = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)

    # predictions
    rf_preds = rf.predict(X_val)
    xgb_preds = xgb.predict(X_val)

    # RMSE
    rf_rmse = np.sqrt(mean_squared_error(y_val, rf_preds))
    xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_preds))

    print(f"RF RMSE: {rf_rmse}")
    print(f"XGB RMSE: {xgb_rmse}")

    return rf, xgb


def save_models(rf, xgb):
    # force save to project root /models
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(BASE_DIR, "models")

    os.makedirs(model_path, exist_ok=True)

    joblib.dump(rf, os.path.join(model_path, "rf_model.pkl"))
    joblib.dump(xgb, os.path.join(model_path, "xgb_model.pkl"))

    print("Models saved at:", model_path)


def load_models():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(BASE_DIR, "models")

    rf = joblib.load(os.path.join(model_path, "rf_model.pkl"))
    xgb = joblib.load(os.path.join(model_path, "xgb_model.pkl"))

    return rf, xgb