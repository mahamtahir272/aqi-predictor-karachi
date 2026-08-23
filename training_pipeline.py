"""
Training Pipeline — runs daily via GitHub Actions.

Pulls historical (features, target) from the Hopsworks Feature Store,
trains Ridge + Random Forest, evaluates with time-series cross-validation,
and registers the best-performing model in the Hopsworks Model Registry.

Required environment variables (set as GitHub Secrets):
    HOPSWORKS_API_KEY
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone

import hopsworks
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
HOPSWORKS_PROJECT = "aqi_predictor_maham"
HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"
HOPSWORKS_API_KEY = os.environ["HOPSWORKS_API_KEY"]

FORECAST_HOURS = 72  # predict AQI 3 days ahead
N_CV_SPLITS = 5

FEATURE_COLS = [
    "pm2_5", "pm10", "no2", "o3", "co", "so2", "temp", "humidity",
    "pressure", "wind_speed", "hour", "day", "month", "day_of_week",
    "aqi", "aqi_change_rate", "aqi_lag_1", "aqi_rolling_mean_3",
]

MODEL_DIR = "aqi_model"


# ---------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------
def load_training_data(fs):
    aqi_fg = fs.get_feature_group(name="aqi_features", version=1)
    df = aqi_fg.read()
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["aqi_target_72h"] = df["aqi"].shift(-FORECAST_HOURS)
    df = df.dropna(subset=["aqi_target_72h"]).reset_index(drop=True)

    X = df[FEATURE_COLS]
    y = df["aqi_target_72h"]
    return X, y


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
def evaluate_with_cv(X, y, n_splits=N_CV_SPLITS):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = {"Ridge": [], "RandomForest": []}

    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        ridge = Ridge(alpha=1.0).fit(X_tr_s, y_tr)
        ridge_pred = ridge.predict(X_te_s)
        scores["Ridge"].append(r2_score(y_te, ridge_pred))

        rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        rf_pred = rf.predict(X_te)
        scores["RandomForest"].append(r2_score(y_te, rf_pred))

    mean_scores = {name: float(np.mean(vals)) for name, vals in scores.items()}
    return mean_scores


def train_final_model(X, y, model_name):
    """Train the winning model type on ALL available data for production use."""
    if model_name == "Ridge":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = Ridge(alpha=1.0).fit(X_scaled, y)
        preds = model.predict(X_scaled)
    else:
        model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X, y)
        preds = model.predict(X)

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y, preds))),
        "mae": float(mean_absolute_error(y, preds)),
        "r2": float(r2_score(y, preds)),
    }
    return model, metrics


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print(f"[{datetime.now(timezone.utc)}] Starting training pipeline run...")

    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        host=HOPSWORKS_HOST,
        port=443,
        api_key_value=HOPSWORKS_API_KEY,
    )
    fs = project.get_feature_store()

    X, y = load_training_data(fs)
    print(f"Loaded {len(X)} training rows.")

    if len(X) < 200:
        print("Not enough data yet for a meaningful training run — skipping.")
        return

    cv_scores = evaluate_with_cv(X, y)
    print("Cross-validated mean R2:", cv_scores)

    best_model_name = max(cv_scores, key=cv_scores.get)
    print(f"Best model by CV: {best_model_name} (R2={cv_scores[best_model_name]:.3f})")

    model, metrics = train_final_model(X, y, best_model_name)
    print("Final in-sample metrics:", metrics)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(FEATURE_COLS, os.path.join(MODEL_DIR, "feature_cols.pkl"))
    joblib.dump(best_model_name, os.path.join(MODEL_DIR, "model_type.pkl"))

    mr = project.get_model_registry()
    aqi_model = mr.python.create_model(
        name="aqi_forecast_model",
        metrics={**metrics, "cv_r2": cv_scores[best_model_name]},
        description=f"{best_model_name} regressor predicting AQI 72h ahead for Karachi (auto-trained)",
        input_example=X.iloc[[0]],
    )
    aqi_model.save(MODEL_DIR)
    print(f"Model registered: {aqi_model.name} version {aqi_model.version}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: training pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)
