"""
Shared utilities for the Pearls AQI Predictor dashboard.
Imported by app.py and every page in pages/.
"""

import streamlit as st
import pandas as pd
import joblib
import hopsworks

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
HOPSWORKS_PROJECT = "aqi_predictor_maham"
HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"
CITY_NAME = "Karachi"

FEATURE_COLS = [
    "pm2_5", "pm10", "no2", "o3", "co", "so2", "temp", "humidity",
    "pressure", "wind_speed", "hour", "day", "month", "day_of_week",
    "aqi", "aqi_change_rate", "aqi_lag_1", "aqi_rolling_mean_3",
]

AQI_LEVELS = [
    (0, 50, "Good", "#00e400"),
    (51, 100, "Moderate", "#dddd00"),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff7e00"),
    (151, 200, "Unhealthy", "#ff0000"),
    (201, 300, "Very Unhealthy", "#8f3f97"),
    (301, 500, "Hazardous", "#7e0023"),
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def aqi_level(aqi):
    for lo, hi, label, color in AQI_LEVELS:
        if lo <= aqi <= hi:
            return label, color
    return "Hazardous", "#7e0023"


@st.cache_resource(show_spinner="Connecting to Hopsworks...")
def get_project():
    return hopsworks.login(
        project=HOPSWORKS_PROJECT,
        host=HOPSWORKS_HOST,
        port=443,
        api_key_value=st.secrets["HOPSWORKS_API_KEY"],
    )


@st.cache_resource(show_spinner="Loading latest model from the registry...")
def load_model():
    project = get_project()
    mr = project.get_model_registry()
    model_meta = mr.get_model("aqi_forecast_model")  # latest version by default
    model_dir = model_meta.download()
    model = joblib.load(f"{model_dir}/model.pkl")
    try:
        feature_cols = joblib.load(f"{model_dir}/feature_cols.pkl")
    except FileNotFoundError:
        feature_cols = FEATURE_COLS
    return model, feature_cols, model_meta.version


@st.cache_data(ttl=1800, show_spinner="Fetching latest feature data...")
def load_features():
    project = get_project()
    fs = project.get_feature_store()
    aqi_fg = fs.get_feature_group(name="aqi_features", version=1)
    df = aqi_fg.read()
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def make_forecast(model, feature_cols, df):
    latest_row = df.iloc[[-1]][feature_cols]
    pred = model.predict(latest_row)[0]
    return latest_row, float(pred)


def render_alert(aqi_value, label):
    if aqi_value >= 151:
        st.error(
            f"⚠️ **Hazardous air quality alert** — forecasted AQI is **{aqi_value:.0f} "
            f"({label})**. Sensitive groups should avoid outdoor activity; consider masks "
            f"and air purifiers indoors."
        )
    elif aqi_value >= 101:
        st.warning(
            f"**Caution** — forecasted AQI is **{aqi_value:.0f} ({label})**. "
            f"Sensitive groups (children, elderly, respiratory conditions) should limit "
            f"prolonged outdoor exertion."
        )
    else:
        st.success(f"Forecasted AQI is **{aqi_value:.0f} ({label})** — no alert needed.")


def load_everything_safely():
    """
    Convenience wrapper used by dashboard pages. Returns (model, feature_cols,
    model_version, df, error) — error is None on success, or a string on failure,
    so pages can render a friendly message instead of crashing.
    """
    try:
        model, feature_cols, model_version = load_model()
        df = load_features()
        return model, feature_cols, model_version, df, None
    except Exception as e:
        return None, None, None, None, str(e)
