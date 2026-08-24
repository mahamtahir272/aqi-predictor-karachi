"""
Pearls AQI Predictor — Streamlit Dashboard

Loads the latest registered model + features from Hopsworks and shows:
  - Current AQI + 3-day-ahead forecast
  - Hazardous AQI alert banner
  - Historical AQI trend chart
  - SHAP feature-importance explanation for the current prediction

Deploy on Streamlit Community Cloud with these secrets configured
(Settings -> Secrets, in TOML format):

    HOPSWORKS_API_KEY = "..."
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
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
    (51, 100, "Moderate", "#ffff00"),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff7e00"),
    (151, 200, "Unhealthy", "#ff0000"),
    (201, 300, "Very Unhealthy", "#8f3f97"),
    (301, 500, "Hazardous", "#7e0023"),
]

st.set_page_config(page_title="Karachi AQI Predictor", page_icon="🌫️", layout="wide")


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


# ---------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------
st.title("🌫️ Pearls AQI Predictor — Karachi")
st.caption("3-day-ahead Air Quality Index forecast, powered by a serverless Hopsworks + scikit-learn pipeline.")

try:
    model, feature_cols, model_version = load_model()
    df = load_features()
except Exception as e:
    st.error(f"Could not load model or data from Hopsworks: {e}")
    st.stop()

latest_row, forecast_aqi = make_forecast(model, feature_cols, df)
current_aqi = df.iloc[-1]["aqi"]
last_updated = df.iloc[-1]["timestamp"]

label, color = aqi_level(forecast_aqi)
current_label, current_color = aqi_level(current_aqi)

# --- Top metrics row ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Current AQI", f"{current_aqi:.0f}", help=current_label)
with col2:
    st.metric("Forecast AQI (+72h)", f"{forecast_aqi:.0f}", delta=f"{forecast_aqi - current_aqi:+.0f}")
with col3:
    st.metric("Model version", model_version)

st.caption(f"Data last updated: {last_updated} UTC")

# --- Alert banner ---
render_alert(forecast_aqi, label)

st.divider()

# --- Historical trend chart ---
st.subheader("📈 Historical AQI Trend")
trend_df = df[["timestamp", "aqi"]].copy()
st.line_chart(trend_df.set_index("timestamp"))

st.divider()

# --- SHAP explanation ---
st.subheader("🔍 Why this forecast? (SHAP feature importance)")
st.caption("Shows which features pushed today's 3-day forecast up or down.")

try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(latest_row)

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(
        shap_values, latest_row, feature_names=feature_cols,
        plot_type="bar", show=False
    )
    st.pyplot(fig, clear_figure=True)
except Exception as e:
    st.info(f"SHAP explanation unavailable for this model type: {e}")

st.divider()
st.caption("Pearls AQI Predictor — internship project. Data: OpenWeather (live) + Open-Meteo (historical).")
