"""
Pearls AQI Predictor — Live Forecast page.

Shows current AQI, the 3-day-ahead forecast, and a hazardous-AQI alert banner.
"""

import streamlit as st
from utils import (
    load_everything_safely,
    make_forecast,
    aqi_level,
    render_alert,
    AQI_LEVELS,
)

st.set_page_config(page_title="Live Forecast — Pearls AQI Predictor", page_icon="📊", layout="wide")

st.title("📊 Live Forecast")
st.caption("Current AQI and 3-day-ahead prediction for Karachi.")

model, feature_cols, model_version, df, error = load_everything_safely()

if error:
    st.error(f"Could not load model or data from Hopsworks: {error}")
    st.info("Make sure `HOPSWORKS_API_KEY` is set correctly in this app's Secrets.")
    st.stop()

latest_row, forecast_aqi = make_forecast(model, feature_cols, df)
current_aqi = df.iloc[-1]["aqi"]
last_updated = df.iloc[-1]["timestamp"]

label, _ = aqi_level(forecast_aqi)
current_label, _ = aqi_level(current_aqi)

# --- Top metrics row ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Current AQI", f"{current_aqi:.0f}", help=current_label)
with col2:
    st.metric(
        "Forecast AQI (+72h)",
        f"{forecast_aqi:.0f}",
        delta=f"{forecast_aqi - current_aqi:+.0f}",
        help=label,
    )
with col3:
    st.metric("Model version", model_version)

st.caption(f"Data last updated: {last_updated} UTC")

st.divider()

# --- Alert banner ---
render_alert(forecast_aqi, label)

st.divider()

# --- AQI scale reference ---
st.subheader("📋 AQI Scale Reference")
scale_cols = st.columns(len(AQI_LEVELS))
for col, (lo, hi, lvl_label, color) in zip(scale_cols, AQI_LEVELS):
    with col:
        st.markdown(
            f"""
            <div style="background-color:{color}; padding:10px; border-radius:8px; text-align:center;">
                <b>{lo}-{hi}</b><br>{lvl_label}
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

# --- Current readings breakdown ---
st.subheader("🔬 Current Readings")
reading_cols = st.columns(4)
readings = [
    ("PM2.5", f"{df.iloc[-1]['pm2_5']:.1f} µg/m³"),
    ("PM10", f"{df.iloc[-1]['pm10']:.1f} µg/m³"),
    ("Temperature", f"{df.iloc[-1]['temp']:.1f} °C"),
    ("Humidity", f"{df.iloc[-1]['humidity']:.0f}%"),
]
for col, (name, value) in zip(reading_cols, readings):
    with col:
        st.metric(name, value)
