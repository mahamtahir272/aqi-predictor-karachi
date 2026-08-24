"""
Pearls AQI Predictor — Trends & Explainability page.

Shows the historical AQI trend and a SHAP feature-importance breakdown
for the current forecast.
"""

import streamlit as st
import matplotlib.pyplot as plt
import shap
from utils import load_everything_safely, make_forecast

st.set_page_config(page_title="Trends & Explainability — Pearls AQI Predictor", page_icon="📈", layout="wide")

st.title("📈 Trends & Explainability")
st.caption("Historical AQI patterns and what's driving the current forecast.")

date_range = st.selectbox(
    "Time range", ["Last 7 days", "Last 30 days", "Last 90 days", "All data"], index=1
)
range_days_map = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90, "All data": None}
days_back = range_days_map[date_range]

if days_back is None:
    st.info("Loading full history — this can take a little longer than the other ranges.")

model, feature_cols, model_version, df, error = load_everything_safely(days_back=days_back)

if error:
    st.error(f"Could not load model or data from Hopsworks: {error}")
    st.info("Make sure `HOPSWORKS_API_KEY` is set correctly in this app's Secrets.")
    st.stop()

latest_row, forecast_aqi = make_forecast(model, feature_cols, df)

# ---------------------------------------------------------------------
# Historical trend
# ---------------------------------------------------------------------
st.header("Historical AQI Trend")

trend_df = df[["timestamp", "aqi"]].copy()
st.line_chart(trend_df.set_index("timestamp"))

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Average AQI (selected range)", f"{trend_df['aqi'].mean():.0f}")
with col2:
    st.metric("Peak AQI (selected range)", f"{trend_df['aqi'].max():.0f}")
with col3:
    st.metric("Lowest AQI (selected range)", f"{trend_df['aqi'].min():.0f}")

st.divider()

# ---------------------------------------------------------------------
# Hourly pattern
# ---------------------------------------------------------------------
st.header("Average AQI by Hour of Day")
hourly_avg = df.groupby("hour")["aqi"].mean()
st.bar_chart(hourly_avg)
st.caption(
    "Shows which hours tend to have worse air quality on average — useful for planning "
    "outdoor activity around typical daily pollution cycles."
)

st.divider()

# ---------------------------------------------------------------------
# SHAP explainability
# ---------------------------------------------------------------------
st.header("🔍 Why This Forecast? (SHAP Feature Importance)")
st.caption(
    "Shows which features pushed the current 3-day-ahead forecast up or down. "
    "Only available for tree-based models (Random Forest); linear models (Ridge) "
    "are explained via their coefficients instead."
)

model_type = type(model).__name__

try:
    if "Forest" in model_type or "Tree" in model_type:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(latest_row)

        fig, ax = plt.subplots(figsize=(9, 6))
        shap.summary_plot(
            shap_values, latest_row, feature_names=feature_cols,
            plot_type="bar", show=False
        )
        st.pyplot(fig, clear_figure=True)

    elif "Ridge" in model_type or "Linear" in model_type:
        import pandas as pd
        coefs = pd.Series(model.coef_, index=feature_cols).sort_values()
        fig, ax = plt.subplots(figsize=(9, 6))
        coefs.plot(kind="barh", ax=ax, color="#1f77b4")
        ax.set_xlabel("Coefficient (impact on forecast)")
        ax.set_title(f"{model_type} feature coefficients")
        st.pyplot(fig, clear_figure=True)

    else:
        st.info(f"SHAP/coefficient explanation not implemented for model type: {model_type}")

except Exception as e:
    st.info(f"Explanation unavailable for this model: {e}")

st.divider()
st.caption(f"Currently deployed model: **{model_type}** (registry version {model_version})")
