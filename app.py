"""
Pearls AQI Predictor — Home / About page.
"""

import streamlit as st

st.set_page_config(page_title="Pearls AQI Predictor", page_icon="🌫️", layout="wide")

# ---------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------
st.title("🌫️ Pearls AQI Predictor")
st.subheader("Forecasting Karachi's Air Quality Index, 3 days ahead — on a 100% serverless stack")

st.markdown(
    """
This project predicts the **Air Quality Index (AQI)** for Karachi over the next
**72 hours**, using an end-to-end machine learning pipeline: automated data collection,
feature engineering, model training, and real-time predictions through this web dashboard.

Use the sidebar to navigate to:
- **📊 Live Forecast** — current AQI, 3-day forecast, and hazardous-air alerts
- **📈 Trends & Explainability** — historical AQI trends and SHAP feature-importance analysis
"""
)

st.divider()

# ---------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------
st.header("🏗️ Architecture")

st.markdown(
    """
```
Weather & Pollution API ──(hourly)──► Feature Pipeline ──► Feature Store
      (OpenWeather)                    (GitHub Actions)    (Hopsworks)
                                                                  │
                                                          (daily) ▼
                                                          Training Pipeline
                                                          (GitHub Actions)
                                                                  │
                                                                  ▼
                                                           Model Registry
                                                            (Hopsworks)
                                                                  │
                                                                  ▼
                                                          Web App (this dashboard)
                                                             (Streamlit)
```

The pipeline runs continuously and unattended:
1. **Feature Pipeline** fetches live weather + pollution data every hour and writes
   engineered features into the Hopsworks Feature Store.
2. **Training Pipeline** re-trains and re-evaluates candidate models daily, and registers
   whichever performs best in the Hopsworks Model Registry.
3. **This dashboard** always reads the *latest* registered model and *latest* features —
   no manual redeployment needed when the model updates.
"""
)

st.divider()

# ---------------------------------------------------------------------
# Tech stack
# ---------------------------------------------------------------------
st.header("🛠️ Technology Stack")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        """
        **Data & Features**
        - Python
        - OpenWeather API (live)
        - Open-Meteo API (historical backfill)
        - Hopsworks Feature Store
        """
    )
with col2:
    st.markdown(
        """
        **Modeling**
        - Scikit-learn (Ridge, Random Forest)
        - MLP Neural Network
        - Time-series cross-validation
        - SHAP (explainability)
        """
    )
with col3:
    st.markdown(
        """
        **Automation & Serving**
        - GitHub Actions (CI/CD)
        - Hopsworks Model Registry
        - Streamlit (this dashboard)
        - Git / GitHub
        """
    )

st.divider()

# ---------------------------------------------------------------------
# Key features
# ---------------------------------------------------------------------
st.header("✨ Key Features")

st.markdown(
    """
- **Feature Pipeline** — fetches raw weather + pollutant data hourly; computes time-based
  features (hour, day, month, day-of-week) and derived features (AQI change rate,
  lag, rolling average)
- **Historical Backfill** — a full year of hourly Karachi data, giving the models complete
  seasonal cycles to learn from
- **Training Pipeline** — evaluates Ridge Regression, Random Forest, and an MLP neural
  network using **time-series cross-validation** (RMSE, MAE, R²), and registers the best
  performer automatically
- **Automated CI/CD** — feature pipeline runs hourly, training pipeline runs daily, both
  via GitHub Actions
- **Explainable Forecasts** — SHAP feature-importance analysis shows *why* the model
  predicted what it did
- **Hazardous AQI Alerts** — color-coded banners warn when forecasted air quality crosses
  unhealthy thresholds
"""
)

st.divider()
st.caption(
    "Pearls AQI Predictor — internship project. "
    "Data sources: OpenWeather (live) + Open-Meteo (historical). "
    "Model: Hopsworks Model Registry."
)
