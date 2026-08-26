# 🌫️ Pearls AQI Predictor — Karachi

An end-to-end, **100% serverless** machine learning system that forecasts the Air Quality
Index (AQI) for Karachi, Pakistan, **72 hours (3 days) ahead**. Data collection, feature
engineering, model training, and serving all run automatically on a schedule — no manual
intervention required.

**🔗 Live dashboard:** https://aqi-predictor-karachi.streamlit.app
**📦 Repository:** https://github.com/mahamtahir272/aqi-predictor-karachi

---

## Table of Contents

- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [How It Works](#how-it-works)
  - [1. Feature Pipeline](#1-feature-pipeline)
  - [2. Historical Backfill](#2-historical-backfill)
  - [3. Training Pipeline](#3-training-pipeline)
  - [4. Automated CI/CD](#4-automated-cicd)
  - [5. Web Dashboard](#5-web-dashboard)
- [Results Summary](#results-summary)
- [Setup Guide](#setup-guide)
- [Local Development](#local-development)
- [Design Decisions & Known Deviations](#design-decisions--known-deviations)
- [Limitations & Future Work](#limitations--future-work)
- [Troubleshooting](#troubleshooting)

---

## Architecture

```
Weather & Pollution API ──(hourly)──► Feature Pipeline ──► Hopsworks Feature Store
      (OpenWeather)                   (GitHub Actions)
                                                                  │
                                                          (daily) ▼
                                                          Training Pipeline
                                                          (GitHub Actions)
                                                                  │
                                                                  ▼
                                                          Hopsworks Model Registry
                                                                  │
                                                                  ▼
                                                          Streamlit Web App
                                                     (About / Live Forecast / Trends)
```

The pipeline runs continuously and unattended:

1. **Feature Pipeline** fetches live weather + pollution data every hour and writes
   engineered features into the Hopsworks Feature Store.
2. **Training Pipeline** re-trains and re-evaluates candidate models daily using 5-fold
   time-series cross-validation, and registers whichever model currently performs best.
3. **The dashboard** always reads the *latest* registered model and *latest* features at
   runtime — no redeployment needed when the model updates.

---

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Live data source | [OpenWeather API](https://openweathermap.org/api) (Air Pollution + Current Weather) |
| Historical data source | [Open-Meteo API](https://open-meteo.com/) (Air Quality + Weather Archive) |
| Feature Store & Model Registry | [Hopsworks](https://www.hopsworks.ai/) (free tier) |
| Modeling | scikit-learn (Ridge Regression, Random Forest), MLPRegressor (neural net) |
| Explainability | SHAP (TreeExplainer for Random Forest; coefficients for Ridge) |
| Automation / CI-CD | GitHub Actions (hourly + daily scheduled workflows) |
| Web application | Streamlit, deployed on Streamlit Community Cloud |
| Version control | Git / GitHub |

> **Note:** AQICN and TensorFlow were part of the original suggested stack but were
> substituted — see [Design Decisions & Known Deviations](#design-decisions--known-deviations)
> for why.

---

## Repository Structure

```
aqi-predictor-karachi/
├── app.py                              # Streamlit entry point — Home/About page
├── utils.py                            # Shared Hopsworks/model/data-loading logic
├── pages/
│   ├── 1_Live_Forecast.py              # Current AQI, 3-day forecast, hazard alerts
│   └── 2_Trends_and_Explainability.py  # Historical trends + SHAP explainability
├── feature_pipeline.py                 # Hourly script: fetch → engineer → write to Feature Store
├── training_pipeline.py                # Daily script: train, cross-validate, register best model
├── requirements.txt                    # Dependencies for the Streamlit app (no Kafka)
├── requirements-pipelines.txt          # Dependencies for the GitHub Actions scripts (needs Kafka)
├── runtime.txt                         # Python version pin (informational; set via Streamlit Cloud UI)
├── .github/workflows/
│   ├── feature_pipeline.yml            # Hourly cron trigger
│   └── training_pipeline.yml           # Daily cron trigger
└── README.md
```

---

## How It Works

### 1. Feature Pipeline

`feature_pipeline.py` runs **hourly** via GitHub Actions. Each run:

- Fetches current pollutant concentrations (PM2.5, PM10, NO2, O3, CO, SO2, NH3) and
  weather conditions (temperature, humidity, pressure, wind speed) for Karachi from
  OpenWeather.
- Computes AQI from PM2.5 using the standard **US EPA breakpoint formula** (0–500 scale)
  rather than OpenWeather's coarser 1–5 index.
- Derives time-based features: `hour`, `day`, `month`, `day_of_week`.
- Derives change features: `aqi_change_rate`, `pm2_5_change_rate`, `aqi_lag_1`, and a
  rolling mean — computed against the most recently stored row (a live run only has one
  new data point to compare against).
- Writes the row into the Hopsworks Feature Store (feature group `aqi_features`, version 1).

A key engineering detail: **the feature group's schema is fixed by its first insert.**
Every subsequent insert — whether from OpenWeather (live) or Open-Meteo (backfill) — must
match that exact schema (column names, types, including a `float64` requirement on the
derived columns). `align_schema_for_hopsworks()` in `feature_pipeline.py` handles this
consistently.

### 2. Historical Backfill

The Feature Store was backfilled with **a full year of hourly data** (8,784 rows,
Aug 2025 – Aug 2026) from Open-Meteo's free historical archive. Earlier attempts with
smaller windows (7 and 90 days) produced misleadingly poor model performance because the
train/test split landed across different seasonal regimes — see
[Results Summary](#results-summary) below.

### 3. Training Pipeline

`training_pipeline.py` runs **daily** via GitHub Actions. It:

1. Pulls all historical (features, target) rows from the Feature Store.
2. Reframes the problem as predicting AQI **72 hours ahead** of each row's timestamp.
3. Evaluates **Ridge Regression** and **Random Forest** using 5-fold
   `TimeSeriesSplit` cross-validation (RMSE, MAE, R²) — never a random split, since this
   is time-series data.
4. Selects whichever model has the higher mean cross-validated R² that day.
5. Retrains that model type on all available data and registers it in the Hopsworks
   Model Registry (`aqi_forecast_model`), with metrics attached.

Model selection is **data-driven, not hardcoded** — as more data accumulates, the pipeline
may switch models automatically if performance characteristics change.

### 4. Automated CI/CD

Two GitHub Actions workflows, both triggerable manually via the **Actions** tab or on
their schedule:

| Workflow | Schedule (UTC) | File |
|---|---|---|
| Feature Pipeline | Hourly (`5 * * * *`) | `.github/workflows/feature_pipeline.yml` |
| Training Pipeline | Daily at 02:00 (`0 2 * * *`) | `.github/workflows/training_pipeline.yml` |

Required **GitHub repository secrets** (Settings → Secrets and variables → Actions):

- `OPENWEATHER_API_KEY`
- `HOPSWORKS_API_KEY`

### 5. Web Dashboard

A multi-page Streamlit app, deployed on Streamlit Community Cloud:

- **🏠 Home / About** (`app.py`) — project overview, architecture, tech stack, key features.
- **📊 Live Forecast** (`pages/1_Live_Forecast.py`) — current AQI, 3-day-ahead forecast
  with delta, model version in use, a color-coded hazardous-AQI alert banner, an AQI
  scale reference, and current pollutant/weather readings.
- **📈 Trends & Explainability** (`pages/2_Trends_and_Explainability.py`) — historical AQI
  trend chart with a selectable time range (7/30/90 days or all data), an
  average-AQI-by-hour chart, and a SHAP (or linear-coefficient) explanation of the current
  forecast.

Data loading is **server-side filtered by date range** and cached (55-minute TTL, matching
the hourly update cadence) to keep page loads fast — see
[Troubleshooting](#troubleshooting) if pages feel slow.

Required **Streamlit Cloud secret** (Settings → Secrets, TOML format):

```toml
HOPSWORKS_API_KEY = "your_hopsworks_api_key_here"
```

> Note: the web app does **not** need `OPENWEATHER_API_KEY` — it only reads from
> Hopsworks, it never calls OpenWeather directly.

---

## Results Summary

**Exploratory findings** (full year of data):

- AQI ranged 43–158, averaging ~75 (moderate to unhealthy-for-sensitive-groups).
- Clear daily cycle: AQI dips in early morning (hours 0–6), peaks late morning/midday
  (hours 9–13).
- PM2.5/PM10 correlate ~0.94–1.00 with AQI (expected, since AQI is derived from PM2.5).
- Temperature correlates positively with AQI (+0.54); humidity negatively (−0.42).

**Model evaluation** (5-fold time-series cross-validation):

| Model | Mean RMSE | Mean R² |
|---|---|---|
| Ridge Regression | 42.07 | −3.495 |
| **Random Forest (selected)** | **24.85** | **−0.179** |

Random Forest was selected because it degrades far more gracefully across folds than
Ridge (which collapsed catastrophically in one fold, R² = −16.99, when a small early
training window collided with an extreme AQI event). Random Forest's final fold — the one
with the most training data — achieved a **positive** R² of 0.14, suggesting performance
should keep improving as the automated pipeline accumulates more history.

**SHAP explainability** revealed that `month`, `pressure`, `temp`, and `day` dominate the
model's predictions — more than the pollutant readings themselves. This indicates that a
72-hour AQI forecast for Karachi is fundamentally a **weather/season prediction problem**
as much as a pollution-tracking one.

For the full write-up, see the project report (`Pearls_AQI_Predictor_Report.docx` / `.pdf`).

---

## Setup Guide

### Prerequisites

- A free [OpenWeather](https://openweathermap.org/api) account and API key
- A free [Hopsworks](https://app.hopsworks.ai) account, project, and API key
  (scopes needed: `FEATURESTORE`, `MODELREGISTRY`, `PROJECT`, `DATASET_CREATE/VIEW/DELETE`, `JOB`)
- A GitHub account (for Actions automation)
- A [Streamlit Community Cloud](https://share.streamlit.io) account (for the dashboard)

### 1. Clone and configure

```bash
git clone https://github.com/mahamtahir272/aqi-predictor-karachi.git
cd aqi-predictor-karachi
```

### 2. Add GitHub repository secrets

Settings → Secrets and variables → Actions → New repository secret:

- `OPENWEATHER_API_KEY`
- `HOPSWORKS_API_KEY`

### 3. Enable GitHub Actions

The workflows in `.github/workflows/` activate automatically once pushed to `main`. You
can trigger either manually: **Actions tab → select workflow → Run workflow**.

### 4. Deploy the dashboard

On [share.streamlit.io](https://share.streamlit.io):

1. **Create app** → **Deploy a public app from GitHub**
2. Repository: `mahamtahir272/aqi-predictor-karachi`, branch: `main`, main file: `app.py`
3. **Advanced settings** → set Python version to **3.11**
4. **Secrets** → add:
   ```toml
   HOPSWORKS_API_KEY = "your_hopsworks_api_key_here"
   ```
5. Deploy.

---

## Local Development

**Run the pipelines locally:**

```bash
pip install -r requirements-pipelines.txt
export OPENWEATHER_API_KEY="..."
export HOPSWORKS_API_KEY="..."
python feature_pipeline.py
python training_pipeline.py
```

**Run the dashboard locally:**

```bash
pip install -r requirements.txt
# create .streamlit/secrets.toml with HOPSWORKS_API_KEY = "..."
streamlit run app.py
```

---

## Design Decisions & Known Deviations

The original brief suggested AQICN and TensorFlow specifically. Both were substituted for
documented, practical reasons:

**Historical data: Open-Meteo instead of AQICN/OpenWeather.**
OpenWeather's free tier has no historical air-pollution endpoint. Open-Meteo provides a
full year of free hourly historical data with no API key, which was essential — smaller
test backfills (7 and 90 days) produced misleadingly poor model performance because the
train/test split landed across different seasonal regimes.

**Deep learning: scikit-learn `MLPRegressor` instead of TensorFlow.**
TensorFlow requires `protobuf >= 5.28`, while the Hopsworks Python client requires
`protobuf < 5.0` — an unresolvable version conflict within the same environment.
`MLPRegressor` (a genuine feed-forward neural network) satisfies the "deep learning
model" requirement without breaking the Feature Store connection.

**Flask/FastAPI not used.**
The brief allows either Streamlit/Gradio *or* Flask/FastAPI. Streamlit alone fully
satisfies the interactive dashboard requirement.

**Two separate `requirements*.txt` files.**
The web app never inserts data (read-only), so it doesn't need `confluent-kafka`
(required only for the feature pipeline's online-store writes). `confluent-kafka` failed
to compile in the Streamlit Cloud environment (missing `librdkafka` system library), so
app and pipeline dependencies were split into `requirements.txt` (app) and
`requirements-pipelines.txt` (GitHub Actions).

---

## Limitations & Future Work

- **Model accuracy**: even the best model (Random Forest) shows negative mean R² under
  time-series cross-validation — it does not yet reliably outperform a naive persistence
  forecast across all seasons. More years of historical data, Karachi-specific features
  (e.g. dust/sandstorm indicators), and ensemble methods are natural next steps.
- **Single city**: the pipeline is hardcoded to Karachi. Generalizing to multiple cities
  would require parameterizing the scripts and Feature Store schema by city.
- **Deep learning ceiling**: with more accumulated data, revisiting an LSTM/temporal model
  (in an environment where the TensorFlow/Hopsworks dependency conflict is resolved, e.g.
  a separate training service) could capture longer-range temporal patterns better than
  the current per-row feature approach.
- **Alerting**: hazardous-AQI alerts are currently dashboard-only (visual banner); a future
  iteration could add push/email/SMS notifications.

---

## Troubleshooting

**Feature group schema errors (`Features are not compatible with Feature Group schema`)**
The feature group's column types are fixed by its first insert. Every dataframe inserted
afterward must match exactly — use `align_schema_for_hopsworks()` as a template for any
new insert path, and explicitly force `float64` on derived columns
(`pd.Series([...], dtype="float64")`), since plain Python floats can still be inferred as
`int64` in edge cases.

**Streamlit Cloud build fails on `confluent-kafka` or with `ModuleNotFoundError: No module named 'imp'`**
This means the app is reading the wrong requirements file, or Hopsworks resolved to an old
version incompatible with newer Python. Confirm `requirements.txt` (app) has **no**
`confluent-kafka` entry and pins `hopsworks>=4.0`, and confirm the app's Python version is
set to 3.11 in Streamlit Cloud's app settings (not just `runtime.txt`, which is not always
honored).

**Dashboard pages loading slowly**
Check that `load_features()` calls are passing a `days_back` argument — an unfiltered read
transfers the entire feature history over the network on every cache miss. See `utils.py`.

**GitHub Actions workflow fails but the code "looks correct" locally**
Confirm the push actually landed on `main` — `git status` and `git diff` locally, then
re-verify the file's content directly on GitHub.com, before re-running the workflow. A
run triggered before a push lands will still use the old commit.

---

## License

Internship project — Pearls AQI Predictor. Data sources: OpenWeather, Open-Meteo. Model
serving: Hopsworks.
