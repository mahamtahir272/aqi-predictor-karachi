# AQI Predictor — Karachi

Serverless, end-to-end AQI (Air Quality Index) forecasting pipeline for Karachi.
Predicts AQI 3 days (72 hours) ahead using weather + pollution data.

## Architecture

```
OpenWeather API ──(hourly)──► feature_pipeline.py ──► Hopsworks Feature Store
                                                              │
                                                              ▼
                                            training_pipeline.py (daily)
                                                              │
                                                              ▼
                                          Hopsworks Model Registry ──► Streamlit app
```

## Setup

### 1. Repo secrets

Go to **Settings → Secrets and variables → Actions → New repository secret** and add:

| Secret name            | Where to get it                                      |
|-------------------------|-------------------------------------------------------|
| `OPENWEATHER_API_KEY`   | https://openweathermap.org/api → API keys tab          |
| `HOPSWORKS_API_KEY`     | Hopsworks project → Account Settings → API Keys        |

### 2. Enable GitHub Actions

Workflows are already defined under `.github/workflows/`:
- `feature_pipeline.yml` — runs `feature_pipeline.py` every hour
- `training_pipeline.yml` — runs `training_pipeline.py` once a day

They'll start running automatically on the defined schedule once pushed to GitHub.
You can also trigger either manually: go to the **Actions** tab → select the workflow →
**Run workflow**.

### 3. Local / Colab testing

```bash
pip install -r requirements.txt
export OPENWEATHER_API_KEY="..."
export HOPSWORKS_API_KEY="..."
python feature_pipeline.py
python training_pipeline.py
```

## Files

- `feature_pipeline.py` — fetches live weather/pollution data, computes features, writes to the Feature Store
- `training_pipeline.py` — pulls historical features from the Feature Store, evaluates Ridge vs Random Forest with time-series cross-validation, registers the best model
- `requirements.txt` — Python dependencies
- `.github/workflows/` — GitHub Actions schedule definitions

## Notes

- The feature group schema was locked in on first insert (includes an `nh3` column and
  64-bit int types for time features) — both scripts align to that schema automatically
  via `align_schema_for_hopsworks()`.
- Model selection is data-driven: `training_pipeline.py` re-runs 5-fold time-series CV
  each day and registers whichever of Ridge/RandomForest currently performs best, rather
  than hardcoding one.
