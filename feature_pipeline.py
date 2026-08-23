"""
Feature Pipeline — runs hourly via GitHub Actions.

Fetches current weather + pollution data for Karachi from OpenWeather,
computes features (time-based + derived), and writes them into the
Hopsworks Feature Store.

Required environment variables (set as GitHub Secrets):
    OPENWEATHER_API_KEY
    HOPSWORKS_API_KEY
"""

import os
import sys
import requests
import pandas as pd
from datetime import datetime, timezone

import hopsworks

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
CITY_NAME = "Karachi"
LAT, LON = 24.8607, 67.0011
HOPSWORKS_PROJECT = "aqi_predictor_maham"
HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"

OPENWEATHER_API_KEY = os.environ["OPENWEATHER_API_KEY"]
HOPSWORKS_API_KEY = os.environ["HOPSWORKS_API_KEY"]


# ---------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------
def fetch_air_pollution(lat, lon, api_key):
    url = "http://api.openweathermap.org/data/2.5/air_pollution"
    params = {"lat": lat, "lon": lon, "appid": api_key}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_weather(lat, lon, api_key):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def calculate_aqi_from_pm25(pm25):
    """EPA AQI breakpoints for PM2.5."""
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= pm25 <= c_hi:
            return round(((i_hi - i_lo) / (c_hi - c_lo)) * (pm25 - c_lo) + i_lo)
    return 500


def build_feature_row(lat, lon, api_key, city_name):
    pollution = fetch_air_pollution(lat, lon, api_key)
    weather = fetch_weather(lat, lon, api_key)

    components = pollution["list"][0]["components"]
    dt_unix = pollution["list"][0]["dt"]
    dt = datetime.fromtimestamp(dt_unix, tz=timezone.utc)

    row = {
        "city": city_name,
        "timestamp": dt,
        "unix_time": dt_unix,
        "pm2_5": components.get("pm2_5"),
        "pm10": components.get("pm10"),
        "no2": components.get("no2"),
        "o3": components.get("o3"),
        "co": components.get("co"),
        "so2": components.get("so2"),
        "nh3": components.get("nh3"),
        "temp": weather["main"].get("temp"),
        "humidity": weather["main"].get("humidity"),
        "pressure": weather["main"].get("pressure"),
        "wind_speed": weather["wind"].get("speed"),
        "hour": dt.hour,
        "day": dt.day,
        "month": dt.month,
        "day_of_week": dt.weekday(),
        "aqi": calculate_aqi_from_pm25(components.get("pm2_5")),
    }
    return row


# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------
def add_derived_features(df, previous_row=None):
    """
    Adds AQI/PM2.5 change rate and lag features.
    In the hourly job we only have ONE new row, so we compare it against
    the most recent row already stored in Hopsworks (previous_row), rather
    than against itself.
    """
    row = df.iloc[0]

    if previous_row is not None:
        aqi_change = row["aqi"] - previous_row["aqi"]
        pm25_change = row["pm2_5"] - previous_row["pm2_5"]
        aqi_lag_1 = previous_row["aqi"]
        aqi_rolling_mean_3 = (row["aqi"] + previous_row["aqi"]) / 2  # simple 2-pt avg fallback
    else:
        aqi_change = 0
        pm25_change = 0
        aqi_lag_1 = row["aqi"]
        aqi_rolling_mean_3 = row["aqi"]

    df["aqi_change_rate"] = float(aqi_change)
    df["pm2_5_change_rate"] = float(pm25_change)
    df["aqi_lag_1"] = float(aqi_lag_1)
    df["aqi_rolling_mean_3"] = float(aqi_rolling_mean_3)
    return df

def align_schema_for_hopsworks(df):
    """Matches the feature group schema locked in by the very first insert."""
    df = df.copy()
    if "nh3" not in df.columns:
        df["nh3"] = pd.NA

    int_cols = ["hour", "day", "month", "day_of_week"]
    for col in int_cols:
        df[col] = df[col].astype("int64")

    df["pressure"] = df["pressure"].round().astype("int64")
    return df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print(f"[{datetime.now(timezone.utc)}] Starting feature pipeline run...")

    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        host=HOPSWORKS_HOST,
        port=443,
        api_key_value=HOPSWORKS_API_KEY,
    )
    fs = project.get_feature_store()
    aqi_fg = fs.get_feature_group(name="aqi_features", version=1)

    # Get the most recent existing row to compute change-rate features against
    try:
        recent = aqi_fg.read().sort_values("timestamp").iloc[-1]
    except Exception as e:
        print(f"Could not read previous row (may be first run): {e}")
        recent = None

    row = build_feature_row(LAT, LON, OPENWEATHER_API_KEY, CITY_NAME)
    df = pd.DataFrame([row])
    df = add_derived_features(df, previous_row=recent)
    df = align_schema_for_hopsworks(df)

    print(df)

    aqi_fg.insert(df, write_options={"wait_for_job": False})
    print("Row inserted successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: feature pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)
