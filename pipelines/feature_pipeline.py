import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import hopsworks

# ── CONFIG ──────────────────────────────────────────────────────────────────
AQICN_TOKEN   = os.environ["AQICN_TOKEN"]
HOPSWORKS_KEY = os.environ["HOPSWORKS_API_KEY"]
CITY          = os.environ.get("CITY", "islamabad")

# ── 1. FETCH RAW DATA ────────────────────────────────────────────────────────
def fetch_aqi_data(city: str) -> dict:
    url = f"https://api.waqi.info/feed/{city}/?token={AQICN_TOKEN}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data["status"] != "ok":
        raise ValueError(f"AQICN error: {data}")
    return data["data"]

def fetch_weather_data() -> dict:
    """Fetch weather from Open-Meteo using hardcoded Islamabad coordinates."""
    lat = 33.72148
    lon = 73.04329
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,relativehumidity_2m,windspeed_10m,precipitation"
        f"&forecast_days=1&timezone=UTC"
    )
    return requests.get(weather_url, timeout=10).json()

# ── 2. COMPUTE FEATURES ──────────────────────────────────────────────────────
def compute_features(aqi_data: dict, weather_data: dict) -> pd.DataFrame:
    now = datetime.utcnow()

    # Core AQI values
    aqi = float(aqi_data["aqi"])

    # Weather features from Open-Meteo (current hour index)
    hourly    = weather_data["hourly"]
    times     = pd.to_datetime(hourly["time"])
    current_i = max(0, (times <= pd.Timestamp(now)).sum() - 1)

    temperature   = float(hourly["temperature_2m"][current_i])
    humidity      = int(hourly["relativehumidity_2m"][current_i])
    windspeed     = float(hourly["windspeed_10m"][current_i])
    precipitation = float(hourly["precipitation"][current_i])

    # Time-based features — use int32 to match schema
    hour      = int(now.hour)
    day       = int(now.day)
    month     = int(now.month)
    weekday   = int(now.weekday())
    is_weekend= int(weekday >= 5)

    # Cyclical features
    hour_sin  = float(np.sin(2 * np.pi * hour  / 24))
    hour_cos  = float(np.cos(2 * np.pi * hour  / 24))
    month_sin = float(np.sin(2 * np.pi * month / 12))
    month_cos = float(np.cos(2 * np.pi * month / 12))

    # Lag features — use AQI as proxy since we don't have history yet
    aqi_lag_1h      = aqi
    aqi_lag_3h      = aqi
    aqi_lag_6h      = aqi
    aqi_lag_24h     = aqi
    aqi_change_rate = 0.0
    aqi_rolling_6h  = aqi
    aqi_rolling_24h = aqi

    row = {
        "timestamp":      pd.Timestamp(now).floor("s"),  # proper timestamp type
        "city":           CITY,
        "aqi":            aqi,
        "temperature":    temperature,
        "humidity":      int(humidity),
        "windspeed":      windspeed,
        "precipitation":  precipitation,
        "hour":           np.int32(hour),
        "day":            np.int32(day),
        "month":          np.int32(month),
        "weekday":        np.int32(weekday),
        "is_weekend":     int(is_weekend),
        "hour_sin":       hour_sin,
        "hour_cos":       hour_cos,
        "month_sin":      month_sin,
        "month_cos":      month_cos,
        "aqi_lag_1h":     aqi_lag_1h,
        "aqi_lag_3h":     aqi_lag_3h,
        "aqi_lag_6h":     aqi_lag_6h,
        "aqi_lag_24h":    aqi_lag_24h,
        "aqi_change_rate":aqi_change_rate,
        "aqi_rolling_6h": aqi_rolling_6h,
        "aqi_rolling_24h":aqi_rolling_24h,
    }
    return pd.DataFrame([row])

# ── 3. STORE IN HOPSWORKS FEATURE STORE ─────────────────────────────────────
def push_to_feature_store(df: pd.DataFrame):
    project = hopsworks.login(api_key_value=HOPSWORKS_KEY)
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["timestamp", "city"],
        description="Hourly AQI features with weather data",
        event_time="timestamp",
    )
    fg.insert(df, write_options={"wait_for_job": False})
    print(f"✅ Inserted {len(df)} row(s) into feature store.")

# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[{datetime.utcnow()}] Starting feature pipeline for '{CITY}'...")

    aqi_raw     = fetch_aqi_data(CITY)
    weather_raw = fetch_weather_data()
    features_df = compute_features(aqi_raw, weather_raw)

    print(f"Features shape: {features_df.shape}")
    print(f"Columns: {list(features_df.columns)}")

    push_to_feature_store(features_df)
    print("Done.")
