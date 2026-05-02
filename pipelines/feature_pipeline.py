import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import hopsworks

# ── CONFIG ───────────────────────────────────────────────────────────────────
OPENWEATHER_TOKEN = os.environ["OPENWEATHER_TOKEN"]
HOPSWORKS_KEY     = os.environ["HOPSWORKS_API_KEY"]
CITY              = os.environ.get("CITY", "islamabad")

# ── 1. FETCH RAW DATA ────────────────────────────────────────────────────────
def fetch_aqi_data() -> dict:
    url  = f"https://api.openweathermap.org/data/2.5/air_pollution?lat=33.72148&lon=73.04329&appid={OPENWEATHER_TOKEN}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    pm25 = data["list"][0]["components"]["pm2_5"]
    if pm25 <= 12.0:    aqi = (50/12.0) * pm25
    elif pm25 <= 35.4:  aqi = 50 + (50/23.4) * (pm25 - 12.0)
    elif pm25 <= 55.4:  aqi = 100 + (50/19.9) * (pm25 - 35.4)
    elif pm25 <= 150.4: aqi = 150 + (50/94.9) * (pm25 - 55.4)
    elif pm25 <= 250.4: aqi = 200 + (100/99.9) * (pm25 - 150.4)
    else:               aqi = 300 + (200/149.9) * (pm25 - 250.4)
    return {"aqi": round(aqi, 1), "pm25": pm25}


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
    for attempt in range(3):  # retry 3 times
        try:
            resp = requests.get(weather_url, timeout=15)
            if resp.status_code == 200 and resp.text.strip():
                return resp.json()
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
        import time
        time.sleep(5)
    raise ValueError("Could not fetch weather data after 3 attempts")


# ── 2. COMPUTE FEATURES ──────────────────────────────────────────────────────
def compute_features(aqi_data: dict, weather_data: dict) -> pd.DataFrame:
    now     = datetime.utcnow()
    aqi     = float(aqi_data["aqi"])
    hourly  = weather_data["hourly"]
    times   = pd.to_datetime(hourly["time"])
    current_i = max(0, (times <= pd.Timestamp(now)).sum() - 1)

    temperature   = float(hourly["temperature_2m"][current_i])
    humidity      = int(hourly["relativehumidity_2m"][current_i])
    windspeed     = float(hourly["windspeed_10m"][current_i])
    precipitation = float(hourly["precipitation"][current_i])

    hour       = int(now.hour)
    day        = int(now.day)
    month      = int(now.month)
    weekday    = int(now.weekday())
    is_weekend = int(weekday >= 5)

    row = {
        "timestamp":       pd.Timestamp(now).floor("s"),
        "city":            CITY,
        "aqi":             aqi,
        "temperature":     temperature,
        "humidity":        humidity,
        "windspeed":       windspeed,
        "precipitation":   precipitation,
        "hour":            hour,
        "day":             day,
        "month":           month,
        "weekday":         weekday,
        "is_weekend":      is_weekend,
        "hour_sin":        float(np.sin(2 * np.pi * hour  / 24)),
        "hour_cos":        float(np.cos(2 * np.pi * hour  / 24)),
        "month_sin":       float(np.sin(2 * np.pi * month / 12)),
        "month_cos":       float(np.cos(2 * np.pi * month / 12)),
        "aqi_lag_1h":      aqi,
        "aqi_lag_3h":      aqi,
        "aqi_lag_6h":      aqi,
        "aqi_lag_24h":     aqi,
        "aqi_change_rate": 0.0,
        "aqi_rolling_6h":  aqi,
        "aqi_rolling_24h": aqi,
    }
    return pd.DataFrame([row])


# ── 3. STORE IN HOPSWORKS ────────────────────────────────────────────────────
def push_to_feature_store(df: pd.DataFrame):
    for col in ["hour", "day", "month", "weekday", "is_weekend"]:
        df[col] = df[col].astype("int32")

    project = hopsworks.login(api_key_value=HOPSWORKS_KEY)
    fs      = project.get_feature_store()
    fg      = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["timestamp", "city"],
        description="Hourly AQI features — Islamabad",
        event_time="timestamp",
    )
    fg.insert(df, write_options={"wait_for_job": False})
    print(f"✅ Inserted {len(df)} row(s) into feature store.")


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[{datetime.utcnow()}] Starting feature pipeline for '{CITY}'...")
    aqi_raw     = fetch_aqi_data()
    weather_raw = fetch_weather_data()
    features_df = compute_features(aqi_raw, weather_raw)
    print(f"Features shape: {features_df.shape}")
    print(f"Columns: {list(features_df.columns)}")
    push_to_feature_store(features_df)
    print("Done.")
