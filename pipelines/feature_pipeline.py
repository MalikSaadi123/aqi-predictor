"""
Feature Pipeline — AQI Predictor
Fetches raw weather + pollutant data, engineers features, stores in Hopsworks.
Run this every hour via GitHub Actions.
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hopsworks

# ── CONFIG ──────────────────────────────────────────────────────────────────
AQICN_TOKEN   = os.environ["AQICN_TOKEN"]        # get free key at aqicn.org/api
HOPSWORKS_KEY = os.environ["HOPSWORKS_API_KEY"]  # get at app.hopsworks.ai
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

def fetch_weather_data(city: str) -> dict:
    """Fetch weather from Open-Meteo (completely free, no key needed)."""
    # First get lat/lon for the city using a geocoding call
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
    geo = requests.get(geo_url, timeout=10).json()
    lat = geo["results"][0]["latitude"]
    lon = geo["results"][0]["longitude"]

    weather_url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,relativehumidity_2m,windspeed_10m,precipitation"
        f"&forecast_days=4&timezone=auto"
    )
    return requests.get(weather_url, timeout=10).json(), lat, lon

# ── 2. COMPUTE FEATURES ──────────────────────────────────────────────────────
def compute_features(aqi_data: dict, weather_data: dict) -> pd.DataFrame:
    now = datetime.utcnow()

    # Core AQI values
    aqi      = aqi_data["aqi"]
    iaqi     = aqi_data.get("iaqi", {})
    pm25     = iaqi.get("pm25", {}).get("v", np.nan)
    pm10     = iaqi.get("pm10", {}).get("v", np.nan)
    no2      = iaqi.get("no2",  {}).get("v", np.nan)
    o3       = iaqi.get("o3",   {}).get("v", np.nan)
    co       = iaqi.get("co",   {}).get("v", np.nan)
    so2      = iaqi.get("so2",  {}).get("v", np.nan)

    # Weather features from Open-Meteo (current hour index)
    hourly    = weather_data["hourly"]
    times     = pd.to_datetime(hourly["time"])
    current_i = (times <= pd.Timestamp(now)).sum() - 1
    current_i = max(0, current_i)

    temperature  = hourly["temperature_2m"][current_i]
    humidity     = hourly["relativehumidity_2m"][current_i]
    windspeed    = hourly["windspeed_10m"][current_i]
    precipitation= hourly["precipitation"][current_i]

    # Time-based features
    hour       = now.hour
    day        = now.day
    month      = now.month
    weekday    = now.weekday()
    is_weekend = int(weekday >= 5)

    # Derived / cyclical features
    hour_sin   = np.sin(2 * np.pi * hour   / 24)
    hour_cos   = np.cos(2 * np.pi * hour   / 24)
    month_sin  = np.sin(2 * np.pi * month  / 12)
    month_cos  = np.cos(2 * np.pi * month  / 12)

    # AQI category (target label for classification tasks)
    def aqi_category(v):
        if v <= 50:   return "Good"
        if v <= 100:  return "Moderate"
        if v <= 150:  return "Unhealthy for Sensitive Groups"
        if v <= 200:  return "Unhealthy"
        if v <= 300:  return "Very Unhealthy"
        return "Hazardous"

    row = {
        "timestamp":    now.strftime("%Y-%m-%d %H:%M:%S"),
        "city":         CITY,
        "aqi":          float(aqi),
        "pm25":         float(pm25) if pm25 == pm25 else None,
        "pm10":         float(pm10) if pm10 == pm10 else None,
        "no2":          float(no2)  if no2  == no2  else None,
        "o3":           float(o3)   if o3   == o3   else None,
        "co":           float(co)   if co   == co   else None,
        "so2":          float(so2)  if so2  == so2  else None,
        "temperature":  temperature,
        "humidity":     humidity,
        "windspeed":    windspeed,
        "precipitation":precipitation,
        "hour":         hour,
        "day":          day,
        "month":        month,
        "weekday":      weekday,
        "is_weekend":   is_weekend,
        "hour_sin":     hour_sin,
        "hour_cos":     hour_cos,
        "month_sin":    month_sin,
        "month_cos":    month_cos,
        "aqi_category": aqi_category(aqi),
    }
    return pd.DataFrame([row])

def add_lag_and_change_rate(df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    """Add lag features and AQI change rate using recent history."""
    combined = pd.concat([history_df, df], ignore_index=True).sort_values("timestamp")
    combined["aqi_lag_1h"]      = combined["aqi"].shift(1)
    combined["aqi_lag_3h"]      = combined["aqi"].shift(3)
    combined["aqi_lag_6h"]      = combined["aqi"].shift(6)
    combined["aqi_lag_24h"]     = combined["aqi"].shift(24)
    combined["aqi_change_rate"] = combined["aqi"].diff()           # Δ per hour
    combined["aqi_rolling_6h"]  = combined["aqi"].rolling(6).mean()
    combined["aqi_rolling_24h"] = combined["aqi"].rolling(24).mean()
    return combined.tail(1)  # return only the latest row with lags filled

# ── 3. STORE IN HOPSWORKS FEATURE STORE ─────────────────────────────────────
def push_to_feature_store(df: pd.DataFrame):
    project = hopsworks.login(api_key_value=HOPSWORKS_KEY)
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["timestamp", "city"],
        description="Hourly AQI features with weather and pollutant data",
        event_time="timestamp",
    )
    fg.insert(df, write_options={"wait_for_job": False})
    print(f"✅ Inserted {len(df)} row(s) into feature store.")

# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[{datetime.utcnow()}] Starting feature pipeline for '{CITY}'...")

    aqi_raw                 = fetch_aqi_data(CITY)
    weather_raw, lat, lon   = fetch_weather_data(CITY)
    features_df             = compute_features(aqi_raw, weather_raw)

    # Try to fetch recent history for lag features (optional, graceful fallback)
    try:
        project    = hopsworks.login(api_key_value=HOPSWORKS_KEY)
        fs         = project.get_feature_store()
        fg         = fs.get_feature_group("aqi_features", version=1)
        history_df = fg.filter(fg.city == CITY).read().tail(24)
        features_df = add_lag_and_change_rate(features_df, history_df)
    except Exception as e:
        print(f"⚠️  Could not fetch history for lags: {e}")

    push_to_feature_store(features_df)
    print("Done.")
