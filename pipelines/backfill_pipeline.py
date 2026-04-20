"""
Backfill Pipeline — AQI Predictor
Fetches historical AQI data for a date range using AQICN's history API
and populates the Hopsworks Feature Store for model training.

Usage:
    python backfill_pipeline.py --start 2024-01-01 --end 2024-12-31 --city rawalpindi
"""

import os
import argparse
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hopsworks
from tqdm import tqdm

AQICN_TOKEN   = os.environ["AQICN_TOKEN"]
HOPSWORKS_KEY = os.environ["HOPSWORKS_API_KEY"]


def fetch_historical_aqi(city: str, date: datetime) -> float | None:
    """AQICN historical endpoint (requires a registered token)."""
    date_str = date.strftime("%Y-%m-%d")
    url = f"https://api.waqi.info/feed/{city}/?token={AQICN_TOKEN}"
    # Note: for true historical data per day, use AQICN's map API or
    # OpenAQ which has free historical data via their API v3.
    try:
        resp = requests.get(url, timeout=10).json()
        if resp["status"] == "ok":
            return float(resp["data"]["aqi"])
    except Exception:
        pass
    return None


def fetch_historical_weather(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Open-Meteo historical API — completely free."""
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start}&end_date={end}"
        f"&hourly=temperature_2m,relativehumidity_2m,windspeed_10m,precipitation"
        f"&timezone=auto"
    )
    data = requests.get(url, timeout=30).json()
    df = pd.DataFrame({
        "timestamp":    pd.to_datetime(data["hourly"]["time"]),
        "temperature":  data["hourly"]["temperature_2m"],
        "humidity":     data["hourly"]["relativehumidity_2m"],
        "windspeed":    data["hourly"]["windspeed_10m"],
        "precipitation":data["hourly"]["precipitation"],
    })
    return df


def geocode(city: str):
    geo = requests.get(
        f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1",
        timeout=10
    ).json()
    r = geo["results"][0]
    return r["latitude"], r["longitude"]


def fetch_openaq_history(city: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    OpenAQ v3 API — free, no key needed for basic use.
    Returns hourly PM2.5, PM10, NO2, O3 readings.
    """
    base = "https://api.openaq.org/v3/locations"
    # Step 1: find location id
    loc_resp = requests.get(
        base,
        params={"city": city, "limit": 1},
        headers={"X-API-Key": os.environ.get("OPENAQ_KEY", "")},
        timeout=10,
    ).json()

    if not loc_resp.get("results"):
        print(f"⚠️  No OpenAQ location found for '{city}'. Skipping pollutant backfill.")
        return pd.DataFrame()

    loc_id = loc_resp["results"][0]["id"]

    # Step 2: fetch measurements
    rows = []
    page = 1
    while True:
        meas = requests.get(
            f"https://api.openaq.org/v3/locations/{loc_id}/measurements",
            params={
                "date_from": start.isoformat(),
                "date_to":   end.isoformat(),
                "limit":     1000,
                "page":      page,
            },
            timeout=15,
        ).json()
        results = meas.get("results", [])
        if not results:
            break
        rows.extend(results)
        page += 1
        if page > 50:  # safety cap
            break

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["date"].apply(lambda d: d["utc"]))
    df = df.pivot_table(index="timestamp", columns="parameter", values="value", aggfunc="mean")
    df = df.reset_index()
    return df


def build_training_rows(
    weather_df: pd.DataFrame,
    pollutant_df: pd.DataFrame,
    city: str,
) -> pd.DataFrame:
    """Merge weather + pollutants and engineer all features."""
    df = weather_df.copy()

    # Merge pollutants if available
    if not pollutant_df.empty:
        df = df.merge(pollutant_df, on="timestamp", how="left")

    df["city"]       = city
    df["hour"]       = df["timestamp"].dt.hour
    df["day"]        = df["timestamp"].dt.day
    df["month"]      = df["timestamp"].dt.month
    df["weekday"]    = df["timestamp"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"]  / 24)
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)

    # Lag features
    for lag in [1, 3, 6, 24]:
        col = f"pm25" if "pm25" in df.columns else "temperature"
        df[f"aqi_lag_{lag}h"] = df[col].shift(lag)

    if "pm25" in df.columns:
        df["aqi_change_rate"] = df["pm25"].diff()
        df["aqi_rolling_6h"]  = df["pm25"].rolling(6).mean()
        df["aqi_rolling_24h"] = df["pm25"].rolling(24).mean()
        df["aqi"]             = df["pm25"]   # use PM2.5 as AQI proxy if no direct AQI

    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df.dropna(subset=["aqi"]) if "aqi" in df.columns else df


def push_to_feature_store(df: pd.DataFrame, batch_size: int = 500):
    project = hopsworks.login(api_key_value=HOPSWORKS_KEY)
    fs = project.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["timestamp", "city"],
        description="Hourly AQI features — backfilled historical data",
        event_time="timestamp",
    )
    for i in tqdm(range(0, len(df), batch_size), desc="Uploading batches"):
        fg.insert(df.iloc[i:i+batch_size], write_options={"wait_for_job": False})
    print(f"✅ Uploaded {len(df)} historical rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end",   default="2024-12-31")
    parser.add_argument("--city",  default=os.environ.get("CITY", "islamabad"))
    args = parser.parse_args()

    print(f"Backfilling {args.city} from {args.start} to {args.end}...")

    lat, lon       = geocode(args.city)
    weather_df     = fetch_historical_weather(lat, lon, args.start, args.end)
    start_dt       = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt         = datetime.strptime(args.end,   "%Y-%m-%d")
    pollutant_df   = fetch_openaq_history(args.city, start_dt, end_dt)
    training_df    = build_training_rows(weather_df, pollutant_df, args.city)

    print(f"Built {len(training_df)} rows of training data.")
    push_to_feature_store(training_df)
