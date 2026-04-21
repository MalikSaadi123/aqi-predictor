import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
import hopsworks
import requests

app = FastAPI(
    title="AQI Predictor API",
    description="Predicts Air Quality Index for Islamabad for the next 3 days",
    version="1.0.0"
)

HOPSWORKS_KEY = os.environ["HOPSWORKS_API_KEY"]

# ── Load model once on startup ────────────────────────────────────────────────
model = None
feature_names = None

@app.on_event("startup")
def load_model():
    global model, feature_names
    project = hopsworks.login(api_key_value=HOPSWORKS_KEY)
    mr = project.get_model_registry()
    model_meta = mr.get_model("aqi_predictor", version=3)
    model_dir  = model_meta.download()
    model      = joblib.load(os.path.join(model_dir, "best_model.pkl"))
    with open(os.path.join(model_dir, "feature_names.json")) as f:
        feature_names = json.load(f)
    print("✅ Model loaded successfully!")


# ── Helper ────────────────────────────────────────────────────────────────────
def fetch_weather() -> pd.DataFrame:
    lat, lon = 33.72148, 73.04329
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,relativehumidity_2m,windspeed_10m,precipitation"
        f"&forecast_days=4&timezone=UTC"
    )
    data = requests.get(url, timeout=10).json()
    hourly = data["hourly"]
    df = pd.DataFrame({
        "timestamp":     pd.to_datetime(hourly["time"]),
        "temperature":   hourly["temperature_2m"],
        "humidity":      hourly["relativehumidity_2m"],
        "windspeed":     hourly["windspeed_10m"],
        "precipitation": hourly["precipitation"],
    })
    return df.head(72)


def build_features(df: pd.DataFrame, last_aqi: float = 100.0) -> pd.DataFrame:
    df["hour"]            = df["timestamp"].dt.hour
    df["day"]             = df["timestamp"].dt.day
    df["month"]           = df["timestamp"].dt.month
    df["weekday"]         = df["timestamp"].dt.weekday
    df["is_weekend"]      = (df["weekday"] >= 5).astype(int)
    df["hour_sin"]        = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]        = np.cos(2 * np.pi * df["hour"]  / 24)
    df["month_sin"]       = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]       = np.cos(2 * np.pi * df["month"] / 12)
    df["aqi_lag_1h"]      = last_aqi
    df["aqi_lag_3h"]      = last_aqi
    df["aqi_lag_6h"]      = last_aqi
    df["aqi_lag_24h"]     = last_aqi
    df["aqi_change_rate"] = 0.0
    df["aqi_rolling_6h"]  = last_aqi
    df["aqi_rolling_24h"] = last_aqi
    available = [c for c in feature_names if c in df.columns]
    return df[available].fillna(0.0)


def aqi_label(v: float) -> str:
    if v <= 50:   return "Good"
    if v <= 100:  return "Moderate"
    if v <= 150:  return "Unhealthy for Sensitive Groups"
    if v <= 200:  return "Unhealthy"
    if v <= 300:  return "Very Unhealthy"
    return "Hazardous"


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "AQI Predictor API is running!",
        "docs": "/docs",
        "endpoints": ["/predict", "/forecast", "/health"]
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/predict")
def predict():
    """Returns current AQI prediction for Islamabad."""
    weather_df = fetch_weather()
    X = build_features(weather_df.head(1).copy())
    prediction = float(np.clip(model.predict(X.values)[0], 0, 500))
    return {
        "city": "islamabad",
        "predicted_aqi": round(prediction, 1),
        "status": aqi_label(prediction),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/forecast")
def forecast():
    """Returns 72-hour AQI forecast for Islamabad."""
    weather_df = fetch_weather()
    X = build_features(weather_df.copy())
    predictions = np.clip(model.predict(X.values), 0, 500)
    weather_df["predicted_aqi"] = predictions
    weather_df["status"] = weather_df["predicted_aqi"].apply(aqi_label)

    # Daily summary
    weather_df["date"] = weather_df["timestamp"].dt.date
    daily = weather_df.groupby("date")["predicted_aqi"].agg(["mean","min","max"]).reset_index()

    return {
        "city": "islamabad",
        "generated_at": datetime.utcnow().isoformat(),
        "hourly_forecast": [
            {
                "timestamp": str(row["timestamp"]),
                "predicted_aqi": round(row["predicted_aqi"], 1),
                "status": row["status"]
            }
            for _, row in weather_df.iterrows()
        ],
        "daily_summary": [
            {
                "date": str(row["date"]),
                "avg_aqi": round(row["mean"], 1),
                "min_aqi": round(row["min"], 1),
                "max_aqi": round(row["max"], 1),
            }
            for _, row in daily.iterrows()
        ]
    }
