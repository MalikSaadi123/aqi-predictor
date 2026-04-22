from flask import Flask, jsonify, request
import numpy as np
import joblib
import json
import os
from datetime import datetime
import requests as req

app = Flask(__name__)

# ── Feature names ─────────────────────────────────────────────
feature_names = [
    "temperature", "humidity", "windspeed", "precipitation",
    "hour", "day", "month", "weekday", "is_weekend",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_lag_24h",
    "aqi_change_rate", "aqi_rolling_6h", "aqi_rolling_24h"
]

def aqi_label(v):
    if v <= 50:  return "Good"
    if v <= 100: return "Moderate"
    if v <= 150: return "Unhealthy for Sensitive Groups"
    if v <= 200: return "Unhealthy"
    if v <= 300: return "Very Unhealthy"
    return "Hazardous"

def fetch_weather():
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        "latitude=33.72148&longitude=73.04329"
        "&hourly=temperature_2m,relativehumidity_2m,"
        "windspeed_10m,precipitation"
        "&forecast_days=1&timezone=UTC"
    )
    data = req.get(url, timeout=10).json()
    h = data["hourly"]
    idx = 0  # current hour
    return {
        "temperature": h["temperature_2m"][idx],
        "humidity":    h["relativehumidity_2m"][idx],
        "windspeed":   h["windspeed_10m"][idx],
        "precipitation": h["precipitation"][idx]
    }

def build_input(weather):
    now = datetime.utcnow()
    hour  = now.hour
    month = now.month
    row = {
        "temperature":    weather["temperature"],
        "humidity":       weather["humidity"],
        "windspeed":      weather["windspeed"],
        "precipitation":  weather["precipitation"],
        "hour":           hour,
        "day":            now.day,
        "month":          month,
        "weekday":        now.weekday(),
        "is_weekend":     int(now.weekday() >= 5),
        "hour_sin":       np.sin(2 * np.pi * hour  / 24),
        "hour_cos":       np.cos(2 * np.pi * hour  / 24),
        "month_sin":      np.sin(2 * np.pi * month / 12),
        "month_cos":      np.cos(2 * np.pi * month / 12),
        "aqi_lag_1h":     80.0,
        "aqi_lag_3h":     80.0,
        "aqi_lag_6h":     80.0,
        "aqi_lag_24h":    80.0,
        "aqi_change_rate": 0.0,
        "aqi_rolling_6h":  80.0,
        "aqi_rolling_24h": 80.0,
    }
    return np.array([[row[f] for f in feature_names]])

# ── Routes ─────────────────────────────────────────────────────
@app.route("/")
def home():
    return jsonify({
        "message": "AQI Flask API is running!",
        "city": "islamabad",
        "endpoints": ["/predict", "/health"]
    })

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route("/predict")
def predict():
    try:
        model = joblib.load("models/best_model.pkl")
        weather = fetch_weather()
        X = build_input(weather)
        aqi = float(np.clip(model.predict(X)[0], 0, 500))
        return jsonify({
            "city": "islamabad",
            "predicted_aqi": round(aqi, 1),
            "status": aqi_label(aqi),
            "weather": weather,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
