import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import hopsworks
import requests

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AQI Forecast",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

HOPSWORKS_KEY = os.environ["HOPSWORKS_API_KEY"]
CITY          = os.environ.get("CITY", "islamabad")

AQI_LEVELS = [
    (0,   50,  "#00e400", "Good",                         "Air quality is satisfactory."),
    (51,  100, "#ffff00", "Moderate",                     "Acceptable quality; some pollutants may concern sensitive groups."),
    (101, 150, "#ff7e00", "Unhealthy for Sensitive Groups","Members of sensitive groups may experience effects."),
    (151, 200, "#ff0000", "Unhealthy",                    "Everyone may begin to experience health effects."),
    (201, 300, "#8f3f97", "Very Unhealthy",               "Health alert — everyone may experience serious effects."),
    (301, 500, "#7e0023", "Hazardous",                    "Emergency conditions. Entire population is likely affected."),
]

def aqi_info(value: float):
    for lo, hi, color, label, desc in AQI_LEVELS:
        if lo <= value <= hi:
            return color, label, desc
    return "#7e0023", "Hazardous", "Extreme pollution."


# ── DATA LOADERS (cached) ────────────────────────────────────────────────────
@st.cache_resource(ttl=3600)
def load_model_and_features():
    project = hopsworks.login(api_key_value=HOPSWORKS_KEY)
    mr = project.get_model_registry()
    model_meta = mr.get_best_model("aqi_predictor", metric="rmse", direction="min")
    model_dir  = model_meta.download()
    model      = joblib.load(os.path.join(model_dir, "best_model.pkl"))
    with open(os.path.join(model_dir, "feature_names.json")) as f:
        feature_names = json.load(f)
    with open(os.path.join(model_dir, "metrics.json")) as f:
        metrics = json.load(f)
    return model, feature_names, metrics


@st.cache_data(ttl=1800)
def load_recent_features(city: str) -> pd.DataFrame:
    project = hopsworks.login(api_key_value=HOPSWORKS_KEY)
    fs = project.get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=1)
    df = fg.filter(fg.city == city.lower()).read()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").tail(72)   # last 3 days
    return df


@st.cache_data(ttl=3600)
def fetch_future_weather(city: str) -> pd.DataFrame:
    geo = requests.get(
        f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1",
        timeout=10,
    ).json()
    lat = geo["results"][0]["latitude"]
    lon = geo["results"][0]["longitude"]
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,relativehumidity_2m,windspeed_10m,precipitation"
        f"&forecast_days=4&timezone=auto"
    )
    data = requests.get(url, timeout=10).json()
    df = pd.DataFrame({
        "timestamp":    pd.to_datetime(data["hourly"]["time"]),
        "temperature":  data["hourly"]["temperature_2m"],
        "humidity":     data["hourly"]["relativehumidity_2m"],
        "windspeed":    data["hourly"]["windspeed_10m"],
        "precipitation":data["hourly"]["precipitation"],
    })
    return df[df["timestamp"] > datetime.now()].head(72)  # next 3 days


def build_forecast_features(future_weather: pd.DataFrame, history: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    df = future_weather.copy()
    df["hour"]       = df["timestamp"].dt.hour
    df["day"]        = df["timestamp"].dt.day
    df["month"]      = df["timestamp"].dt.month
    df["weekday"]    = df["timestamp"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"]  / 24)
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)

    # Use last known AQI stats for lag features
    last_aqi = history["aqi"].iloc[-1] if "aqi" in history.columns and len(history) else 100.0
    df["aqi_lag_1h"]      = last_aqi
    df["aqi_lag_3h"]      = last_aqi
    df["aqi_lag_6h"]      = last_aqi
    df["aqi_lag_24h"]     = last_aqi
    df["aqi_change_rate"] = 0.0
    df["aqi_rolling_6h"]  = last_aqi
    df["aqi_rolling_24h"] = last_aqi

    # Pollutant columns — fill with recent median
    for col in ["pm25","pm10","no2","o3","co","so2"]:
        if col in history.columns:
            df[col] = history[col].median()
        else:
            df[col] = np.nan

    # Keep only the columns the model was trained on
    available = [c for c in feature_names if c in df.columns]
    df_model = df[available].fillna(df[available].median())
    return df, df_model, available


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4150/4150900.png", width=80)
    st.title("⚙️ Settings")
    city_input = st.text_input("City", value=CITY)
    st.markdown("---")
    st.markdown("**About**")
    st.caption("End-to-end AQI Predictor using ML + Open-Meteo weather forecasts.")
    st.caption("Data refreshes every hour automatically.")

# ── MAIN LAYOUT ───────────────────────────────────────────────────────────────
st.title(f"🌫️ AQI Forecast — {city_input}")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

with st.spinner("Loading model and data..."):
    try:
        model, feature_names, metrics = load_model_and_features()
        history_df   = load_recent_features(city_input)
        future_wx    = fetch_future_weather(city_input)
        df_future, X_future, used_features = build_forecast_features(future_wx, history_df, feature_names)

        predictions = model.predict(X_future.values)
        df_future["predicted_aqi"] = np.clip(predictions, 0, 500)

        # ── CURRENT AQI CARD ──────────────────────────────────────────────────
        current_aqi = history_df["aqi"].iloc[-1] if len(history_df) else predictions[0]
        color, label, desc = aqi_info(current_aqi)

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.markdown(
                f"""<div style='background:{color};border-radius:12px;padding:20px;text-align:center;'>
                <h1 style='color:black;margin:0;font-size:64px'>{int(current_aqi)}</h1>
                <h3 style='color:black;margin:4px'>{label}</h3>
                <p style='color:black;font-size:13px'>{desc}</p></div>""",
                unsafe_allow_html=True,
            )
        with col2:
            st.metric("📊 Model RMSE", f"{metrics.get('rmse',0):.1f}")
            st.metric("📐 R² Score",   f"{metrics.get('r2',0):.3f}")
        with col3:
            avg_3day = df_future["predicted_aqi"].mean()
            peak_3day = df_future["predicted_aqi"].max()
            st.metric("📅 Avg (3-day forecast)", f"{avg_3day:.0f}")
            st.metric("⚠️ Peak AQI",             f"{peak_3day:.0f}")

        st.markdown("---")

        # ── ALERT ────────────────────────────────────────────────────────────
        if peak_3day > 150:
            st.error(f"🚨 **Air Quality Alert** — AQI is forecast to reach **{peak_3day:.0f}** in the next 3 days. "
                     f"Sensitive groups should limit outdoor activity.")
        elif peak_3day > 100:
            st.warning(f"⚠️ Moderate pollution expected. Max forecast AQI: **{peak_3day:.0f}**.")

        # ── 72-HOUR FORECAST CHART ────────────────────────────────────────────
        st.subheader("📈 72-Hour AQI Forecast")

        # Build color bands
        fig = go.Figure()
        band_colors = [("Good","#00e400",0,50),("Moderate","#ffff00",50,100),
                       ("Unhealthy SG","#ff7e00",100,150),("Unhealthy","#ff0000",150,200),
                       ("Very Unhealthy","#8f3f97",200,300),("Hazardous","#7e0023",300,500)]
        for name, col, lo, hi in band_colors:
            fig.add_hrect(y0=lo, y1=hi, fillcolor=col, opacity=0.08, line_width=0, annotation_text=name,
                          annotation_position="left", annotation_font_size=10)

        # Historical AQI
        if len(history_df):
            fig.add_trace(go.Scatter(
                x=history_df["timestamp"], y=history_df["aqi"],
                name="Historical AQI", line=dict(color="#636EFA", width=2),
                mode="lines",
            ))

        # Forecast
        fig.add_trace(go.Scatter(
            x=df_future["timestamp"], y=df_future["predicted_aqi"],
            name="Forecast AQI", line=dict(color="#EF553B", width=2, dash="dash"),
            mode="lines+markers", marker=dict(size=4),
        ))

        # Now line
        fig.add_vline(x=datetime.now(), line_dash="dot", line_color="gray", annotation_text="Now")

        fig.update_layout(
            xaxis_title="Time", yaxis_title="AQI",
            yaxis_range=[0, max(300, peak_3day + 20)],
            legend=dict(orientation="h", y=1.08),
            height=400, margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── DAILY SUMMARY TABLE ───────────────────────────────────────────────
        st.subheader("📆 3-Day Daily Summary")
        df_future["date"] = df_future["timestamp"].dt.date
        daily = df_future.groupby("date")["predicted_aqi"].agg(["mean","min","max"]).reset_index()
        daily.columns = ["Date","Avg AQI","Min AQI","Max AQI"]
        daily["Status"] = daily["Avg AQI"].apply(lambda v: aqi_info(v)[1])

        def color_status(val):
            c, _, _ = aqi_info(
                daily.loc[daily["Status"]==val,"Avg AQI"].values[0]
            )
            return f"background-color:{c}20; color:black"

        st.dataframe(daily.style.format({"Avg AQI":"{:.0f}","Min AQI":"{:.0f}","Max AQI":"{:.0f}"}),
                     use_container_width=True)

        # ── WEATHER CONTEXT ───────────────────────────────────────────────────
        with st.expander("🌤️ Weather Forecast Context"):
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df_future["timestamp"], y=df_future["temperature"],
                                      name="Temp (°C)", yaxis="y1"))
            fig2.add_trace(go.Bar(x=df_future["timestamp"], y=df_future["precipitation"],
                                  name="Precipitation (mm)", yaxis="y2", opacity=0.4))
            fig2.update_layout(
                yaxis=dict(title="Temperature °C"),
                yaxis2=dict(title="Precipitation mm", overlaying="y", side="right"),
                height=300, margin=dict(l=0,r=0,t=10,b=0),
            )
            st.plotly_chart(fig2, use_container_width=True)

        # ── SHAP FEATURE IMPORTANCE ────────────────────────────────────────────
        with st.expander("🔍 Feature Importance (SHAP)"):
            shap_img = "models/shap_feature_importance.png"
            if os.path.exists(shap_img):
                st.image(shap_img, caption="SHAP Feature Importance")
            else:
                st.info("SHAP plot not available. Run the training pipeline first.")

    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Make sure your HOPSWORKS_API_KEY and AQICN_TOKEN are set as environment variables, "
                "and the feature + training pipelines have run at least once.")
