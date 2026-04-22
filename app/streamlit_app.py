import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
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
st.cache_data.clear()
CITY        = os.environ.get("CITY", "islamabad")
AQICN_TOKEN = os.environ.get("AQICN_TOKEN", "")

AQI_LEVELS = [
    (0,   50,  "#00e400", "Good",                           "Air quality is satisfactory."),
    (51,  100, "#ffff00", "Moderate",                       "Acceptable quality; some pollutants may concern sensitive groups."),
    (101, 150, "#ff7e00", "Unhealthy for Sensitive Groups",  "Members of sensitive groups may experience effects."),
    (151, 200, "#ff0000", "Unhealthy",                      "Everyone may begin to experience health effects."),
    (201, 300, "#8f3f97", "Very Unhealthy",                 "Health alert — everyone may experience serious effects."),
    (301, 500, "#7e0023", "Hazardous",                      "Emergency conditions. Entire population is likely affected."),
]

def aqi_info(value: float):
    for lo, hi, color, label, desc in AQI_LEVELS:
        if lo <= value <= hi:
            return color, label, desc
    return "#7e0023", "Hazardous", "Extreme pollution."


# ── DATA LOADERS ─────────────────────────────────────────────────────────────
@st.cache_resource(ttl=3600)
def load_model_and_features():
    project = hopsworks.login(api_key_value=HOPSWORKS_KEY)
    mr = project.get_model_registry()
    model_meta = mr.get_model("aqi_predictor", version=3)
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
    df = df.sort_values("timestamp").tail(72)
    return df


@st.cache_data(ttl=3600)
def fetch_future_weather() -> pd.DataFrame:
    lat = 33.72148
    lon = 73.04329
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,relativehumidity_2m,windspeed_10m,precipitation"
        f"&forecast_days=4&timezone=UTC"
    )
    resp   = requests.get(url, timeout=10)
    data   = resp.json()
    hourly = data.get("hourly", {})
    df = pd.DataFrame({
        "timestamp":     pd.to_datetime(hourly["time"]),
        "temperature":   hourly["temperature_2m"],
        "humidity":      hourly["relativehumidity_2m"],
        "windspeed":     hourly["windspeed_10m"],
        "precipitation": hourly["precipitation"],
    })
    return df.head(72)


@st.cache_data(ttl=1800)
def fetch_real_aqi(city: str, token: str) -> float:
    try:
        url  = f"https://api.waqi.info/feed/{city}/?token={token}"
        resp = requests.get(url, timeout=10).json()
        if resp["status"] == "ok":
            return float(resp["data"]["aqi"])
    except Exception:
        pass
    return None


def build_forecast_features(future_weather: pd.DataFrame, history: pd.DataFrame, feature_names: list):
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

    last_aqi = history["aqi"].iloc[-1] if "aqi" in history.columns and len(history) else 100.0
    df["aqi_lag_1h"]      = last_aqi
    df["aqi_lag_3h"]      = last_aqi
    df["aqi_lag_6h"]      = last_aqi
    df["aqi_lag_24h"]     = last_aqi
    df["aqi_change_rate"] = 0.0
    df["aqi_rolling_6h"]  = last_aqi
    df["aqi_rolling_24h"] = last_aqi

    for col in ["pm25", "pm10", "no2", "o3", "co", "so2"]:
        df[col] = history[col].median() if col in history.columns else np.nan

    available = [c for c in feature_names if c in df.columns]
    df_model  = df[available].fillna(df[available].median())
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

# ── MAIN ──────────────────────────────────────────────────────────────────────
st.title(f"🌫️ AQI Forecast — {city_input}")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

with st.spinner("Loading model and data..."):
    try:
        model, feature_names, metrics = load_model_and_features()
        history_df = load_recent_features(city_input)
        future_wx  = fetch_future_weather()
        df_future, X_future, used_features = build_forecast_features(future_wx, history_df, feature_names)

        predictions = model.predict(X_future.values)
        df_future["predicted_aqi"] = np.clip(predictions, 0, 500)

        # Use real AQI from AQICN for current reading
        real_aqi = fetch_real_aqi(city_input, AQICN_TOKEN)
        current_aqi = real_aqi if real_aqi is not None else float(predictions[0])
        color, label, desc = aqi_info(current_aqi)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f"""<div style='background:{color};border-radius:12px;padding:20px;text-align:center;'>
                <h1 style='color:black;margin:0;font-size:64px'>{int(current_aqi)}</h1>
                <h3 style='color:black;margin:4px'>{label}</h3>
                <p style='color:black;font-size:13px'>{desc}</p></div>""",
                unsafe_allow_html=True,
            )
        with col2:
            st.metric("📊 Model RMSE", f"{metrics.get('rmse', 0):.2f}")
            st.metric("📐 R² Score",   f"{metrics.get('r2', 0):.3f}")
        with col3:
            avg_3day  = df_future["predicted_aqi"].mean()
            peak_3day = df_future["predicted_aqi"].max()
            st.metric("📅 Avg (3-day forecast)", f"{avg_3day:.0f}")
            st.metric("⚠️ Peak AQI",             f"{peak_3day:.0f}")

        st.markdown("---")

        if peak_3day > 150:
            st.error(f"🚨 **Air Quality Alert** — AQI forecast to reach **{peak_3day:.0f}**. Limit outdoor activity.")
        elif peak_3day > 100:
            st.warning(f"⚠️ Moderate pollution expected. Max forecast AQI: **{peak_3day:.0f}**.")

        # ── 72-HOUR CHART ─────────────────────────────────────────────────────
        st.subheader("📈 72-Hour AQI Forecast")
        fig = go.Figure()
        bands = [("Good","#00e400",0,50),("Moderate","#ffff00",50,100),
                 ("Unhealthy SG","#ff7e00",100,150),("Unhealthy","#ff0000",150,200),
                 ("Very Unhealthy","#8f3f97",200,300),("Hazardous","#7e0023",300,500)]
        for bname, bcol, lo, hi in bands:
            fig.add_hrect(y0=lo, y1=hi, fillcolor=bcol, opacity=0.08, line_width=0,
                          annotation_text=bname, annotation_position="left", annotation_font_size=10)
        if len(history_df):
            fig.add_trace(go.Scatter(x=history_df["timestamp"], y=history_df["aqi"],
                                     name="Historical AQI", line=dict(color="#636EFA", width=2)))
        fig.add_trace(go.Scatter(x=df_future["timestamp"], y=df_future["predicted_aqi"],
                                 name="Forecast AQI", line=dict(color="#EF553B", width=2, dash="dash"),
                                 mode="lines+markers", marker=dict(size=4)))
        fig.add_vline(x=pd.Timestamp.now().timestamp() * 1000, line_dash="dot",
                      line_color="gray", annotation_text="Now")
        fig.update_layout(xaxis_title="Time", yaxis_title="AQI",
                          yaxis_range=[0, 400],
                          legend=dict(orientation="h", y=1.08),
                          height=400, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # ── EDA SECTION ───────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📊 Exploratory Data Analysis")
        try:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**AQI Distribution**")
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(x=history_df["aqi"], nbinsx=20, marker_color="#636EFA"))
                fig_hist.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0),
                                       xaxis_title="AQI", yaxis_title="Count")
                st.plotly_chart(fig_hist, use_container_width=True)
            with col2:
                st.markdown("**AQI Over Time**")
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(x=history_df["timestamp"], y=history_df["aqi"],
                                               mode="lines", line=dict(color="#EF553B")))
                fig_trend.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0),
                                        xaxis_title="Time", yaxis_title="AQI")
                st.plotly_chart(fig_trend, use_container_width=True)
            col3, col4 = st.columns(2)
            with col3:
                st.markdown("**AQI by Hour of Day**")
                hourly_avg = history_df.groupby("hour")["aqi"].mean().reset_index()
                fig_hour = go.Figure()
                fig_hour.add_trace(go.Bar(x=hourly_avg["hour"], y=hourly_avg["aqi"],
                                          marker_color="#00CC96"))
                fig_hour.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0),
                                       xaxis_title="Hour", yaxis_title="Avg AQI")
                st.plotly_chart(fig_hour, use_container_width=True)
            with col4:
                st.markdown("**Temperature vs AQI**")
                fig_scatter = go.Figure()
                fig_scatter.add_trace(go.Scatter(x=history_df["temperature"], y=history_df["aqi"],
                                                 mode="markers", marker=dict(color="#AB63FA", size=4)))
                fig_scatter.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0),
                                          xaxis_title="Temperature (°C)", yaxis_title="AQI")
                st.plotly_chart(fig_scatter, use_container_width=True)
            st.markdown("**📈 AQI Statistics**")
            stats = history_df["aqi"].describe().round(2)
            st.dataframe(stats.to_frame().T, use_container_width=True)
        except Exception as eda_err:
            st.warning(f"EDA error: {eda_err}")

        # ── DAILY SUMMARY ─────────────────────────────────────────────────────
        st.subheader("📆 3-Day Daily Summary")
        df_future["date"] = df_future["timestamp"].dt.date
        daily = df_future.groupby("date")["predicted_aqi"].agg(["mean","min","max"]).reset_index()
        daily.columns = ["Date", "Avg AQI", "Min AQI", "Max AQI"]
        daily["Status"] = daily["Avg AQI"].apply(lambda v: aqi_info(v)[1])
        st.dataframe(daily.style.format({"Avg AQI":"{:.0f}","Min AQI":"{:.0f}","Max AQI":"{:.0f}"}),
                     use_container_width=True)

        # ── WEATHER CONTEXT ───────────────────────────────────────────────────
        with st.expander("🌤️ Weather Forecast Context"):
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df_future["timestamp"], y=df_future["temperature"],
                                      name="Temp (°C)", yaxis="y1"))
            fig2.add_trace(go.Bar(x=df_future["timestamp"], y=df_future["precipitation"],
                                  name="Precipitation (mm)", yaxis="y2", opacity=0.4))
            fig2.update_layout(yaxis=dict(title="Temperature °C"),
                               yaxis2=dict(title="Precipitation mm", overlaying="y", side="right"),
                               height=300, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig2, use_container_width=True)

        with st.expander("🔍 Feature Importance (SHAP)"):
            try:
                import shap
                explainer  = shap.Explainer(model.predict, X_future.values)
                shap_values = explainer(X_future.values)
                importance  = np.abs(shap_values.values).mean(axis=0)
                shap_df = pd.DataFrame({
                    "Feature":    used_features,
                    "Importance": importance
                }).sort_values("Importance", ascending=True)
                fig_shap = go.Figure()
                fig_shap.add_trace(go.Bar(x=shap_df["Importance"], y=shap_df["Feature"],
                                          orientation="h", marker_color="#EF553B"))
                fig_shap.update_layout(title="SHAP Feature Importance",
                                       xaxis_title="Mean |SHAP Value|",
                                       yaxis_title="Feature",
                                       height=500, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig_shap, use_container_width=True)
            except Exception as shap_err:
                st.warning(f"SHAP error: {shap_err}")

    except Exception as e:
        import traceback
        st.error(f"Error loading data: {e}")
        st.code(traceback.format_exc())
