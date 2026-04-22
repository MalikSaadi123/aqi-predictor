import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import shap
import hopsworks

HOPSWORKS_KEY = os.environ["HOPSWORKS_API_KEY"]

FEATURE_COLS = [
    "temperature", "humidity", "windspeed", "precipitation",
    "hour", "day", "month", "weekday", "is_weekend",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_lag_24h",
    "aqi_change_rate", "aqi_rolling_6h", "aqi_rolling_24h",
    "pm25", "pm10", "no2", "o3", "co", "so2",
]
TARGET_COL = "aqi"


# ── 1. FETCH FEATURES FROM HOPSWORKS ────────────────────────────────────────
def load_training_data() -> pd.DataFrame:
    project = hopsworks.login(api_key_value=HOPSWORKS_KEY)
    fs = project.get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=1)
    df = fg.read()
    print(f"Loaded {len(df)} rows from feature store.")
    return df


def prepare_data(df: pd.DataFrame):
    available = [c for c in FEATURE_COLS if c in df.columns]
    df = df[available + [TARGET_COL]].dropna(subset=[TARGET_COL])

    # Fill pollutant NaNs with column median
    df[available] = df[available].fillna(df[available].median())

    X = df[available].values
    y = df[TARGET_COL].values
    return X, y, available


# ── 2. TRAIN & EVALUATE MODELS ───────────────────────────────────────────────
def evaluate(model, X_test, y_test, name: str) -> dict:
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)
    print(f"  {name:35s} → RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.4f}")
    return {"name": name, "rmse": rmse, "mae": mae, "r2": r2, "model": model}


def train_all_models(X_train, X_test, y_train, y_test) -> list[dict]:
    results = []

    # Ridge Regression
    ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
    ridge.fit(X_train, y_train)
    results.append(evaluate(ridge, X_test, y_test, "Ridge Regression"))

    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    results.append(evaluate(rf, X_test, y_test, "Random Forest"))

    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    results.append(evaluate(gb, X_test, y_test, "Gradient Boosting"))

    # Try TensorFlow LSTM if available
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, LSTM
        from tensorflow.keras.callbacks import EarlyStopping

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)
        X_tr_lstm = X_tr_s.reshape((X_tr_s.shape[0], 1, X_tr_s.shape[1]))
        X_te_lstm = X_te_s.reshape((X_te_s.shape[0], 1, X_te_s.shape[1]))

        lstm_model = Sequential([
            LSTM(64, input_shape=(1, X_train.shape[1]), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        lstm_model.compile(optimizer="adam", loss="mse")
        lstm_model.fit(
            X_tr_lstm, y_train,
            epochs=50, batch_size=32, verbose=0,
            validation_split=0.1,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        )

        class LSTMWrapper:
            """Wrap Keras model + scaler so it has a sklearn-like .predict()."""
            def __init__(self, model, scaler):
                self.model = model
                self.scaler = scaler
            def predict(self, X):
                Xs = self.scaler.transform(X)
                Xs = Xs.reshape((Xs.shape[0], 1, Xs.shape[1]))
                return self.model.predict(Xs, verbose=0).flatten()

        wrapper = LSTMWrapper(lstm_model, scaler)
        results.append(evaluate(wrapper, X_test, y_test, "LSTM (TensorFlow)"))
    except ImportError:
        print("  TensorFlow not installed — skipping LSTM.")

    return results


# ── 3. SHAP FEATURE IMPORTANCE ───────────────────────────────────────────────
def compute_shap(model, X_train, feature_names: list, output_dir: str = "models"):
    os.makedirs(output_dir, exist_ok=True)
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_train[:200])  # sample for speed
        shap.summary_plot(shap_vals, X_train[:200], feature_names=feature_names,
                          show=False, plot_type="bar")
        import matplotlib.pyplot as plt
        plt.tight_layout()
        plt.savefig(f"{output_dir}/shap_feature_importance.png", dpi=120)
        plt.close()
        print("  SHAP plot saved.")
    except Exception as e:
        print(f"  SHAP skipped: {e}")


# ── 4. SAVE BEST MODEL TO HOPSWORKS MODEL REGISTRY ──────────────────────────
def save_to_registry(model, feature_names: list, metrics: dict):
    import os, json, shutil

    # Save all files into one folder
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/best_model.pkl")

    with open("models/feature_names.json", "w") as f:
        json.dump(feature_names, f)

    with open("models/metrics.json", "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f)

    project = hopsworks.login(api_key_value=HOPSWORKS_KEY)
    mr = project.get_model_registry()

    # Auto-increment version
    version = 1
    while True:
        try:
            mr.get_model("aqi_predictor", version=version)
            version += 1
        except Exception:
            break

    model_dir = mr.sklearn.create_model(
        name="aqi_predictor",
        version=version,
        metrics=metrics,
        description=f"Best AQI model: {type(model).__name__}",
    )
    model_dir.save("models")   # ← saves entire folder at once
    print(f"✅ Model saved as version {version}!")


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n[{datetime.utcnow()}] Starting training pipeline...")
    os.makedirs("models", exist_ok=True)

    df = load_training_data()
    X, y, feature_names = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\nTraining on {len(X_train)} samples, testing on {len(X_test)} samples.")
    print(f"Features: {feature_names}\n")

    results = train_all_models(X_train, X_test, y_train, y_test)

    # Pick best model by RMSE
    best = min(results, key=lambda r: r["rmse"])
    print(f"\n🏆 Best model: {best['name']}  (RMSE={best['rmse']:.2f})")

    # SHAP (tree-based models only)
    if hasattr(best["model"], "feature_importances_"):
        compute_shap(best["model"], X_train, feature_names)

    metrics = {"rmse": best["rmse"], "mae": best["mae"], "r2": best["r2"]}
    save_to_registry(best, feature_names, metrics)
    print("\nTraining pipeline complete.")
