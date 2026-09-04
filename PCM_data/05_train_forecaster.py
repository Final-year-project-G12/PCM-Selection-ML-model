"""
STEP 5 — TRAIN CLIMATE FORECASTER
=================================
Trains a model to predict next-day climate parameters:
  - GHI_next_day  (mean Global Horizontal Irradiance)
  - T_amb_next_day (mean Ambient Temperature)

Features used for forecasting:
  - Current day averages: GHI_mean, T_amb_mean, T_amb_max, CSI_mean, cloud_cover_mean, RHum_mean, W_spd_mean
  - Lagged features (1, 2, and 7 days ago): GHI_mean_lag1, GHI_mean_lag2, GHI_mean_lag7, etc.
  - Rolling features: GHI_mean_roll7, T_amb_mean_roll7
  - Temporal metadata: month, DOY

Models are evaluated on a temporal split (80% train, 20% validation per city)
using MAE, RMSE, and R2 score.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────────
# PATHS & DIRECTORY SETUP
# ─────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
if os.path.basename(_HERE) == "PCM_data":
    BASE_DIR = os.path.dirname(_HERE)
else:
    BASE_DIR = _HERE

DATASET_CSV = os.path.join(BASE_DIR, "data", "processed", "classifier_dataset.csv")
MODEL_DIR   = os.path.join(BASE_DIR, "models", "forecaster")
PLOT_DIR    = os.path.join(BASE_DIR, "data", "plots", "forecaster")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# DATA PREPARATION
# ─────────────────────────────────────────────

def load_and_preprocess_data(csv_path):
    print(f"[LOAD] Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Sort chronologically to preserve time sequence
    df = df.sort_values(by=["city", "date"]).reset_index(drop=True)
    
    # Identify forecasting features and targets
    feature_cols = [
        "GHI_mean", "T_amb_mean", "T_amb_max", "CSI_mean", "cloud_cover_mean",
        "RHum_mean", "W_spd_mean", "month", "DOY",
        "GHI_mean_lag1", "GHI_mean_lag2", "GHI_mean_lag7",
        "T_amb_mean_lag1", "T_amb_mean_lag2", "T_amb_mean_lag7",
        "GHI_mean_roll7", "T_amb_mean_roll7"
    ]
    
    # Keep only columns that exist in the dataset
    feature_cols = [c for c in feature_cols if c in df.columns]
    targets = ["GHI_next_day", "T_amb_next_day"]
    
    # Filter columns and drop rows with missing values (due to lags and targets)
    all_cols = ["city", "date"] + feature_cols + targets
    df_clean = df[all_cols].dropna().reset_index(drop=True)
    
    print(f"  Total rows before dropna: {len(df)}")
    print(f"  Total rows after dropna:  {len(df_clean)}")
    print(f"  Features used: {feature_cols}")
    
    return df_clean, feature_cols, targets


def split_data(df, feature_cols, targets):
    """
    Perform a chronological split: 80% train, 20% validation per city.
    """
    train_dfs = []
    val_dfs = []
    
    for city, city_df in df.groupby("city"):
        city_df = city_df.sort_values("date")
        n = len(city_df)
        split_idx = int(n * 0.8)
        
        train_dfs.append(city_df.iloc[:split_idx])
        val_dfs.append(city_df.iloc[split_idx:])
        
    train_data = pd.concat(train_dfs, ignore_index=True)
    val_data = pd.concat(val_dfs, ignore_index=True)
    
    X_train = train_data[feature_cols]
    y_train = train_data[targets]
    
    X_val = val_data[feature_cols]
    y_val = val_data[targets]
    
    print(f"[SPLIT] Train set size: {len(X_train)} | Val set size: {len(X_val)}")
    return X_train, X_val, y_train, y_val, val_data


# ─────────────────────────────────────────────
# MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────

def train_and_save_models(X_train, y_train, X_val, y_val, feature_cols, targets):
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Save scaler
    scaler_path = os.path.join(MODEL_DIR, "forecaster_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[SCALER] Saved scaler to: {scaler_path}")
    
    models = {}
    metrics_summary = {}
    
    for target in targets:
        print(f"\n[TRAIN] Training forecaster for target: {target}...")
        
        # Primary Model: Multi-Layer Perceptron (Neural Network Regressor)
        # Fallback Model: Random Forest Regressor
        model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=300,
            random_state=42,
            early_stopping=True
        )
        
        y_train_target = y_train[target].values
        y_val_target = y_val[target].values
        
        try:
            model.fit(X_train_scaled, y_train_target)
            print("  Neural Network (MLP) model converged successfully.")
        except Exception as e:
            print(f"  MLP failed to train: {e}. Falling back to Random Forest.")
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train_scaled, y_train_target)
            
        # Predict & Evaluate
        preds = model.predict(X_val_scaled)
        
        mae = mean_absolute_error(y_val_target, preds)
        rmse = np.sqrt(mean_squared_error(y_val_target, preds))
        r2 = r2_score(y_val_target, preds)
        
        metrics_summary[target] = {"MAE": mae, "RMSE": rmse, "R2": r2}
        print(f"  Validation Metrics:")
        print(f"    Mean Absolute Error (MAE) : {mae:.4f}")
        print(f"    Root Mean Sq. Error (RMSE): {rmse:.4f}")
        print(f"    R2 Score                  : {r2:.4f}")
        
        # Save model
        model_path = os.path.join(MODEL_DIR, f"forecaster_{target}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"  Saved model to: {model_path}")
        
        models[target] = model
        
    return models, scaler, metrics_summary


def plot_predictions(val_data, X_val, models, scaler, targets):
    """
    Generate plots of predicted vs actual weather parameters.
    """
    X_val_scaled = scaler.transform(X_val)
    
    # Choose a city to plot (e.g. Coimbatore or the first city)
    sample_city = val_data["city"].unique()[0]
    city_mask = val_data["city"] == sample_city
    city_val_data = val_data[city_mask].copy()
    
    X_city_scaled = scaler.transform(city_val_data[X_val.columns])
    
    plt.figure(figsize=(15, 10))
    sns.set_theme(style="darkgrid")
    
    for i, target in enumerate(targets, 1):
        plt.subplot(2, 1, i)
        
        actual = city_val_data[target].values
        predicted = models[target].predict(X_city_scaled)
        dates = pd.to_datetime(city_val_data["date"])
        
        # Plot last 100 days of predictions for readability
        plot_len = min(100, len(actual))
        
        plt.plot(dates.iloc[-plot_len:], actual[-plot_len:], label="Actual", color="#1f77b4", linewidth=2)
        plt.plot(dates.iloc[-plot_len:], predicted[-plot_len:], label="Predicted", color="#ff7f0e", linestyle="--", linewidth=2)
        
        plt.title(f"{target} Forecast Verification - {sample_city} (Last {plot_len} Days)")
        plt.xlabel("Date")
        if "GHI" in target:
            plt.ylabel("Solar Irradiance (W/m²)")
        else:
            plt.ylabel("Ambient Temperature (°C)")
        plt.legend()
        
    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, "forecaster_predictions.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\n[PLOT] Saved verification chart to: {plot_path}")
    plt.close()


if __name__ == "__main__":
    if not os.path.exists(DATASET_CSV):
        print(f"[ERROR] Dataset file not found at {DATASET_CSV}. Please run 04_fuse_data.py first.")
    else:
        df, feature_cols, targets = load_and_preprocess_data(DATASET_CSV)
        X_train, X_val, y_train, y_val, val_data = split_data(df, feature_cols, targets)
        models, scaler, metrics = train_and_save_models(X_train, y_train, X_val, y_val, feature_cols, targets)
        
        # Save feature list for reference during inference
        with open(os.path.join(MODEL_DIR, "forecaster_features.pkl"), "wb") as f:
            pickle.dump(feature_cols, f)
            
        plot_predictions(val_data, X_val, models, scaler, targets)
        print("\n✅ Climate Forecaster Training Complete.")
