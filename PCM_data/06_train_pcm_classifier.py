"""
STEP 6 — TRAIN PCM CLASSIFIER
==============================
Trains a classifier to select the optimal PCM based on the next-day weather forecast
and household hot-water demand profile:
  - Input features: GHI_next_day, T_amb_next_day, demand_total_L, T_set, lat, lon,
                    altitude_m, month, DOY, season_code
  - Target label: pcm_label_code (representing the optimal GRG-based PCM product)

Features:
  - Supports XGBoost Classifier (if available) with fallback to Scikit-Learn's
    RandomForestClassifier for environment compatibility.
  - Automatically loads pcm_label_encoder.csv to decode class predictions.
  - Outputs precision, recall, F1 metrics and saves the classification reports.
  - Plots feature importances and confusion matrices for paper validation.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score

# Try to import XGBoost, otherwise fall back gracefully
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ─────────────────────────────────────────────
# PATHS & DIRECTORY SETUP
# ─────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
if os.path.basename(_HERE) == "PCM_data":
    BASE_DIR = os.path.dirname(_HERE)
else:
    BASE_DIR = _HERE

DATASET_CSV   = os.path.join(BASE_DIR, "data", "processed", "classifier_dataset.csv")
ENCODER_CSV   = os.path.join(BASE_DIR, "data", "processed", "pcm_label_encoder.csv")
MODEL_DIR     = os.path.join(BASE_DIR, "models", "classifier")
PLOT_DIR      = os.path.join(BASE_DIR, "data", "plots", "classifier")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# DATA LOADING & PREPARATION
# ─────────────────────────────────────────────

def load_data():
    print(f"[LOAD] Loading dataset from: {DATASET_CSV}")
    df = pd.read_csv(DATASET_CSV)
    
    # Target and features definition
    target_col = "pcm_label_code"
    
    feature_cols = [
        "GHI_next_day", "T_amb_next_day", "demand_total_L", "T_set",
        "lat", "lon", "altitude_m", "month", "DOY", "season_code"
    ]
    
    # Verify presence of all columns
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    # Drop rows where target or key features are missing
    df_clean = df[feature_cols + [target_col]].dropna().reset_index(drop=True)
    
    print(f"  Total samples available: {len(df_clean)}")
    print(f"  Features list: {feature_cols}")
    
    # Display class distribution
    class_counts = df_clean[target_col].value_counts()
    print("\n[INFO] Target Class Distribution:")
    
    # Read classes from encoder map if it exists
    if os.path.exists(ENCODER_CSV):
        enc = pd.read_csv(ENCODER_CSV)
        class_map = dict(zip(enc["pcm_label_code"], enc["pcm_label"]))
        for val, count in class_counts.items():
            print(f"  Class {val} ({class_map.get(val, 'Unknown')}): {count} samples ({count/len(df_clean)*100:.2f}%)")
    else:
        for val, count in class_counts.items():
            print(f"  Class {val}: {count} samples ({count/len(df_clean)*100:.2f}%)")
            
    return df_clean, feature_cols, target_col


# ─────────────────────────────────────────────
# TRAINING AND EVALUATION
# ─────────────────────────────────────────────

def train_classifier(df_clean, feature_cols, target_col):
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    # Stratified split to ensure balanced class distributions in train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n[SPLIT] Training Set Size: {len(X_train)} | Test Set Size: {len(X_test)}")
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    scaler_path = os.path.join(MODEL_DIR, "classifier_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[SCALER] Saved scaler to: {scaler_path}")
    
    # Train model
    if XGB_AVAILABLE:
        print("\n[TRAIN] Training XGBoost Classifier...")
        model = xgb.XGBClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=6,
            random_state=42,
            eval_metric="mlogloss",
            n_jobs=-1
        )
    else:
        print("\n[TRAIN] XGBoost not available in environment. Training Scikit-Learn RandomForestClassifier...")
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
        
    model.fit(X_train_scaled, y_train)
    print("  Model training completed successfully.")
    
    # Save model
    model_path = os.path.join(MODEL_DIR, "pcm_classifier.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved classifier model to: {model_path}")
    
    # Evaluate
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    
    print(f"\n[EVAL] Accuracy: {acc:.4f} | Weighted F1-Score: {f1:.4f}")
    
    # Decode target names for classification report
    target_names = None
    if os.path.exists(ENCODER_CSV):
        enc = pd.read_csv(ENCODER_CSV)
        # Filter classes that are actually in the test set to avoid report warning
        present_classes = sorted(list(y_test.unique()))
        target_names = [enc.loc[enc["pcm_label_code"] == c, "pcm_label"].values[0] for c in present_classes]
    
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=target_names))
    
    return model, scaler, X_test, y_test, preds, target_names


# ─────────────────────────────────────────────
# VISUALIZATION GENERATORS
# ─────────────────────────────────────────────

def plot_feature_importance(model, feature_cols):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="darkgrid")
        
        sns.barplot(
            x=[importances[i] for i in indices],
            y=[feature_cols[i] for i in indices],
            palette="viridis"
        )
        plt.title("Classifier Feature Importance (Decision Boundaries Contribution)")
        plt.xlabel("Relative Importance")
        plt.ylabel("Features")
        plt.tight_layout()
        
        importance_path = os.path.join(PLOT_DIR, "feature_importances.png")
        plt.savefig(importance_path, dpi=150)
        print(f"[PLOT] Saved feature importance plot to: {importance_path}")
        plt.close()


def plot_confusion_matrix_heatmap(y_test, preds, target_names):
    cm = confusion_matrix(y_test, preds)
    
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")
    
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=target_names if target_names else "auto",
        yticklabels=target_names if target_names else "auto"
    )
    plt.title("PCM Classifier Confusion Matrix (Predicted vs Actual Labels)")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    
    cm_path = os.path.join(PLOT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"[PLOT] Saved confusion matrix heatmap to: {cm_path}")
    plt.close()


if __name__ == "__main__":
    if not os.path.exists(DATASET_CSV):
        print(f"[ERROR] Dataset file not found at {DATASET_CSV}. Please run 04_fuse_data.py first.")
    else:
        df_clean, feature_cols, target_col = load_data()
        model, scaler, X_test, y_test, preds, target_names = train_classifier(df_clean, feature_cols, target_col)
        
        # Save feature list for reference during inference
        with open(os.path.join(MODEL_DIR, "classifier_features.pkl"), "wb") as f:
            pickle.dump(feature_cols, f)
            
        plot_feature_importance(model, feature_cols)
        plot_confusion_matrix_heatmap(y_test, preds, target_names)
        print("\n✅ PCM Classifier Model Training Complete.")
