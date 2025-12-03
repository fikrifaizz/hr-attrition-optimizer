import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from xgboost import XGBClassifier

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_DIR / "data" / "processed" / "hr_cleaned.parquet"
MODEL_DIR = PROJECT_DIR / "data" / "models"
MODEL_PATH = MODEL_DIR / "xgboost_final.pkl"
COLS_PATH = MODEL_DIR / "model_columns.pkl"

def train_final_model():
    print("=== MEMULAI TRAINING FINAL ===")
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data tidak ditemukan di {DATA_PATH}")
    
    df = pd.read_parquet(DATA_PATH)
    print(f"Data loaded. Shape: {df.shape}")

    X = df.drop(columns=['attrition'])
    y = df['attrition']
    
    X_encoded = pd.get_dummies(X, drop_first=True)
    feature_names = X_encoded.columns.tolist()
    print("Melatih XGBoost dengan Best Params...")

    model = XGBClassifier(
        learning_rate=0.01,
        max_depth=3,
        n_estimators=100,
        subsample=0.8,
        scale_pos_weight=7.78, # Hasil tuning user
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_encoded, y)
    print("Model trained.")

    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_names, COLS_PATH)
    
    print(f"Model disimpan di: {MODEL_PATH}")
    print(f"Kolom fitur disimpan di: {COLS_PATH}")

if __name__ == "__main__":
    train_final_model()