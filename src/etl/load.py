import pandas as pd
from pathlib import Path
import os

# Konfigurasi Path Output
PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_FILENAME = "hr_cleaned.parquet"
OUTPUT_PATH = PROCESSED_DATA_DIR / OUTPUT_FILENAME

def load_data(df: pd.DataFrame):
    print("Memulai proses penyimpanan...")
    
    if not PROCESSED_DATA_DIR.exists():
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
    try:
        df.to_parquet(OUTPUT_PATH, index=False)
        print(f"Data tersimpan di: {OUTPUT_PATH}")
    except Exception as e:
        print(f"Gagal menyimpan data: {e}")
        raise
