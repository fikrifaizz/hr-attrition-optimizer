import pandas as pd
import os
from pathlib import Path
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

KAGGLE_DATASET_ID = "pavansubhasht/ibm-hr-analytics-attrition-dataset"

PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
FINAL_FILENAME = "hr_raw.csv"
DESTINATION_PATH = RAW_DATA_DIR / FINAL_FILENAME

def extract_data():
    print(f"\nMemulai proses download dari Kaggle...")
    print(f"Dataset ID: {KAGGLE_DATASET_ID}")
    
    if not RAW_DATA_DIR.exists():
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DESTINATION_PATH.exists():
        print(f"File '{FINAL_FILENAME}' sudah ada. Skip download.")
        return pd.read_csv(DESTINATION_PATH)
    try:
        api = KaggleApi()
        api.authenticate() 
        
        print(f"Sedang mengunduh data... (Mungkin butuh waktu tergantung internet)")
        api.dataset_download_files(
            KAGGLE_DATASET_ID, 
            path=RAW_DATA_DIR, 
            unzip=True
        )
        print("Download & Unzip selesai.")
        
    except Exception as e:
        print(f"Gagal download dari Kaggle. Pastikan kaggle.json sudah benar.")
        print(f"Error detail: {e}")
        sys.exit(1)

    try:
        downloaded_files = list(RAW_DATA_DIR.glob("*.csv"))
        target_file = None
        for f in downloaded_files:
            if f.name != FINAL_FILENAME:
                target_file = f
                break
        
        if target_file:
            print(f"Mengubah nama {target_file.name} -> {FINAL_FILENAME}")
            os.rename(target_file, DESTINATION_PATH)
        elif not DESTINATION_PATH.exists():
            print("File CSV tidak ditemukan setelah download.")
            sys.exit(1)
            
    except Exception as e:
        print(f"Gagal melakukan rename file: {e}")
        sys.exit(1)

    df = pd.read_csv(DESTINATION_PATH)
    print(f"Data siap digunakan! Shape: {df.shape}")
    return df