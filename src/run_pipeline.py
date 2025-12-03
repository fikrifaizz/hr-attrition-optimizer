from etl.extract import extract_data
from etl.transform import transform_data
from etl.load import load_data

def main():
    print("=== STARTING ETL PIPELINE ===")
    
    # 1. Extract
    df_raw = extract_data()
    
    # 2. Transform
    df_clean = transform_data(df_raw)
    
    # 3. Load
    load_data(df_clean)
    
    print("=== ETL PIPELINE COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()