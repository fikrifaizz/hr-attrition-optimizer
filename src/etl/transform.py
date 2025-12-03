import pandas as pd

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Memulai pembersihan data...")
    
    useless_cols = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
    cols_to_drop = [c for c in useless_cols if c in df.columns]
    
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"Dropped columns: {cols_to_drop}")
    
    if 'Attrition' in df.columns:
        if df['Attrition'].dtype == 'object':
            df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
            print("Encoded 'Attrition' to binary (1/0)")
            
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    
    print(f"Data cleaned. Current Shape: {df.shape}")
    return df