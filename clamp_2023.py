import pandas as pd

def filter_2023(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)

    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    
    else:
        raise ValueError("Unsupported file format. Use CSV or Excel.")
    
    df['week'] = pd.to_datetime(df['week'], format='%Y-%m-%d', errors='coerce')
    
    df = df[(df['week'].dt.year == 2023) & df['week'].dt.month.between(1, 6)]
    
    return df

df_filtered = filter_2023("raw/all-weeks-countries.xlsx")
df_filtered.to_csv("countries_clamped.csv")

df_filtered = filter_2023("raw/all-weeks-global.xlsx")
df_filtered.to_csv("global_clamped.csv")
