import pandas as pd

def filter_by_names(csv1, csv2):
    # Load both CSV files
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)
    
    # Ensure column names match (assuming 'name' is the column with movie/show names)
    df1['Title'] = df1['Title'].astype(str).str.strip()
    df2['title'] = df2['title'].astype(str).str.strip()
    
    # Filter df1 to keep only rows where 'name' is in df2
    df_filtered = df1[df1['Title'].apply(lambda x: any(name in x for name in df2['title']))]
    
    return df_filtered

# Example usage
df_filtered = filter_by_names("raw/engagement_2023_h2.csv", "raw/titles_netflix.csv")
df_filtered.to_csv("limited_engagement_2023_h2.csv")