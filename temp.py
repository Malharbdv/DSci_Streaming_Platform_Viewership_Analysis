import pandas as pd

df = pd.read_csv("countries_clamped.csv")
print(df.country_name.unique())