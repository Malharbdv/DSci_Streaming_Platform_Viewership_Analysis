import pandas as pd

df = pd.read_excel("engagement_2023_h1.xlsx")
df = df.iloc[5:]  # Removes the first 5 rows
df.columns = ['None', 'Title', 'Available Globally?', 'Release Date', 'Hours Viewed']  # Ensure the number of names matches the number of columns
df.reset_index(drop=True, inplace=True)  # Reset index after deletion
df.to_csv("engagement_2023_h1.csv")

df = pd.read_excel("engagement_2023_h2.xlsx")
df = df.iloc[5:]  # Removes the first 5 rows
df.columns = ['None', 'Title', 'Available Globally?', 'Release Date', 'Hours Viewed', 'runtime', 'views']  # Ensure the number of names matches the number of columns
df.reset_index(drop=True, inplace=True)  # Reset index after deletion
df.to_csv("engagement_2023_h2.csv")