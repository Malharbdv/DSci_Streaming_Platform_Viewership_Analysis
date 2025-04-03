import pandas as pd

df = pd.read_csv("cleaned.csv")
df.info()

df = pd.read_csv("countries_clamped.csv")
df.info()

df = pd.read_csv("global_clamped.csv")
df.info()
