import pandas as pd
import numpy as np

df = pd.read_csv('UTT.csv')
print(df)
mosaic_df = df[(df["Mosaic"] == True)]
large_df = df[df["Width"] == 48]
n_df = df[~((df["Mosaic"] == True) | (df["Width"] == 48))]
print(n_df)

print(df.dtypes)