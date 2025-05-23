import pandas as pd

df = pd.read_csv('data/census.csv')
print(df.head())
print(df.columns)
print(df['salary'].value_counts())
print(df['workclass'].value_counts())
