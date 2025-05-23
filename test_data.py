import pandas as pd
from ml.data import process_data

data = pd.read_csv('data/census.csv')
cat_features = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country'
]
X, y, encoder, lb = process_data(
    data, categorical_features=cat_features, label='salary', training=True
)
print("X shape:", X.shape)
print("First 5 y values:", y[:5])
print("Encoded feature names:", encoder.get_feature_names_out())
