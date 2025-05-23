import pandas as pd
from ml.data import process_data
from ml.model import train_model, inference, save_model, load_model

data = pd.read_csv('data/census.csv')
cat_features = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country'
]
X, y, encoder, lb = process_data(
    data, categorical_features=cat_features, label='salary', training=True
)
model = train_model(X, y)
preds = inference(model, X)
save_model(model, 'model/model.pkl')
save_model(encoder, 'model/encoder.pkl')
save_model(lb, 'model/lb.pkl')
loaded_model = load_model('model/model.pkl')
loaded_encoder = load_model('model/encoder.pkl')
loaded_lb = load_model('model/lb.pkl')
print("Model trained and saved successfully!")
print("First 5 predictions:", preds[:5])
