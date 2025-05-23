import pytest
# TODO: add necessary import
import pandas as pd
import numpy as np
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics, performance_on_categorical_slice

@pytest.fixture
def data():
    """Fixture to load sample data."""
    return pd.read_csv('data/census.csv')

# TODO: implement the first test. Change the function name and input as needed
def test_process_data(data):
    """Test that process_data returns correct shapes and types."""
    cat_features = [
        'workclass', 'education', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'native-country'
    ]
    X, y, encoder, lb = process_data(data, categorical_features=cat_features, 
                                    label='salary', training=True)
    assert isinstance(X, np.ndarray), "X should be a numpy array"
    assert isinstance(y, np.ndarray), "y should be a numpy array"
    assert X.shape[0] == y.shape[0], "X and y should have same number of rows"
    assert X.shape[1] == 108, "X should have 108 columns after one-hot encoding"

# TODO: implement the second test. Change the function name and input as needed
def test_train_model(data):
    """Test that train_model returns a fitted model."""
    cat_features = [
        'workclass', 'education', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'native-country'
    ]
    X, y, _, _ = process_data(data, categorical_features=cat_features, 
                             label='salary', training=True)
    model = train_model(X, y)
    assert hasattr(model, 'predict'), "Model should have a predict method"
    assert model.n_estimators > 0, "Model should be fitted"


# TODO: implement the third test. Change the function name and input as needed
def test_performance_on_categorical_slice(data):
    """Test that performance_on_categorical_slice returns valid metrics."""
    cat_features = [
        'workclass', 'education', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'native-country'
    ]
    X, y, encoder, lb = process_data(data, categorical_features=cat_features, 
                                    label='salary', training=True)
    model = train_model(X, y)
    precision, recall, fbeta = performance_on_categorical_slice(
        data, 'workclass', 'Private', cat_features, 'salary', encoder, lb, model
    )
    assert 0 <= precision <= 1, "Precision should be between 0 and 1"
    assert 0 <= recall <= 1, "Recall should be between 0 and 1"
    assert 0 <= fbeta <= 1, "F1 should be between 0 and 1"
