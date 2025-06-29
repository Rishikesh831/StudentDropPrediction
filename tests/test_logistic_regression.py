import numpy as np
from src.models.logistic_regression import LogisticRegression

def test_logistic_regression_fit_predict():
    # Simple dataset: AND logic
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,0,0,1])
    model = LogisticRegression(learning_rate=0.1, epochs=10000, threshold=0.5)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.array_equal(preds, y), f"Expected {y}, got {preds}"

def test_predict_proba_shape():
    X = np.random.rand(5, 3)
    y = np.array([0,1,0,1,0])
    model = LogisticRegression(learning_rate=0.1, epochs=1000)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (5,), f"Expected shape (5,), got {probs.shape}" 