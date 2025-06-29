"""
Scratch Logistic Regression Model for Student Dropout Prediction
"""

import numpy as np
import pickle
from typing import Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class LogisticRegression:
    """
    Scratch implementation of Logistic Regression for binary classification
    """
    
    def __init__(self, learning_rate: float = 0.02, epochs: int = 100000, threshold: float = 0.4):
        """
        Initialize Logistic Regression model
        
        Args:
            learning_rate: Learning rate for gradient descent
            epochs: Number of training epochs
            threshold: Classification threshold
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.threshold = threshold
        self.w = None
        self.b = None
        self.costs = []
        self.mean = None
        self.std = None
        
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function
        
        Args:
            z: Input values
            
        Returns:
            Sigmoid output
        """
        return 1 / (1 + np.exp(-z))
    
    def compute_cost(self, X: np.ndarray, w: np.ndarray, y: np.ndarray, b: float) -> float:
        """
        Compute logistic regression cost function
        
        Args:
            X: Input features
            w: Weight vector
            y: Target labels
            b: Bias term
            
        Returns:
            Cost value
        """
        m = X.shape[0]
        
        # Compute predictions
        z = np.dot(X, w) + b
        y_hat = self.sigmoid(z)
        
        # Clip values to avoid log(0)
        y_hat = np.clip(y_hat, 1e-8, 1 - 1e-8)
        
        # Compute cost
        total_cost = - (1/m) * np.sum(
            y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)
        )
        
        return total_cost
    
    def compute_gradients(self, X: np.ndarray, w: np.ndarray, y: np.ndarray, b: float) -> Tuple[np.ndarray, float]:
        """
        Compute gradients for logistic regression
        
        Args:
            X: Input features
            w: Weight vector
            y: Target labels
            b: Bias term
            
        Returns:
            Tuple of (weight gradients, bias gradient)
        """
        m = X.shape[0]
        
        # Compute predictions
        z = np.dot(X, w) + b
        y_hat = self.sigmoid(z)
        
        # Compute gradients
        dw = (1/m) * np.dot(X.T, (y_hat - y))
        db = (1/m) * np.sum(y_hat - y)
        
        return dw, db
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """
        Train the logistic regression model
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Self for method chaining
        """
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0.0
        self.costs = []
        
        logger.info(f"Training logistic regression with {self.epochs} epochs")
        
        for epoch in range(self.epochs):
            # Compute gradients
            dw, db = self.compute_gradients(X, self.w, y, self.b)
            
            # Update parameters
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            
            # Compute cost every 10000 epochs
            if epoch % 10000 == 0 or epoch == self.epochs - 1:
                cost = self.compute_cost(X, self.w, y, self.b)
                self.costs.append(cost)
                logger.info(f"Epoch {epoch}: Cost = {cost:.6f}")
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities
        """
        if self.w is None:
            raise ValueError("Model must be fitted before making predictions")
        
        z = np.dot(X, self.w) + self.b
        return self.sigmoid(z)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict classes
        
        Args:
            X: Input features
            
        Returns:
            Predicted classes
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= self.threshold).astype(int)
    
    def save_model(self, filepath: str) -> None:
        """
        Save model to pickle file
        
        Args:
            filepath: Path to save the model
        """
        model_params = {
            "w": self.w,
            "b": self.b,
            "mean": self.mean,
            "std": self.std,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "threshold": self.threshold
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(model_params, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> 'LogisticRegression':
        """
        Load model from pickle file
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Self for method chaining
        """
        with open(filepath, "rb") as f:
            model_params = pickle.load(f)
        
        self.w = model_params["w"]
        self.b = model_params["b"]
        self.mean = model_params["mean"]
        self.std = model_params["std"]
        self.learning_rate = model_params.get("learning_rate", 0.02)
        self.epochs = model_params.get("epochs", 100000)
        self.threshold = model_params.get("threshold", 0.4)
        
        logger.info(f"Model loaded from {filepath}")
        return self
    
    def standardize_features(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize features using stored mean and std
        
        Args:
            X: Input features
            
        Returns:
            Standardized features
        """
        if self.mean is None or self.std is None:
            raise ValueError("Model must be fitted or loaded with standardization parameters")
        
        return (X - self.mean) / self.std


def train_and_save_model(X: np.ndarray, y: np.ndarray, model_path: str) -> LogisticRegression:
    """
    Train and save a logistic regression model
    
    Args:
        X: Training features
        y: Training labels
        model_path: Path to save the model
        
    Returns:
        Trained model
    """
    # Initialize and train model
    model = LogisticRegression()
    model.fit(X, y)
    
    # Save model
    model.save_model(model_path)
    
    return model


def load_trained_model(model_path: str) -> LogisticRegression:
    """
    Load a trained logistic regression model
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded model
    """
    model = LogisticRegression()
    model.load_model(model_path)
    return model 