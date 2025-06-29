#!/usr/bin/env python3
"""
Test script to verify the app components work correctly
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from src.preprocessing.tabular_preprocessor import load_preprocessor
from src.models.logistic_regression import load_trained_model

def test_model_and_preprocessor():
    """Test loading and using the model and preprocessor"""
    print("Testing model and preprocessor...")
    
    try:
        # Load model and preprocessor
        model = load_trained_model("artifacts/logistic_model.pkl")
        preprocessor = load_preprocessor("artifacts/preprocessor.pkl")
        
        print("✅ Model and preprocessor loaded successfully")
        
        # Test with sample data
        sample_data = {
            'total_logins': 10,
            'time_on_platform_hours': 8.5,
            'forum_posts': 3,
            '%_assignments_completed': 75.0,
            'avg_quiz_score': 70.0,
            'days_since_last_login': 7,
            'time_per_login': 0.85
        }
        
        # Create DataFrame
        df = pd.DataFrame([sample_data])
        
        # Transform data
        X = preprocessor.transform(df)
        
        # Make prediction
        prob = model.predict_proba(X)[0]
        pred = model.predict(X)[0]
        
        print(f"✅ Prediction successful:")
        print(f"   Probability: {prob:.4f}")
        print(f"   Prediction: {pred}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_and_preprocessor()
    if success:
        print("\n🎉 All tests passed! The app should work correctly.")
    else:
        print("\n💥 Tests failed. Please check the error messages above.") 