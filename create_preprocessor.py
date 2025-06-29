#!/usr/bin/env python3
"""
Script to create and save the preprocessor from existing data
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from src.preprocessing.tabular_preprocessor import create_preprocessor_from_data

def main():
    """Create preprocessor from existing data"""
    print("Creating preprocessor from existing data...")
    
    # Use the raw data which has all base features
    data_path = "data/raw/student_data.csv"
    preprocessor_path = "artifacts/preprocessor.pkl"
    
    print(f"Looking for data at: {os.path.abspath(data_path)}")
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
    
    print(f"✅ Data file found at {data_path}")
    
    try:
        # Load data first to check structure
        df = pd.read_csv(data_path)
        print(f"Data shape: {df.shape}")
        print(f"Data columns: {list(df.columns)}")
        
        # Create and save preprocessor
        print("Creating preprocessor...")
        preprocessor = create_preprocessor_from_data(data_path, preprocessor_path)
        print(f"✅ Preprocessor saved to {preprocessor_path}")
        
        # Print feature information
        print(f"Features: {preprocessor.get_feature_names()}")
        print(f"Feature ranges: {preprocessor.get_feature_ranges()}")
        
    except Exception as e:
        print(f"Error creating preprocessor: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 