"""
Tabular Preprocessor for Student Dropout Prediction
"""

import pandas as pd
import numpy as np
import pickle
from typing import List, Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TabularPreprocessor:
    """
    Preprocessor for student dropout prediction data
    """
    
    def __init__(self):
        """
        Initialize the preprocessor
        """
        # Base features that should be in the raw data
        self.base_features = [
            'total_logins',
            'time_on_platform_hours',
            'forum_posts',
            '%_assignments_completed',
            'avg_quiz_score',
            'days_since_last_login'
        ]
        
        # All required features (base + derived)
        self.required_features = [
            'total_logins',
            'time_on_platform_hours',
            'forum_posts',
            '%_assignments_completed',
            'avg_quiz_score',
            'days_since_last_login',
            'time_per_login',
            'low_quiz_flag',
            'inactive_flag',
            'assignments_per_login',
            'engagement_index'
        ]
        
        self.feature_ranges = {
            'total_logins': (1, 50),
            'time_on_platform_hours': (0.1, 30.0),
            'forum_posts': (0, 20),
            '%_assignments_completed': (0.0, 100.0),
            'avg_quiz_score': (0.0, 100.0),
            'days_since_last_login': (0, 100),
            'time_per_login': (0.1, 10.0),
            'low_quiz_flag': (0, 1),
            'inactive_flag': (0, 1),
            'assignments_per_login': (0.1, 50.0),
            'engagement_index': (0.0, 50.0)
        }
        
        self.mean = None
        self.std = None
        
    def validate_base_features(self, df: pd.DataFrame) -> bool:
        """
        Validate that the dataframe contains all required base features
        
        Args:
            df: Input dataframe
            
        Returns:
            True if valid, raises ValueError if not
        """
        missing_features = set(self.base_features) - set(df.columns)
        
        if missing_features:
            raise ValueError(f"Missing required base features: {missing_features}")
        
        extra_features = set(df.columns) - set(self.base_features) - {'student_id', 'dropped_out'}
        if extra_features:
            logger.warning(f"Extra features found: {extra_features}")
        
        return True
    
    def validate_all_features(self, df: pd.DataFrame) -> bool:
        """
        Validate that the dataframe contains all required features (base + derived)
        
        Args:
            df: Input dataframe
            
        Returns:
            True if valid, raises ValueError if not
        """
        missing_features = set(self.required_features) - set(df.columns)
        
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        extra_features = set(df.columns) - set(self.required_features) - {'student_id', 'dropped_out'}
        if extra_features:
            logger.warning(f"Extra features found: {extra_features}")
        
        return True
    
    def validate_feature_ranges(self, df: pd.DataFrame) -> bool:
        """
        Validate that features are within expected ranges
        
        Args:
            df: Input dataframe
            
        Returns:
            True if valid, raises ValueError if not
        """
        for feature, (min_val, max_val) in self.feature_ranges.items():
            if feature in df.columns:
                invalid_mask = (df[feature] < min_val) | (df[feature] > max_val)
                if invalid_mask.any():
                    invalid_count = invalid_mask.sum()
                    logger.warning(f"Feature {feature} has {invalid_count} values outside range [{min_val}, {max_val}]")
        
        return True
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features from base data
        
        Args:
            df: Input dataframe with base features
            
        Returns:
            Dataframe with derived features
        """
        df_processed = df.copy()
        
        # Calculate time_per_login if not present
        if 'time_per_login' not in df_processed.columns:
            df_processed['time_per_login'] = df_processed['time_on_platform_hours'] / df_processed['total_logins']
            df_processed['time_per_login'] = df_processed['time_per_login'].fillna(0.1)
        
        # Calculate assignments_per_login if not present
        if 'assignments_per_login' not in df_processed.columns:
            # Assuming total assignments is 100, calculate completed assignments
            completed_assignments = (df_processed['%_assignments_completed'] / 100) * 100
            df_processed['assignments_per_login'] = completed_assignments / df_processed['total_logins']
            df_processed['assignments_per_login'] = df_processed['assignments_per_login'].fillna(0.1)
        
        # Calculate engagement_index if not present
        if 'engagement_index' not in df_processed.columns:
            df_processed['engagement_index'] = (
                df_processed['total_logins'] * 0.3 +
                df_processed['time_on_platform_hours'] * 0.2 +
                df_processed['forum_posts'] * 0.2 +
                df_processed['%_assignments_completed'] * 0.2 +
                df_processed['avg_quiz_score'] * 0.1
            )
        
        # Create low_quiz_flag if not present
        if 'low_quiz_flag' not in df_processed.columns:
            df_processed['low_quiz_flag'] = (df_processed['avg_quiz_score'] < 60).astype(int)
        
        # Create inactive_flag if not present
        if 'inactive_flag' not in df_processed.columns:
            df_processed['inactive_flag'] = (df_processed['days_since_last_login'] > 30).astype(int)
        
        return df_processed
    
    def fit(self, df: pd.DataFrame) -> 'TabularPreprocessor':
        """
        Fit the preprocessor on training data
        
        Args:
            df: Training dataframe with base features
            
        Returns:
            Self for method chaining
        """
        # Validate base features
        self.validate_base_features(df)
        
        # Create derived features
        df_processed = self.create_derived_features(df)
        
        # Select only required features
        df_features: pd.DataFrame = df_processed[self.required_features]
        
        # Validate ranges
        self.validate_feature_ranges(df_features)
        
        # Compute standardization parameters
        self.mean = df_features.mean()
        self.std = df_features.std()
        
        # Handle zero standard deviations
        self.std = self.std.replace(0, 1)
        
        logger.info("Preprocessor fitted successfully")
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform input data
        
        Args:
            df: Input dataframe with base features
            
        Returns:
            Transformed numpy array
        """
        if self.mean is None or self.std is None:
            raise ValueError("Preprocessor must be fitted before transforming data")
        
        # Validate base features
        self.validate_base_features(df)
        
        # Create derived features
        df_processed = self.create_derived_features(df)
        
        # Select only required features
        df_features: pd.DataFrame = df_processed[self.required_features]
        
        # Validate ranges
        self.validate_feature_ranges(df_features)
        
        # Standardize features
        df_standardized = (df_features - self.mean) / self.std
        
        return df_standardized.values
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform data
        
        Args:
            df: Input dataframe with base features
            
        Returns:
            Transformed numpy array
        """
        return self.fit(df).transform(df)
    
    def save_preprocessor(self, filepath: str) -> None:
        """
        Save preprocessor to pickle file
        
        Args:
            filepath: Path to save the preprocessor
        """
        preprocessor_params = {
            "base_features": self.base_features,
            "required_features": self.required_features,
            "feature_ranges": self.feature_ranges,
            "mean": self.mean,
            "std": self.std
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(preprocessor_params, f)
        
        logger.info(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath: str) -> 'TabularPreprocessor':
        """
        Load preprocessor from pickle file
        
        Args:
            filepath: Path to the saved preprocessor
            
        Returns:
            Self for method chaining
        """
        with open(filepath, "rb") as f:
            preprocessor_params = pickle.load(f)
        
        self.base_features = preprocessor_params["base_features"]
        self.required_features = preprocessor_params["required_features"]
        self.feature_ranges = preprocessor_params["feature_ranges"]
        self.mean = preprocessor_params["mean"]
        self.std = preprocessor_params["std"]
        
        logger.info(f"Preprocessor loaded from {filepath}")
        return self
    
    def get_base_feature_names(self) -> List[str]:
        """
        Get list of base feature names
        
        Returns:
            List of base feature names
        """
        return self.base_features.copy()
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature names (base + derived)
        
        Returns:
            List of all feature names
        """
        return self.required_features.copy()
    
    def get_feature_ranges(self) -> Dict[str, Tuple[float, float]]:
        """
        Get feature ranges for validation
        
        Returns:
            Dictionary of feature ranges
        """
        return self.feature_ranges.copy()


def create_preprocessor_from_data(data_path: str, preprocessor_path: str) -> TabularPreprocessor:
    """
    Create and save a preprocessor from training data
    
    Args:
        data_path: Path to training data
        preprocessor_path: Path to save the preprocessor
        
    Returns:
        Fitted preprocessor
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Create and fit preprocessor
    preprocessor = TabularPreprocessor()
    preprocessor.fit(df)
    
    # Save preprocessor
    preprocessor.save_preprocessor(preprocessor_path)
    
    return preprocessor


def load_preprocessor(preprocessor_path: str) -> TabularPreprocessor:
    """
    Load a saved preprocessor
    
    Args:
        preprocessor_path: Path to the saved preprocessor
        
    Returns:
        Loaded preprocessor
    """
    preprocessor = TabularPreprocessor()
    preprocessor.load_preprocessor(preprocessor_path)
    return preprocessor 