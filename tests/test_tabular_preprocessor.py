import pandas as pd
import numpy as np
from src.preprocessing.tabular_preprocessor import TabularPreprocessor

def test_preprocessor_fit_transform():
    data = {
        'total_logins': [10, 20],
        'time_on_platform_hours': [5.0, 10.0],
        'forum_posts': [2, 4],
        '%_assignments_completed': [80.0, 90.0],
        'avg_quiz_score': [70.0, 85.0],
        'days_since_last_login': [5, 10],
        'time_per_login': [0.5, 0.5],
        'low_quiz_flag': [0, 0],
        'inactive_flag': [0, 0],
        'assignments_per_login': [8.0, 9.0],
        'engagement_index': [15.0, 20.0]
    }
    df = pd.DataFrame(data)
    preprocessor = TabularPreprocessor()
    X = preprocessor.fit_transform(df)
    assert X.shape == (2, 11), f"Expected shape (2, 11), got {X.shape}"
    assert np.allclose(X.mean(axis=0), [0]*11, atol=1e-6), "Mean should be ~0 after standardization"

def test_preprocessor_validate_features():
    data = {
        'total_logins': [10],
        'time_on_platform_hours': [5.0],
        'forum_posts': [2],
        '%_assignments_completed': [80.0],
        'avg_quiz_score': [70.0],
        'days_since_last_login': [5],
        'time_per_login': [0.5],
        'low_quiz_flag': [0],
        'inactive_flag': [0],
        'assignments_per_login': [8.0],
        'engagement_index': [15.0]
    }
    df = pd.DataFrame(data)
    preprocessor = TabularPreprocessor()
    assert preprocessor.validate_features(df) == True 