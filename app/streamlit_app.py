import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import sys

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.tabular_preprocessor import TabularPreprocessor, load_preprocessor
from src.models.logistic_regression import LogisticRegression, load_trained_model

# --- Custom Styling ---
CUSTOM_CSS = """
<style>
/* Entire App Background */
.stApp {
    background: linear-gradient(135deg, #cce3ff 0%, #99c2ff 100%);
    color: #0a1d4c;
    font-family: 'Segoe UI', 'Roboto', sans-serif;
}

/* Main Container (center) */
[data-testid="stAppViewContainer"] {
    background-color: rgba(255, 255, 255, 0.85);
    border-radius: 10px;
    padding: 20px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #003366 0%, #336699 100%);
    color: #ffffff;
}

/* Sidebar text */
[data-testid="stSidebar"] * {
    color: #ffffff;
}

/* Titles */
h1, h2, h3, h4, h5, h6 {
    color: #0a1d4c;
    font-family: 'Segoe UI', 'Roboto', sans-serif;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #003366 0%, #336699 100%);
    color: #ffffff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5em 2em;
}

/* File uploader */
.stFileUploader {
    background: #e6f0ff;
    border-radius: 8px;
}

/* DataFrame/table background */
.stDataFrame, .stTable {
    background: #e6f0ff;
    border-radius: 8px;
}

/* Progress bar (confidence gauge) */
.stProgress > div > div {
    background: linear-gradient(90deg, #003366 0%, #336699 100%);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- Constants ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "artifacts", "logistic_model.pkl")
PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, "artifacts", "preprocessor.pkl")

BASE_FEATURES = [
    'total_logins',
    'time_on_platform_hours',
    'forum_posts',
    '%_assignments_completed',
    'avg_quiz_score',
    'days_since_last_login',
    'time_per_login'
]

ALL_REQUIRED_FEATURES = [
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

# --- Load Model and Preprocessor ---
@st.cache_resource(show_spinner=False)
def get_model_and_preprocessor():
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found at: {MODEL_PATH}")
            return None, None
        if not os.path.exists(PREPROCESSOR_PATH):
            st.error(f"Preprocessor file not found at: {PREPROCESSOR_PATH}")
            return None, None
        
        model = load_trained_model(MODEL_PATH)
        preprocessor = load_preprocessor(PREPROCESSOR_PATH)
        return model, preprocessor
        
    except Exception as e:
        st.error(f"Error loading model or preprocessor: {str(e)}")
        return None, None

model, preprocessor = get_model_and_preprocessor()

if model is None or preprocessor is None:
    st.stop()

# --- Feature Computation ---
def compute_derived_features(base_features: dict) -> dict:
    features = base_features.copy()
    features['low_quiz_flag'] = 1 if features['avg_quiz_score'] < 60 else 0
    features['inactive_flag'] = 1 if features['days_since_last_login'] > 30 else 0
    completed_assignments = (features['%_assignments_completed'] / 100) * 100
    features['assignments_per_login'] = (
        completed_assignments / features['total_logins']
        if features['total_logins'] > 0 else 0.1
    )
    features['engagement_index'] = (
        features['total_logins'] * 0.3 +
        features['time_on_platform_hours'] * 0.2 +
        features['forum_posts'] * 0.2 +
        features['%_assignments_completed'] * 0.2 +
        features['avg_quiz_score'] * 0.1
    )
    return features

def compute_derived_features_batch(df: pd.DataFrame) -> pd.DataFrame:
    df_processed = df.copy()
    df_processed['low_quiz_flag'] = (df_processed['avg_quiz_score'] < 60).astype(int)
    df_processed['inactive_flag'] = (df_processed['days_since_last_login'] > 30).astype(int)
    completed_assignments = (df_processed['%_assignments_completed'] / 100) * 100
    df_processed['assignments_per_login'] = completed_assignments / df_processed['total_logins']
    df_processed['assignments_per_login'] = df_processed['assignments_per_login'].fillna(0.1)
    df_processed['engagement_index'] = (
        df_processed['total_logins'] * 0.3 +
        df_processed['time_on_platform_hours'] * 0.2 +
        df_processed['forum_posts'] * 0.2 +
        df_processed['%_assignments_completed'] * 0.2 +
        df_processed['avg_quiz_score'] * 0.1
    )
    return df_processed

# --- App Title ---
st.markdown("""
# 🎓 Student Dropout Risk Predictor
""", unsafe_allow_html=True)
st.markdown("""
#### Predict the risk of student dropout using key engagement features. Enter base features below or upload a batch CSV for instant predictions.
""")

# --- Sidebar ---
st.sidebar.markdown("""
## About
This app predicts the risk of student dropout using a custom logistic regression model trained on engagement and performance data.

**Base Features (User Input):**
- Total Logins
- Time on Platform
- Forum Posts
- % Assignments Completed
- Average Quiz Score
- Days Since Last Login
- Time per Login

**Derived Features (Auto-computed):**
- Low Quiz Flag
- Inactive Flag
- Assignments per Login
- Engagement Index

- **Modern UI**
- **Batch CSV upload**
- **Strict feature validation**
- **Confidence gauge**
- **Blue theme**

**Author:** Rishikesh Bhatt
""")

# --- Input Widgets ---
def user_input_form():
    st.markdown("## 🔢 Enter Student Base Features")
    with st.form("student_form"):
        c1, c2 = st.columns(2)
        with c1:
            total_logins = st.number_input("Total Logins", min_value=1, max_value=50, value=10)
            time_on_platform_hours = st.slider("Time on Platform (hours)", 0.1, 30.0, 8.0)
            forum_posts = st.number_input("Forum Posts", min_value=0, max_value=20, value=3)
            percent_assignments_completed = st.slider("% Assignments Completed", 0.0, 100.0, 75.0)
        with c2:
            avg_quiz_score = st.slider("Average Quiz Score", 0.0, 100.0, 70.0)
            days_since_last_login = st.number_input("Days Since Last Login", min_value=0, max_value=100, value=7)
            time_per_login = st.slider("Time per Login (hours)", 0.1, 10.0, 1.0)
        
        submitted = st.form_submit_button("Predict Dropout Risk")
    
    base_features = {
        'total_logins': total_logins,
        'time_on_platform_hours': time_on_platform_hours,
        'forum_posts': forum_posts,
        '%_assignments_completed': percent_assignments_completed,
        'avg_quiz_score': avg_quiz_score,
        'days_since_last_login': days_since_last_login,
        'time_per_login': time_per_login
    }
    return base_features, submitted

base_features, submitted = user_input_form()

# --- Prediction Logic ---
def predict_single(base_features: dict):
    if model is None or preprocessor is None:
        st.error("Model or preprocessor not loaded.")
        return None, None, None
    
    all_features = compute_derived_features(base_features)
    df = pd.DataFrame([all_features])
    X = preprocessor.transform(df)
    
    prob = float(model.predict_proba(X)[0])
    pred = int(model.predict(X)[0])
    return pred, prob, all_features

def confidence_gauge(prob):
    st.markdown("#### Confidence Gauge")
    confidence = abs(prob - 0.5) * 2
    st.progress(int(confidence * 100))
    st.markdown(f"**Confidence:** {confidence:.2f}")

if submitted:
    pred, prob, all_features = predict_single(base_features)
    
    if pred is not None:
        st.markdown("---")
        st.markdown("## 🧮 Prediction Result")
        st.markdown(f"### Dropout Prediction: <span style='color:{'red' if pred else 'green'}; font-weight:bold'>{'Yes' if pred else 'No'}</span>", unsafe_allow_html=True)
        st.markdown(f"**Dropout Probability:** `{prob:.2%}`")
        confidence_gauge(prob)
        
        with st.expander("📊 Computed Derived Features"):
            derived_features = {
                'Low Quiz Flag': 'Yes' if all_features['low_quiz_flag'] else 'No',
                'Inactive Flag': 'Yes' if all_features['inactive_flag'] else 'No',
                'Assignments per Login': f"{all_features['assignments_per_login']:.2f}",
                'Engagement Index': f"{all_features['engagement_index']:.2f}"
            }
            for feature, value in derived_features.items():
                st.write(f"**{feature}:** {value}")

# --- Batch Prediction ---
st.markdown("---")
st.markdown("## 📁 Batch CSV Prediction")
st.markdown("Upload a CSV file with only these base feature columns:")
st.code(", ".join(BASE_FEATURES), language="text")
st.markdown("*Derived features will be computed automatically.*")

uploaded_file = st.file_uploader("Upload CSV for Batch Prediction", type=["csv"])

if uploaded_file:
    if model is None or preprocessor is None:
        st.error("Model or preprocessor not loaded.")
    else:
        try:
            df = pd.read_csv(uploaded_file)
            
            if set(df.columns) != set(BASE_FEATURES):
                st.error(f"❌ Uploaded CSV must have exactly these base feature columns: {BASE_FEATURES}")
                st.error(f"Found columns: {list(df.columns)}")
            else:
                df_with_derived = compute_derived_features_batch(df)
                X = preprocessor.transform(df_with_derived)
                probs = model.predict_proba(X)
                preds = model.predict(X)
                
                results = df.copy()
                results['Dropout Prediction'] = np.where(preds == 1, 'Yes', 'No')
                results['Dropout Probability'] = probs
                results['Low Quiz Flag'] = np.where(df_with_derived['low_quiz_flag'] == 1, 'Yes', 'No')
                results['Inactive Flag'] = np.where(df_with_derived['inactive_flag'] == 1, 'Yes', 'No')
                results['Assignments per Login'] = df_with_derived['assignments_per_login'].round(2)
                results['Engagement Index'] = df_with_derived['engagement_index'].round(2)
                
                st.markdown("### Batch Prediction Results")
                st.dataframe(
                    results.style.applymap(
                        lambda v: 'color: red' if v == 'Yes' else 'color: green', 
                        subset=['Dropout Prediction']
                    ).format({'Dropout Probability': '{:.2%}'}),
                    use_container_width=True
                )
                
                st.markdown("#### 📊 Batch Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", len(results))
                with col2:
                    st.metric("Predicted Dropouts", int((preds == 1).sum()))
                with col3:
                    st.metric("Average Dropout Probability", f"{probs.mean():.2%}")
                
                csv = results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name="batch_predictions.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error processing file: {e}")
