# 🎓 Student Dropout Risk Predictor

A modern, end-to-end machine learning project for predicting student dropout risk using a scratch logistic regression model and a stylish Streamlit app.

## 🚀 Features
- **Custom scratch logistic regression model** (no scikit-learn for modeling)
- **Tabular preprocessor** for strict feature engineering
- **Streamlit app** with modern blue/green UI
- **Single and batch (CSV) prediction**
- **Automatic derived feature computation**
- **Strict input validation** (base features only)
- **Confidence gauge and summary stats**
- **Dockerized for easy deployment**

## 📁 Project Structure
```
project-root/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Modeling.ipynb
│
├── src/
│   ├── models/
│   ├── preprocessing/
│   ├── utils/
│
├── artifacts/
│   ├── logistic_model.pkl
│   ├── preprocessor.pkl
│   ├── batch_predictions.csv
│
├── logs/
│
├── app/
│   ├── streamlit_app.py
│
├── tests/
│
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── README.md
├── .gitignore
```

## 🧑‍💻 Setup & Usage

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create preprocessor** (if not already created)
   ```bash
   python create_preprocessor.py
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run app/streamlit_app.py
   ```

4. **Use the app**
   - **Single Prediction**: Enter base features only:
     - Total Logins
     - Time on Platform (hours)
     - Forum Posts
     - % Assignments Completed
     - Average Quiz Score
     - Days Since Last Login
     - Time per Login
   
   - **Batch Prediction**: Upload CSV with exactly these base feature columns:
     ```
     total_logins,time_on_platform_hours,forum_posts,%_assignments_completed,avg_quiz_score,days_since_last_login,time_per_login
     ```
   
   - **Derived features** are computed automatically:
     - Low Quiz Flag (<60)
     - Inactive Flag (>30 days)
     - Assignments per Login
     - Engagement Index

## 🐳 Docker

1. **Build the Docker image**
   ```bash
   docker build -t student-dropout-app .
   ```
2. **Run the container**
   ```bash
   docker run -p 8501:8501 student-dropout-app
   ```

## 📝 Notes
- Only base features are required for user input
- Derived features are computed automatically in the app
- Batch CSV uploads are strictly validated for base feature columns only
- The app uses a custom color scheme (no default Streamlit grey)
- Sample batch data is provided in `sample_batch_data.csv`

---

**Author:** Your Name 