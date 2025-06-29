from setuptools import setup, find_packages

setup(
    name="student-dropout-predictor",
    version="1.0.0",
    description="A machine learning project for predicting student dropout risk",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.25.0",
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.1.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
    ],
    python_requires=">=3.8",
) 