# multiple-disease-prediction
Multiple Disease Prediction System

This application uses machine learning to predict the likelihood of various diseases based on patient data.
Diseases Covered

    Diabetes
    Heart Disease
    Parkinson's Disease
    Lung Cancer
    Hypothyroidism

Setup Instructions
1. Install Required Packages

The application will automatically install required packages when run, but you can install them manually:

pip install streamlit streamlit-option-menu scikit-learn pandas

2. Download Disease Datasets

Run the dataset downloader script to get the necessary datasets:

python download_datasets.py

This script will:

    Download the lung cancer dataset
    Download the hypothyroid dataset
    Process both datasets to match the format expected by the application
    Save them as lung_cancer.csv and hypothyroid.csv

3. Run the Application

streamlit run "multiple disease pred.py"

Using the Application

    Select a disease from the sidebar menu
    Enter the required patient data in the input fields
    Click the "Test Result" button to get a prediction

Dataset Information
Diabetes Dataset

    Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
    Target: Outcome (1 for diabetes, 0 for no diabetes)

Heart Disease Dataset

    Features: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
    Target: Last column (1 for heart disease, 0 for no heart disease)

Parkinson's Dataset

    Features: Various voice measurements (MDVP:Fo(Hz), MDVP:Fhi(Hz), etc.)
    Target: status (1 for Parkinson's, 0 for no Parkinson's)

Lung Cancer Dataset

    Features: Age, Gender, Smoking, Yellow_Fingers, Anxiety, etc.
    Target: Cancer (1 for cancer, 0 for no cancer)

Hypothyroid Dataset

    Features: age, gender, TSH, T3, T4, T4U, FTI, on_thyroxine, etc.
    Target: Hypothyroid (1 for hypothyroid, 0 for no hypothyroid)

Customizing the Datasets

If you have your own datasets, you can replace the downloaded ones with your own, as long as they follow the same format:

    CSV files with feature columns and a target column
    Target column named 'Outcome' for diabetes, last column for heart disease, 'status' for Parkinson's, 'Cancer' for lung cancer, and 'Hypothyroid' for hypothyroid# Multiple Disease Prediction System

This website is designed to predict up to three diseases (Diabetes, Heart Disease, and Parkinson's Disease). The required machine learning algorithms are designed and trained to improve the accuracy of the models. All three models are then turned into a single webapp easily using Streamlit.
How to Run
Make sure you have Python installed on your system
Install Streamlit:

pip install streamlit

Run the application:

streamlit run "multiple disease pred.py"

Features

    The application will automatically install any missing dependencies
    Uses machine learning models to predict disease likelihood
    Simple and intuitive user interface

Troubleshooting

If you encounter the error "Import 'streamlit_option_menu' could not be resolved", the application will now automatically install this dependency when you run it.
Technical Details

    Built with Streamlit
    Uses scikit-learn for machine learning models
    Implements RandomForest classifiers for predictions
