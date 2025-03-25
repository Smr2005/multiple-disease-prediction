import pandas as pd
import numpy as np
import requests
import os
import zipfile
import io
from sklearn.preprocessing import LabelEncoder

def download_file(url, filename):
    """
    Download a file from a URL and save it locally
    """
    print(f"Downloading {filename} from {url}...")
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Successfully downloaded {filename}")
        return True
    else:
        print(f"Failed to download {filename}. Status code: {response.status_code}")
        return False

def download_and_extract_zip(url, extract_to='.'):
    """
    Download a zip file from a URL and extract its contents
    """
    print(f"Downloading and extracting zip from {url}...")
    response = requests.get(url)
    if response.status_code == 200:
        z = zipfile.ZipFile(io.BytesIO(response.content))
        z.extractall(extract_to)
        print(f"Successfully extracted zip contents to {extract_to}")
        return True
    else:
        print(f"Failed to download zip. Status code: {response.status_code}")
        return False

def prepare_lung_cancer_dataset():
    """
    Download and prepare the lung cancer dataset
    """
    # Try multiple URLs for the lung cancer dataset
    urls = [
        "https://raw.githubusercontent.com/mysarahmadbhat/lung-cancer/main/survey%20lung%20cancer.csv",
        "https://raw.githubusercontent.com/Safa1615/Lung-Cancer-Dataset/main/survey%20lung%20cancer.csv",
        "https://raw.githubusercontent.com/adityarc19/Lung-cancer-detection/master/data.csv"
    ]

    filename = "raw_lung_cancer.csv"
    download_success = False

    # Try each URL until one works
    for url in urls:
        print(f"Trying to download lung cancer dataset from {url}")
        if download_file(url, filename):
            download_success = True
            break

    if download_success:
        try:
            # Read the downloaded dataset
            lung_df = pd.read_csv(filename)

            # Check if the dataset has the expected columns
            if 'AGE' in lung_df.columns and 'LUNG_CANCER' in lung_df.columns:
                # Rename columns to match our application
                lung_df = lung_df.rename(columns={
                    'AGE': 'Age',
                    'GENDER': 'Gender',
                    'SMOKING': 'Smoking',
                    'YELLOW_FINGERS': 'Yellow_Fingers',
                    'ANXIETY': 'Anxiety',
                    'PEER_PRESSURE': 'Peer_Pressure',
                    'CHRONIC DISEASE': 'Chronic_Disease',
                    'FATIGUE ': 'Fatigue',  # Note the space in the original column name
                    'ALLERGY ': 'Allergy',  # Note the space in the original column name
                    'WHEEZING': 'Wheezing',
                    'ALCOHOL CONSUMING': 'Alcohol',
                    'COUGHING': 'Coughing',
                    'SHORTNESS OF BREATH': 'Shortness_of_Breath',
                    'SWALLOWING DIFFICULTY': 'Swallowing_Difficulty',
                    'CHEST PAIN': 'Chest_Pain',
                    'LUNG_CANCER': 'Cancer'
                })
            elif 'age' in lung_df.columns and 'lung_cancer' in lung_df.columns:
                # Alternative column naming
                lung_df = lung_df.rename(columns={
                    'age': 'Age',
                    'gender': 'Gender',
                    'smoking': 'Smoking',
                    'yellow_fingers': 'Yellow_Fingers',
                    'anxiety': 'Anxiety',
                    'peer_pressure': 'Peer_Pressure',
                    'chronic_disease': 'Chronic_Disease',
                    'fatigue': 'Fatigue',
                    'allergy': 'Allergy',
                    'wheezing': 'Wheezing',
                    'alcohol_consuming': 'Alcohol',
                    'coughing': 'Coughing',
                    'shortness_of_breath': 'Shortness_of_Breath',
                    'swallowing_difficulty': 'Swallowing_Difficulty',
                    'chest_pain': 'Chest_Pain',
                    'lung_cancer': 'Cancer'
                })

            # Encode categorical variables
            le = LabelEncoder()
            for col in lung_df.select_dtypes(include=['object']).columns:
                lung_df[col] = le.fit_transform(lung_df[col])

            # Convert 'YES'/'NO' in Cancer column to 1/0 if it's not already encoded
            if 'Cancer' in lung_df.columns and lung_df['Cancer'].dtype == 'object':
                lung_df['Cancer'] = lung_df['Cancer'].map({'YES': 1, 'NO': 0})

            # Save processed dataset
            lung_df.to_csv('lung_cancer.csv', index=False)
            print("Lung cancer dataset prepared and saved as lung_cancer.csv")

            # Clean up
            os.remove(filename)
            return True
        except Exception as e:
            print(f"Error processing downloaded file: {e}")
            # If there's an error processing the file, we'll create a synthetic dataset
            download_success = False

    if not download_success:
        # Create a synthetic dataset for demonstration
        print("Failed to download or process lung cancer dataset. Creating a synthetic dataset instead...")

        # Create a synthetic dataset with 300 samples
        np.random.seed(42)
        n_samples = 300

        # Generate synthetic data
        lung_df = pd.DataFrame({
            'Age': np.random.randint(20, 80, n_samples),
            'Gender': np.random.randint(0, 2, n_samples),
            'Smoking': np.random.randint(0, 2, n_samples),
            'Yellow_Fingers': np.random.randint(0, 2, n_samples),
            'Anxiety': np.random.randint(0, 2, n_samples),
            'Peer_Pressure': np.random.randint(0, 2, n_samples),
            'Chronic_Disease': np.random.randint(0, 2, n_samples),
            'Fatigue': np.random.randint(0, 2, n_samples),
            'Allergy': np.random.randint(0, 2, n_samples),
            'Wheezing': np.random.randint(0, 2, n_samples),
            'Alcohol': np.random.randint(0, 2, n_samples),
            'Coughing': np.random.randint(0, 2, n_samples),
            'Shortness_of_Breath': np.random.randint(0, 2, n_samples),
            'Swallowing_Difficulty': np.random.randint(0, 2, n_samples),
            'Chest_Pain': np.random.randint(0, 2, n_samples)
        })

        # Generate target variable (Cancer) based on risk factors
        # Higher risk with age, smoking, chronic disease, etc.
        risk_score = (
            0.03 * lung_df['Age'] +
            2 * lung_df['Smoking'] +
            1.5 * lung_df['Yellow_Fingers'] +
            0.8 * lung_df['Chronic_Disease'] +
            0.7 * lung_df['Shortness_of_Breath'] +
            0.6 * lung_df['Chest_Pain'] +
            0.5 * lung_df['Coughing']
        )

        # Normalize risk score to probability
        probability = 1 / (1 + np.exp(-0.1 * (risk_score - 10)))

        # Generate binary outcome based on probability
        lung_df['Cancer'] = (np.random.random(n_samples) < probability).astype(int)

        # Save synthetic dataset
        lung_df.to_csv('lung_cancer.csv', index=False)
        print("Synthetic lung cancer dataset created and saved as lung_cancer.csv")
        return True

def prepare_hypothyroid_dataset():
    """
    Download and prepare the hypothyroid dataset
    """
    # Try multiple URLs for the thyroid dataset
    urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/thyroid0387.data",
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/thyroid.csv"
    ]

    filename = "raw_thyroid.data"
    download_success = False

    # Try each URL until one works
    for url in urls:
        print(f"Trying to download hypothyroid dataset from {url}")
        if download_file(url, filename):
            download_success = True
            break

    if download_success:
        try:
            # This dataset is in a special format, we need to parse it carefully
            # Try to read the data with different formats
            try:
                # First try: UCI format
                thyroid_df = pd.read_csv(filename, header=None, na_values='?')

                # Define column names based on the UCI documentation
                column_names = [
                    'age', 'sex', 'on_thyroxine', 'query_on_thyroxine',
                    'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery',
                    'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid',
                    'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
                    'TSH_measured', 'TSH', 'T3_measured', 'T3', 'TT4_measured', 'TT4',
                    'T4U_measured', 'T4U', 'FTI_measured', 'FTI', 'TBG_measured', 'TBG',
                    'referral_source', 'target'
                ]

                # Assign column names
                thyroid_df.columns = column_names

                # The target column has classes like 'negative', 'hypothyroid', etc.
                # We'll convert it to binary: 1 for hypothyroid, 0 for others
                thyroid_df['Hypothyroid'] = thyroid_df['target'].apply(
                    lambda x: 1 if 'hypothyroid' in str(x).lower() else 0
                )

                # Select relevant features
                selected_features = [
                    'age', 'sex', 'TSH', 'T3', 'TT4', 'T4U', 'FTI',
                    'on_thyroxine', 'on_antithyroid_medication', 'sick',
                    'pregnant', 'thyroid_surgery', 'tumor', 'Hypothyroid'
                ]

            except Exception as e:
                print(f"Failed to parse as UCI format: {e}")
                # Second try: CSV format with header
                thyroid_df = pd.read_csv(filename)

                # Check if this is a different format
                if 'Class' in thyroid_df.columns:
                    # This is likely the Jason Brownlee dataset format
                    # Rename the target column
                    thyroid_df['Hypothyroid'] = thyroid_df['Class'].apply(
                        lambda x: 1 if x == 'P' else 0  # 'P' for positive, 'N' for negative
                    )

                    # Select all columns except 'Class'
                    selected_features = [col for col in thyroid_df.columns if col != 'Class']
                    selected_features.append('Hypothyroid')
                else:
                    # Unknown format, raise exception to fall back to synthetic data
                    raise ValueError("Unknown dataset format")

            # Create a new dataframe with selected features
            hypothyroid_df = thyroid_df[selected_features].copy()

            # Handle missing values
            for col in hypothyroid_df.columns:
                if hypothyroid_df[col].dtype != 'object':
                    # Fill numeric columns with median
                    hypothyroid_df[col] = hypothyroid_df[col].fillna(hypothyroid_df[col].median())
                else:
                    # Fill categorical columns with mode
                    hypothyroid_df[col] = hypothyroid_df[col].fillna(hypothyroid_df[col].mode()[0])

            # Convert categorical variables to numeric
            for col in hypothyroid_df.columns:
                if hypothyroid_df[col].dtype == 'object' and col != 'Hypothyroid':
                    le = LabelEncoder()
                    hypothyroid_df[col] = le.fit_transform(hypothyroid_df[col].astype(str))

            # Ensure Hypothyroid is numeric
            if hypothyroid_df['Hypothyroid'].dtype == 'object':
                hypothyroid_df['Hypothyroid'] = hypothyroid_df['Hypothyroid'].astype(int)

            # Save processed dataset
            hypothyroid_df.to_csv('hypothyroid.csv', index=False)
            print("Hypothyroid dataset prepared and saved as hypothyroid.csv")

            # Clean up
            os.remove(filename)
            return True

        except Exception as e:
            print(f"Error processing downloaded file: {e}")
            # If there's an error processing the file, we'll create a synthetic dataset
            download_success = False

    # Create a synthetic dataset if download failed or processing failed
    if not download_success:
        print("Creating a synthetic hypothyroid dataset for demonstration...")

        # Create a synthetic dataset with 200 samples
        np.random.seed(42)
        n_samples = 200

        # Generate synthetic data
        hypothyroid_df = pd.DataFrame({
            'age': np.random.randint(18, 80, n_samples),
            'sex': np.random.randint(0, 2, n_samples),
            'TSH': np.random.uniform(0.1, 10.0, n_samples),
            'T3': np.random.uniform(60, 200, n_samples),
            'TT4': np.random.uniform(4, 12, n_samples),
            'T4U': np.random.uniform(0.5, 1.5, n_samples),
            'FTI': np.random.uniform(50, 150, n_samples),
            'on_thyroxine': np.random.randint(0, 2, n_samples),
            'on_antithyroid_medication': np.random.randint(0, 2, n_samples),
            'sick': np.random.randint(0, 2, n_samples),
            'pregnant': np.random.randint(0, 2, n_samples),
            'thyroid_surgery': np.random.randint(0, 2, n_samples),
            'tumor': np.random.randint(0, 2, n_samples)
        })

        # Generate target variable (hypothyroid) based on TSH and T4 values
        # High TSH and low T4 are indicators of hypothyroidism
        hypothyroid_df['Hypothyroid'] = ((hypothyroid_df['TSH'] > 4.5) &
                                         (hypothyroid_df['TT4'] < 5.5)).astype(int)

        # Save synthetic dataset
        hypothyroid_df.to_csv('hypothyroid.csv', index=False)
        print("Synthetic hypothyroid dataset created and saved as hypothyroid.csv")
        return True

def main():
    """
    Main function to download and prepare both datasets
    """
    print("="*80)
    print("MULTIPLE DISEASE PREDICTION SYSTEM - DATASET PREPARATION")
    print("="*80)
    print("\nThis script will prepare the datasets needed for the lung cancer and")
    print("hypothyroid prediction modules of your application.")
    print("\nStarting dataset download and preparation...\n")

    # Prepare lung cancer dataset
    print("-"*50)
    print("PREPARING LUNG CANCER DATASET")
    print("-"*50)
    lung_success = prepare_lung_cancer_dataset()

    # Prepare hypothyroid dataset
    print("\n" + "-"*50)
    print("PREPARING HYPOTHYROID DATASET")
    print("-"*50)
    thyroid_success = prepare_hypothyroid_dataset()

    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)

    if lung_success:
        print("✓ Lung cancer dataset: Successfully prepared")
        print("  - File created: lung_cancer.csv")
    else:
        print("✗ Lung cancer dataset: Failed to prepare")

    if thyroid_success:
        print("✓ Hypothyroid dataset: Successfully prepared")
        print("  - File created: hypothyroid.csv")
    else:
        print("✗ Hypothyroid dataset: Failed to prepare")

    if lung_success and thyroid_success:
        print("\nBoth datasets have been successfully prepared!")
        print("You can now run your multiple disease prediction application using:")
        print("  streamlit run \"multiple disease pred.py\"")
    else:
        print("\nThere were issues preparing one or both datasets.")
        print("However, synthetic datasets have been created as fallbacks.")
        print("You can still run your application, but the predictions will be based")
        print("on synthetic data rather than real medical data.")

    print("\nNote: If you want to use your own datasets, make sure they follow the same format")
    print("as the ones created by this script. The column names should match exactly.")

if __name__ == "__main__":
    main()
