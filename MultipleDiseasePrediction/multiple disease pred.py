# -*- coding: utf-8 -*-

import pickle
import streamlit as st
import sys
import subprocess
import importlib

# Check if required packages are installed, if not install them
required_packages = {
    'streamlit_option_menu': 'streamlit-option-menu',
    'scikit-learn': 'scikit-learn',  # Changed from 'sklearn' to 'scikit-learn'
    'pandas': 'pandas'
}

for package_name in required_packages.values():
    try:
        # Try to import the package to check if it's installed
        if package_name == 'scikit-learn':
            # For scikit-learn, we need to check sklearn module
            importlib.import_module('sklearn')
            # Also explicitly check for sklearn.model_selection
            importlib.import_module('sklearn.model_selection')
        elif package_name == 'streamlit-option-menu':
            importlib.import_module('streamlit_option_menu')
        else:
            importlib.import_module(package_name)
    except ImportError:
        st.info(f"Installing required package: {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
        st.rerun()

# Now import after ensuring it's installed
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Loading the saved models
# We need to train models first since we only have CSV data files
import pandas as pd

# Import sklearn modules with proper error handling
try:
    # First try to import the base sklearn module
    import sklearn
    # Then import specific modules
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    # Verify sklearn is properly installed by using it
    # Get version safely
    try:
        sklearn_version = sklearn.__version__
        st.info(f"Using scikit-learn version: {sklearn_version}")
    except AttributeError:
        # If __version__ is not available, just confirm it's imported
        st.info("scikit-learn is successfully imported")

except ImportError as e:
    st.error(f"Error importing sklearn modules: {e}")
    st.info("Attempting to reinstall scikit-learn...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "scikit-learn"])
    # After reinstalling, try to import again
    try:
        import sklearn
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier

        # Try to get version again
        try:
            sklearn_version = sklearn.__version__
            st.info(f"Using scikit-learn version: {sklearn_version}")
        except AttributeError:
            st.info("scikit-learn is successfully imported after reinstallation")

    except ImportError as e:
        st.error(f"Still having issues with scikit-learn: {e}")
        st.error("Please try restarting the application or installing scikit-learn manually.")
        sys.exit(1)  # Exit if sklearn cannot be imported after reinstall attempt
    st.rerun()

# Load and prepare diabetes data
diabetes_data = pd.read_csv('diabetes.csv')
X_diabetes = diabetes_data.drop('Outcome', axis=1)
y_diabetes = diabetes_data['Outcome']
diabetes_model = RandomForestClassifier()
diabetes_model.fit(X_diabetes, y_diabetes)

# Load and prepare heart disease data
heart_data = pd.read_csv('heart.csv')
X_heart = heart_data.iloc[:, :-1]  # All columns except the last one
y_heart = heart_data.iloc[:, -1]   # Last column is the target
heart_disease_model = RandomForestClassifier()
heart_disease_model.fit(X_heart, y_heart)

# Load and prepare Parkinson's data
parkinsons_data = pd.read_csv('parkinsons.csv')
# Drop the name column as it's not a feature
parkinsons_data = parkinsons_data.drop('name', axis=1)
# Extract features and target
X_parkinsons = parkinsons_data.drop('status', axis=1)
y_parkinsons = parkinsons_data['status']
parkinsons_model = RandomForestClassifier()
parkinsons_model.fit(X_parkinsons, y_parkinsons)

# Check if lung cancer dataset exists and load it
try:
    # Load and prepare lung cancer data
    lung_cancer_data = pd.read_csv('lung_cancer.csv')
    # Assuming the target column is named 'Cancer' (1 for cancer, 0 for no cancer)
    # Modify this according to your actual dataset
    X_lung_cancer = lung_cancer_data.drop('Cancer', axis=1)
    y_lung_cancer = lung_cancer_data['Cancer']
    lung_cancer_model = RandomForestClassifier()
    lung_cancer_model.fit(X_lung_cancer, y_lung_cancer)
    lung_cancer_model_ready = True
except FileNotFoundError:
    st.warning("Lung cancer dataset not found. Lung cancer prediction will be disabled.")
    lung_cancer_model_ready = False

# Check if hypothyroid dataset exists and load it
try:
    # Load and prepare hypothyroid data
    hypothyroid_data = pd.read_csv('hypothyroid.csv')
    # Assuming the target column is named 'Hypothyroid' (1 for hypothyroid, 0 for normal)
    # Modify this according to your actual dataset
    X_hypothyroid = hypothyroid_data.drop('Hypothyroid', axis=1)
    y_hypothyroid = hypothyroid_data['Hypothyroid']
    hypothyroid_model = RandomForestClassifier()
    hypothyroid_model.fit(X_hypothyroid, y_hypothyroid)
    hypothyroid_model_ready = True
except FileNotFoundError:
    st.warning("Hypothyroid dataset not found. Hypothyroid prediction will be disabled.")
    hypothyroid_model_ready = False


#Sidebar for navigators
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',

                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction',
                            'Lung Cancer Prediction',
                            'Hypothyroid Prediction'],
                           icons = ['activity', 'heart', 'person', 'lungs', 'thermometer'],
                           default_index = 0)
    
# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):

    #page title
    st.title('Diabetes Prediction Using ML')

    st.write("""
    ### About Diabetes
    Diabetes is a chronic condition that affects how your body processes blood sugar (glucose).
    This tool uses several health metrics to assess the likelihood of diabetes.

    **Note:** This tool is for educational purposes only and should not replace professional medical advice.
    """)

    # Example values for diabetes
    diabetes_example = {
        'Pregnancies': '6',
        'Glucose': '148',
        'BloodPressure': '72',
        'SkinThickness': '35',
        'Insulin': '0',
        'BMI': '33.6',
        'DiabetesPedigreeFunction': '0.627',
        'Age': '50'
    }

    # Add example values button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button('Fill Example Values', key='diabetes_example'):
            for key, value in diabetes_example.items():
                st.session_state[key] = value

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
       Pregnancies = st.text_input('Number of Pregnancies',
                                  key='Pregnancies',
                                  help='Number of times pregnant (0 for males)')

    with col2:
       Glucose = st.text_input('Glucose Level (mg/dL)',
                              key='Glucose',
                              help='Plasma glucose concentration after 2 hours in an oral glucose tolerance test. Normal fasting: 70-99 mg/dL')

    with col3:
       BloodPressure = st.text_input('Blood Pressure (mm Hg)',
                                    key='BloodPressure',
                                    help='Diastolic blood pressure. Normal: <80 mm Hg')

    with col1:
       SkinThickness = st.text_input('Skin Thickness (mm)',
                                    key='SkinThickness',
                                    help='Triceps skin fold thickness. Measures fat content')

    with col2:
       Insulin = st.text_input('Insulin Level (μU/ml)',
                              key='Insulin',
                              help='2-Hour serum insulin. Normal fasting: <25 μU/ml')

    with col3:
       BMI = st.text_input('BMI value',
                          key='BMI',
                          help='Body Mass Index. Normal: 18.5-24.9')

    with col1:
       DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function',
                                               key='DiabetesPedigreeFunction',
                                               help='Scores likelihood of diabetes based on family history')

    with col2:
       Age = st.text_input('Age (years)',
                          key='Age',
                          help='Age in years')

    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction
    if st.button('Diabetes Test Result'):
        try:
            # Create a dictionary with feature names matching the training data
            input_dict = {
                'Pregnancies': Pregnancies,
                'Glucose': Glucose,
                'BloodPressure': BloodPressure,
                'SkinThickness': SkinThickness,
                'Insulin': Insulin,
                'BMI': BMI,
                'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
                'Age': Age
            }

            # Convert to DataFrame with proper feature names
            input_df = pd.DataFrame([input_dict])

            # Convert string values to float
            for column in input_df.columns:
                input_df[column] = input_df[column].astype(float)

            # Make prediction using DataFrame with feature names
            diab_prediction = diabetes_model.predict(input_df)

            if diab_prediction[0] == 1:
                diab_diagnosis = 'The person is likely to have diabetes'
                st.warning("""
                **What is Diabetes?**
                Diabetes is a metabolic disease that causes high blood sugar levels. The hormone insulin
                moves sugar from the blood into your cells to be stored or used for energy. With diabetes,
                your body either doesn't make enough insulin or can't effectively use the insulin it makes.

                **Next Steps:**
                If you received a positive prediction, please consult with a healthcare provider for proper
                testing and diagnosis. This prediction is not a medical diagnosis.
                """)
            else:
                diab_diagnosis = 'The person is not likely to have diabetes'

            st.success(diab_diagnosis)

            # Show risk factors if glucose is high but prediction is negative
            try:
                glucose_value = float(Glucose)
                bmi_value = float(BMI)
                if glucose_value > 99 and diab_prediction[0] == 0:
                    st.info("""
                    **Note:** Your glucose level appears to be elevated, which can be a risk factor for prediabetes.
                    Consider discussing these results with your healthcare provider.
                    """)
                if bmi_value > 25 and diab_prediction[0] == 0:
                    st.info("""
                    **Note:** Your BMI indicates you may be overweight, which is a risk factor for diabetes.
                    Maintaining a healthy weight through diet and exercise can help reduce your risk.
                    """)
            except:
                pass

        except ValueError:
            st.error("Please enter valid numerical values for all fields or use the 'Fill Example Values' button")
        except Exception as e:
            st.error(f"An error occurred: {e}")



# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):

    #page title
    st.title('Heart Disease Prediction Using ML')

    st.write("""
    ### About Heart Disease
    Heart disease refers to several types of heart conditions. The most common is coronary artery disease,
    which can lead to heart attack. This tool uses various health metrics to assess the likelihood of heart disease.

    **Note:** This tool is for educational purposes only and should not replace professional medical advice.
    """)

    # Example values for heart disease
    heart_example = {
        'age': '63',
        'sex': '1',  # Male
        'cp': '3',   # Chest pain type (3 = asymptomatic)
        'trestbps': '145',
        'chol': '233',
        'fbs': '1',
        'restecg': '0',
        'thalach': '150',
        'exang': '0',
        'oldpeak': '2.3',
        'slope': '0',
        'ca': '0',
        'thal': '1'
    }

    # Add example values button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button('Fill Example Values', key='heart_example'):
            for key, value in heart_example.items():
                st.session_state[key] = value

    # Create tabs for better organization
    tab1, tab2 = st.tabs(["Basic Information", "Clinical Measurements"])

    with tab1:
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.text_input('Age (years)',
                              key='age',
                              help='Age in years')

        with col2:
            sex = st.text_input('Sex (0=Female, 1=Male)',
                              key='sex',
                              help='0 = Female, 1 = Male')

        with col3:
            cp = st.text_input('Chest Pain Type (0-3)',
                             key='cp',
                             help='0 = Typical angina, 1 = Atypical angina, 2 = Non-anginal pain, 3 = Asymptomatic')

        with col1:
            trestbps = st.text_input('Resting Blood Pressure (mm Hg)',
                                    key='trestbps',
                                    help='Resting blood pressure. Normal: <120 mm Hg')

        with col2:
            chol = st.text_input('Serum Cholesterol (mg/dL)',
                               key='chol',
                               help='Serum cholesterol. Desirable: <200 mg/dL')

        with col3:
            fbs = st.text_input('Fasting Blood Sugar > 120 mg/dL (1=Yes, 0=No)',
                              key='fbs',
                              help='1 = Fasting blood sugar > 120 mg/dL, 0 = Fasting blood sugar <= 120 mg/dL')

    with tab2:
        col1, col2, col3 = st.columns(3)

        with col1:
            restecg = st.text_input('Resting ECG Results (0-2)',
                                   key='restecg',
                                   help='0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy')

        with col2:
            thalach = st.text_input('Maximum Heart Rate Achieved',
                                   key='thalach',
                                   help='Maximum heart rate achieved during exercise. Normal max: 220 - age')

        with col3:
            exang = st.text_input('Exercise Induced Angina (1=Yes, 0=No)',
                                key='exang',
                                help='1 = Yes, 0 = No')

        with col1:
            oldpeak = st.text_input('ST Depression Induced by Exercise',
                                   key='oldpeak',
                                   help='ST depression induced by exercise relative to rest')

        with col2:
            slope = st.text_input('Slope of Peak Exercise ST Segment (0-2)',
                                key='slope',
                                help='0 = Upsloping, 1 = Flat, 2 = Downsloping')

        with col3:
            ca = st.text_input('Number of Major Vessels (0-3)',
                             key='ca',
                             help='Number of major vessels colored by fluoroscopy (0-3)')

        with col1:
            thal = st.text_input('Thalassemia (0-3)',
                               key='thal',
                               help='0 = Normal, 1 = Fixed defect, 2 = Reversible defect, 3 = Unknown')

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction
    if st.button('Heart Disease Test Result'):
        try:
            # Get the column names from the original heart dataset
            feature_names = X_heart.columns.tolist()

            # Create a dictionary with feature names matching the training data
            input_values = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
            input_dict = {feature_names[i]: input_values[i] for i in range(len(feature_names))}

            # Convert to DataFrame with proper feature names
            input_df = pd.DataFrame([input_dict])

            # Convert string values to float
            for column in input_df.columns:
                input_df[column] = input_df[column].astype(float)

            # Make prediction using DataFrame with feature names
            heart_prediction = heart_disease_model.predict(input_df)

            if heart_prediction[0] == 1:
                heart_diagnosis = 'The person is likely to have heart disease'
                st.warning("""
                **What is Heart Disease?**
                Heart disease describes a range of conditions that affect your heart, including coronary artery disease,
                heart rhythm problems (arrhythmias), and heart defects you're born with (congenital heart defects).

                **Next Steps:**
                If you received a positive prediction, please consult with a healthcare provider for proper
                evaluation and diagnosis. This prediction is not a medical diagnosis.
                """)
            else:
                heart_diagnosis = 'The person is not likely to have heart disease'

            st.success(heart_diagnosis)

            # Show risk factors if certain values are concerning
            try:
                chol_value = float(chol)
                bp_value = float(trestbps)
                if chol_value > 200 and heart_prediction[0] == 0:
                    st.info("""
                    **Note:** Your cholesterol level is above the desirable range, which can be a risk factor for heart disease.
                    Consider discussing these results with your healthcare provider.
                    """)
                if bp_value > 120 and heart_prediction[0] == 0:
                    st.info("""
                    **Note:** Your blood pressure is elevated, which is a risk factor for heart disease.
                    Maintaining healthy blood pressure through diet, exercise, and medication (if prescribed) can help reduce your risk.
                    """)
            except:
                pass

        except ValueError:
            st.error("Please enter valid numerical values for all fields or use the 'Fill Example Values' button")
        except Exception as e:
            st.error(f"An error occurred: {e}")



# Parkinson's Prediction Page
if (selected == 'Parkinsons Prediction'):

    # page title
    st.title('Parkinsons Prediction Using ML')

    st.write("""
    ### About Parkinson's Disease Voice Analysis
    This tool analyzes voice recordings to detect patterns associated with Parkinson's disease.
    The measurements below come from sustained phonations (saying 'ahhh'), where various
    properties of the voice are analyzed.

    **Note:** If you don't have these measurements, you can use the example values provided
    or consult with a healthcare professional for a proper assessment.
    """)

    # Create tabs for organized input
    tab1, tab2, tab3 = st.tabs(["Frequency Measures", "Amplitude Measures", "Nonlinear Measures"])

    with tab1:
        st.subheader("Frequency Measurements")
        st.write("These measure the fundamental frequency of the voice and its variations.")

        col1, col2 = st.columns(2)

        with col1:
            fo = st.text_input('Average vocal frequency (Hz)',
                              help='Normal range: 107-177 Hz for men, 164-258 Hz for women')

            fhi = st.text_input('Maximum vocal frequency (Hz)',
                               help='Usually higher than the average frequency')

            flo = st.text_input('Minimum vocal frequency (Hz)',
                               help='Usually lower than the average frequency')

        with col2:
            Jitter_percent = st.text_input('Jitter in percentage (%)',
                                         help='Frequency variation - normal range: <1.04%')

            Jitter_Abs = st.text_input('Absolute jitter in microseconds',
                                     help='Frequency variation - normal range: <83.2 μs')

            RAP = st.text_input('Relative Amplitude Perturbation',
                              help='Frequency variation - normal range: <0.680%')

            PPQ = st.text_input('Five-point Period Perturbation',
                              help='Frequency variation - normal range: <0.840%')

            DDP = st.text_input('Average difference of differences',
                              help='Frequency variation - normal range: <2.040%')

    with tab2:
        st.subheader("Amplitude Measurements")
        st.write("These measure the amplitude variations in the voice.")

        col1, col2 = st.columns(2)

        with col1:
            Shimmer = st.text_input('Shimmer in percentage (%)',
                                  help='Amplitude variation - normal range: <3.810%')

            Shimmer_dB = st.text_input('Shimmer in decibels (dB)',
                                     help='Amplitude variation - normal range: <0.350 dB')

            APQ3 = st.text_input('Three-point Amplitude Perturbation',
                               help='Amplitude variation - normal range: <3.070%')

        with col2:
            APQ5 = st.text_input('Five-point Amplitude Perturbation',
                               help='Amplitude variation - normal range: <3.420%')

            APQ = st.text_input('11-point Amplitude Perturbation',
                              help='Amplitude variation - normal range: <3.960%')

            DDA = st.text_input('Average absolute differences',
                              help='Amplitude variation - normal range: <3.590%')

    with tab3:
        st.subheader("Nonlinear Measures & Noise Ratios")
        st.write("These measure noise, nonlinear dynamics, and signal complexity.")

        col1, col2 = st.columns(2)

        with col1:
            NHR = st.text_input('Noise-to-Harmonics Ratio',
                              help='Ratio of noise to tonal components - normal range: <0.190')

            HNR = st.text_input('Harmonics-to-Noise Ratio (dB)',
                              help='Ratio of harmonics to noise - higher values are better')

            RPDE = st.text_input('Recurrence Period Density Entropy',
                               help='Measures voice irregularity - range: 0 to 1')

            DFA = st.text_input('Detrended Fluctuation Analysis',
                              help='Signal fractal scaling exponent - range: 0.5 to 1')

        with col2:
            spread1 = st.text_input('Frequency variation measure 1',
                                  help='Nonlinear measure of frequency variation')

            spread2 = st.text_input('Frequency variation measure 2',
                                  help='Nonlinear measure of frequency variation')

            D2 = st.text_input('Correlation dimension',
                             help='Measures complexity of the voice signal')

            PPE = st.text_input('Pitch Period Entropy',
                              help='Measures impaired control of stable pitch - range: 0 to 1')

    # Example values for Parkinson's
    parkinsons_example = {
        'fo': '119.992', 'fhi': '157.302', 'flo': '74.997',
        'Jitter_percent': '0.00662', 'Jitter_Abs': '0.00004',
        'RAP': '0.00401', 'PPQ': '0.00506', 'DDP': '0.01204',
        'Shimmer': '0.04374', 'Shimmer_dB': '0.426',
        'APQ3': '0.02182', 'APQ5': '0.02971', 'APQ': '0.02971',
        'DDA': '0.06545', 'NHR': '0.02211', 'HNR': '21.033',
        'RPDE': '0.414783', 'DFA': '0.815285',
        'spread1': '-4.813031', 'spread2': '0.266482',
        'D2': '2.301442', 'PPE': '0.284654'
    }

    # Add example values button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button('Fill Example Values', key='parkinsons_example'):
            for key, value in parkinsons_example.items():
                st.session_state[key] = value

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction
    if st.button("Parkinson's Test Result"):
        try:
            # Get the column names from the original parkinsons dataset
            feature_names = X_parkinsons.columns.tolist()

            # Create a list of input values
            input_values = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                          RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5,
                          APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

            # Create a dictionary with feature names matching the training data
            input_dict = {feature_names[i]: input_values[i] for i in range(len(feature_names))}

            # Convert to DataFrame with proper feature names
            input_df = pd.DataFrame([input_dict])

            # Convert string values to float
            for column in input_df.columns:
                input_df[column] = input_df[column].astype(float)

            # Make prediction using DataFrame with feature names
            parkinsons_prediction = parkinsons_model.predict(input_df)

            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = "The person may have Parkinson's disease"
                st.warning("""
                **Note:** This prediction is based solely on voice analysis and should not be
                considered a medical diagnosis. Please consult with a neurologist for proper
                evaluation and diagnosis.
                """)
            else:
                parkinsons_diagnosis = "The person does not show voice patterns associated with Parkinson's disease"

            st.success(parkinsons_diagnosis)

        except ValueError:
            st.error("Please enter valid numerical values for all fields or use the 'Fill with Example Values' button")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Lung Cancer Prediction Page
if (selected == 'Lung Cancer Prediction'):

    #page title
    st.title('Lung Cancer Prediction Using ML')

    if not lung_cancer_model_ready:
        st.error("Lung cancer prediction is currently unavailable. Please add a lung_cancer.csv file to enable this feature.")
        st.info("The lung cancer dataset should contain relevant features such as age, smoking history, exposure to pollutants, family history, etc., with a target column named 'Cancer'.")
    else:
        st.write("""
        ### About Lung Cancer
        Lung cancer is a type of cancer that begins in the lungs and is the leading cause of cancer deaths worldwide.
        This tool uses various risk factors and symptoms to assess the likelihood of lung cancer.

        **Note:** This tool is for educational purposes only and should not replace professional medical advice.
        """)

        # Example values for lung cancer
        lung_cancer_example = {
            'lung_age': '65',
            'gender': '1',  # Male
            'smoking': '1',  # Yes
            'yellow_fingers': '1',  # Yes
            'anxiety': '1',  # Yes
            'peer_pressure': '0',  # No
            'chronic_disease': '1',  # Yes
            'fatigue': '1',  # Yes
            'allergy': '0',  # No
            'wheezing': '1',  # Yes
            'alcohol': '0',  # No
            'coughing': '1',  # Yes
            'shortness_of_breath': '1',  # Yes
            'swallowing_difficulty': '1',  # Yes
            'chest_pain': '1'   # Yes
        }

        # Add example values button
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button('Fill Example Values', key='lung_cancer_example'):
                for key, value in lung_cancer_example.items():
                    st.session_state[key] = value

        # Create tabs for better organization
        tab1, tab2 = st.tabs(["Demographics & Habits", "Symptoms"])

        with tab1:
            col1, col2, col3 = st.columns(3)

            with col1:
                age = st.text_input('Age (years)',
                                  key='lung_age',
                                  help='Age in years')

            with col2:
                gender = st.text_input('Gender (0=Female, 1=Male)',
                                     key='gender',
                                     help='0 = Female, 1 = Male')

            with col3:
                smoking = st.text_input('Smoking (0=No, 1=Yes)',
                                      key='smoking',
                                      help='Do you smoke or have a history of smoking?')

            with col1:
                yellow_fingers = st.text_input('Yellow Fingers (0=No, 1=Yes)',
                                             key='yellow_fingers',
                                             help='Discoloration of fingers due to smoking')

            with col2:
                alcohol = st.text_input('Alcohol Consumption (0=No, 1=Yes)',
                                      key='alcohol',
                                      help='Regular alcohol consumption')

            with col3:
                peer_pressure = st.text_input('Peer Pressure (0=No, 1=Yes)',
                                            key='peer_pressure',
                                            help='Influenced by peers to smoke or drink')

        with tab2:
            col1, col2, col3 = st.columns(3)

            with col1:
                chronic_disease = st.text_input('Chronic Disease (0=No, 1=Yes)',
                                              key='chronic_disease',
                                              help='Presence of any chronic disease')

            with col2:
                fatigue = st.text_input('Fatigue (0=No, 1=Yes)',
                                      key='fatigue',
                                      help='Persistent fatigue or weakness')

            with col3:
                allergy = st.text_input('Allergy (0=No, 1=Yes)',
                                      key='allergy',
                                      help='Presence of allergies')

            with col1:
                wheezing = st.text_input('Wheezing (0=No, 1=Yes)',
                                       key='wheezing',
                                       help='Noisy breathing or whistling sound when breathing')

            with col2:
                coughing = st.text_input('Coughing (0=No, 1=Yes)',
                                       key='coughing',
                                       help='Persistent coughing')

            with col3:
                shortness_of_breath = st.text_input('Shortness of Breath (0=No, 1=Yes)',
                                                  key='shortness_of_breath',
                                                  help='Difficulty breathing or shortness of breath')

            with col1:
                swallowing_difficulty = st.text_input('Swallowing Difficulty (0=No, 1=Yes)',
                                                    key='swallowing_difficulty',
                                                    help='Difficulty swallowing food or liquids')

            with col2:
                chest_pain = st.text_input('Chest Pain (0=No, 1=Yes)',
                                         key='chest_pain',
                                         help='Pain in the chest area')

            with col3:
                anxiety = st.text_input('Anxiety (0=No, 1=Yes)',
                                      key='anxiety',
                                      help='Feelings of anxiety or worry')

        # code for Prediction
        lung_cancer_diagnosis = ''

        # creating a button for Prediction
        if st.button('Lung Cancer Test Result'):
            try:
                # Get the column names from the original lung cancer dataset
                feature_names = X_lung_cancer.columns.tolist()

                # Create a list of input values
                input_values = [age, gender, smoking, yellow_fingers, anxiety, peer_pressure,
                              chronic_disease, fatigue, allergy, wheezing, alcohol, coughing,
                              shortness_of_breath, swallowing_difficulty, chest_pain]

                # Ensure the number of input values matches the number of features
                if len(input_values) != len(feature_names):
                    st.error(f"Error: Number of input values ({len(input_values)}) does not match number of features ({len(feature_names)})")
                else:
                    # Create a dictionary with feature names matching the training data
                    input_dict = {feature_names[i]: input_values[i] for i in range(len(feature_names))}

                    # Convert to DataFrame with proper feature names
                    input_df = pd.DataFrame([input_dict])

                    # Convert string values to float
                    for column in input_df.columns:
                        input_df[column] = input_df[column].astype(float)

                    # Make prediction using DataFrame with feature names
                    lung_cancer_prediction = lung_cancer_model.predict(input_df)

                    if lung_cancer_prediction[0] == 1:
                        lung_cancer_diagnosis = "The person may have lung cancer"
                        st.warning("""
                        **What is Lung Cancer?**
                        Lung cancer is a type of cancer that begins in the lungs and is often related to smoking,
                        though it can occur in non-smokers as well. It is characterized by uncontrolled cell growth
                        in tissues of the lung, which can spread to other parts of the body.

                        **Next Steps:**
                        If you received a positive prediction, please consult with a healthcare provider immediately
                        for proper evaluation and diagnosis. Early detection is crucial for effective treatment.
                        This prediction is not a medical diagnosis.
                        """)
                    else:
                        lung_cancer_diagnosis = "The person does not show indicators of lung cancer"

                    st.success(lung_cancer_diagnosis)

                    # Show risk factors if smoking is present but prediction is negative
                    try:
                        smoking_value = float(smoking)
                        if smoking_value == 1 and lung_cancer_prediction[0] == 0:
                            st.info("""
                            **Note:** While your current assessment doesn't indicate lung cancer, smoking is the
                            leading risk factor for developing lung cancer. Quitting smoking can significantly
                            reduce your risk over time.
                            """)
                    except:
                        pass

            except ValueError:
                st.error("Please enter valid numerical values for all fields or use the 'Fill Example Values' button")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Hypothyroid Prediction Page
if (selected == 'Hypothyroid Prediction'):

    #page title
    st.title('Hypothyroid Prediction Using ML')

    if not hypothyroid_model_ready:
        st.error("Hypothyroid prediction is currently unavailable. Please add a hypothyroid.csv file to enable this feature.")
        st.info("The hypothyroid dataset should contain relevant features such as TSH levels, T3, T4, age, gender, etc., with a target column named 'Hypothyroid'.")
    else:
        st.write("""
        ### About Hypothyroidism
        Hypothyroidism is a condition where the thyroid gland doesn't produce enough thyroid hormones.
        This prediction tool uses laboratory test results and patient information to assess the likelihood
        of hypothyroidism.

        **Note:** This tool is for educational purposes only and should not replace professional medical advice.
        """)

        # Create tabs for better organization
        tab1, tab2 = st.tabs(["Thyroid Tests", "Patient Information"])

        with tab1:
            st.subheader("Thyroid Function Tests")
            st.write("These are blood tests that measure thyroid hormone levels.")

            col1, col2 = st.columns(2)

            with col1:
                tsh = st.text_input('TSH Level (mIU/L)',
                                  help='Normal range: 0.4-4.0 mIU/L. Higher values may indicate hypothyroidism.')

                t3 = st.text_input('T3 Level (ng/dL)',
                                 help='Normal range: 80-200 ng/dL. Lower values may indicate hypothyroidism.')

            with col2:
                t4 = st.text_input('T4 Level (μg/dL)',
                                 help='Normal range: 5.0-12.0 μg/dL. Lower values may indicate hypothyroidism.')

                t4u = st.text_input('T4U Level',
                                  help='T4 Uptake Ratio. Normal range: 0.8-1.2.')

                fti = st.text_input('FTI Value',
                                  help='Free Thyroxine Index. Normal range: 6.0-10.5.')

        with tab2:
            st.subheader("Patient Information")
            st.write("These are details about the patient's health history and demographics.")

            col1, col2 = st.columns(2)

            with col1:
                age = st.text_input('Age', key='hypo_age',
                                  help='Patient age in years')

                gender = st.text_input('Gender (0 for Female, 1 for Male)', key='hypo_gender',
                                     help='0 = Female, 1 = Male')

                on_thyroxine = st.text_input('On Thyroxine Medication (0 for No, 1 for Yes)',
                                           help='Is the patient currently taking thyroxine medication?')

                on_antithyroid_meds = st.text_input('On Antithyroid Medication (0 for No, 1 for Yes)',
                                                  help='Is the patient currently taking anti-thyroid medication?')

            with col2:
                sick = st.text_input('Currently Sick (0 for No, 1 for Yes)',
                                   help='Is the patient currently sick with another illness?')

                pregnant = st.text_input('Pregnant (0 for No, 1 for Yes)',
                                       help='Is the patient pregnant?')

                thyroid_surgery = st.text_input('Previous Thyroid Surgery (0 for No, 1 for Yes)',
                                              help='Has the patient had thyroid surgery in the past?')

                tumor = st.text_input('Thyroid Tumor (0 for No, 1 for Yes)',
                                    help='Does the patient have a thyroid tumor?')

        # Example values for hypothyroidism
        hypothyroid_example = {
            'hypo_age': '55',
            'hypo_gender': '0',  # Female
            'tsh': '7.8',   # Elevated (normal is 0.4-4.0)
            't3': '70',     # Low (normal is 80-200)
            't4': '4.2',    # Low (normal is 5.0-12.0)
            't4u': '0.9',
            'fti': '5.1',   # Low (normal is 6.0-10.5)
            'on_thyroxine': '0',
            'on_antithyroid_meds': '0',
            'sick': '0',
            'pregnant': '0',
            'thyroid_surgery': '0',
            'tumor': '0'
        }

        # Add example values button
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button('Fill Example Values', key='hypothyroid_example'):
                for key, value in hypothyroid_example.items():
                    st.session_state[key] = value

        # code for Prediction
        hypothyroid_diagnosis = ''

        # creating a button for Prediction
        if st.button('Hypothyroid Test Result'):
            try:
                # Get the column names from the original hypothyroid dataset
                feature_names = X_hypothyroid.columns.tolist()

                # Create a list of input values
                input_values = [age, gender, tsh, t3, t4, t4u, fti, on_thyroxine,
                              on_antithyroid_meds, sick, pregnant, thyroid_surgery, tumor]

                # Ensure the number of input values matches the number of features
                if len(input_values) != len(feature_names):
                    st.error(f"Error: Number of input values ({len(input_values)}) does not match number of features ({len(feature_names)})")
                else:
                    # Create a dictionary with feature names matching the training data
                    input_dict = {feature_names[i]: input_values[i] for i in range(len(feature_names))}

                    # Convert to DataFrame with proper feature names
                    input_df = pd.DataFrame([input_dict])

                    # Convert string values to float
                    for column in input_df.columns:
                        input_df[column] = input_df[column].astype(float)

                    # Make prediction using DataFrame with feature names
                    hypothyroid_prediction = hypothyroid_model.predict(input_df)

                    if hypothyroid_prediction[0] == 1:
                        hypothyroid_diagnosis = "The person may have hypothyroidism"

                        # Display additional information about the condition
                        st.warning("""
                        **What is Hypothyroidism?**
                        Hypothyroidism is a condition where the thyroid gland doesn't produce enough thyroid hormones.
                        Common symptoms include fatigue, weight gain, cold intolerance, dry skin, and depression.

                        **Next Steps:**
                        If you're experiencing symptoms of hypothyroidism, please consult with a healthcare provider
                        for proper diagnosis and treatment. This prediction is not a medical diagnosis.
                        """)
                    else:
                        hypothyroid_diagnosis = "The person does not show indicators of hypothyroidism"

                    st.success(hypothyroid_diagnosis)

                    # If TSH is high but prediction is negative, add a note
                    try:
                        tsh_value = float(tsh)
                        if tsh_value > 4.0 and hypothyroid_prediction[0] == 0:
                            st.info("""
                            **Note:** Your TSH level appears to be elevated, which can sometimes indicate subclinical
                            hypothyroidism. Consider discussing these results with your healthcare provider.
                            """)
                    except:
                        pass

            except ValueError:
                st.error("Please enter valid numerical values for all fields or use the 'Fill with Example Values' button")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    
# -*- coding: utf-8 -*-

import pickle
import streamlit as st
import sys
import subprocess
import importlib

# Check if required packages are installed, if not install them
required_packages = {
    'streamlit_option_menu': 'streamlit-option-menu',
    'scikit-learn': 'scikit-learn',  
    'pandas': 'pandas'
}

for package_name in required_packages.values():
    try:
        if package_name == 'scikit-learn':
            importlib.import_module('sklearn')
            importlib.import_module('sklearn.model_selection')
        elif package_name == 'streamlit-option-menu':
            importlib.import_module('streamlit_option_menu')
        else:
            importlib.import_module(package_name)
    except ImportError:
        st.info(f"Installing required package: {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
        st.rerun()

# Now import after ensuring it's installed
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="")

# Loading the saved models
import pandas as pd

# Import sklearn modules with proper error handling
try:
    import sklearn
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier()

    try:
        sklearn_version = sklearn.__version__
        st.info(f"Using scikit-learn version: {sklearn_version}")
    except AttributeError:
        st.info("scikit-learn is successfully imported")

except ImportError as e:
    st.error(f"Error importing sklearn modules: {e}")
    st.info("Attempting to reinstall scikit-learn...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "scikit-learn"])
    try:
        import sklearn
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier()

        try:
            sklearn_version = sklearn.__version__
            st.info(f"Using scikit-learn version: {sklearn_version}")
        except AttributeError:
            st.info("scikit-learn is successfully imported after reinstallation")

    except ImportError as e:
        st.error(f"Still having issues with scikit-learn: {e}")
        st.error("Please try restarting the application or installing scikit-learn manually.")
        sys.exit(1)  
    st.rerun()

# Load and prepare diabetes data
diabetes_data = pd.read_csv('diabetes.csv')
X_diabetes = diabetes_data.drop('Outcome', axis=1)
y_diabetes = diabetes_data['Outcome']
diabetes_model = RandomForestClassifier()
diabetes_model.fit(X_diabetes, y_diabetes)

# Load and prepare heart disease data
heart_data = pd.read_csv('heart.csv')
X_heart = heart_data.iloc[:, :-1]  
y_heart = heart_data.iloc[:, -1]   
heart_disease_model = RandomForestClassifier()
heart_disease_model.fit(X_heart, y_heart)

# Load and prepare Parkinson's data
parkinsons_data = pd.read_csv('parkinsons.csv')
parkinsons_data = parkinsons_data.drop('name', axis=1)
X_parkinsons = parkinsons_data.drop('status', axis=1)
y_parkinsons = parkinsons_data['status']
parkinsons_model = RandomForestClassifier()
parkinsons_model.fit(X_parkinsons, y_parkinsons)

# Check if lung cancer dataset exists and load it
try:
    lung_cancer_data = pd.read_csv('lung_cancer.csv')
    X_lung_cancer = lung_cancer_data.drop('Cancer', axis=1)
    y_lung_cancer = lung_cancer_data['Cancer']
    lung_cancer_model = RandomForestClassifier()
    lung_cancer_model.fit(X_lung_cancer, y_lung_cancer)
    lung_cancer_model_ready = True
except FileNotFoundError:
    st.warning("Lung cancer dataset not found. Lung cancer prediction will be disabled.")
    lung_cancer_model_ready = False

# Check if hypothyroid dataset exists and load it
try:
    hypothyroid_data = pd.read_csv('hypothyroid.csv')
    X_hypothyroid = hypothyroid_data.drop('Hypothyroid', axis=1)
    y_hypothyroid = hypothyroid_data['Hypothyroid']
    hypothyroid_model = RandomForestClassifier()
    hypothyroid_model.fit(X_hypothyroid, y_hypothyroid)
    hypothyroid_model_ready = True
except FileNotFoundError:
    st.warning("Hypothyroid dataset not found. Hypothyroid prediction will be disabled.")
    hypothyroid_model_ready = False

#Sidebar for navigators
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction',
                            'Lung Cancer Prediction',
                            'Hypothyroid Prediction'],
                           icons = ['activity', 'heart', 'person', 'lungs', 'thermometer'],
                           default_index = 0)
    
# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):

    st.title('Diabetes Prediction Using ML')

    st.write("""
    ### About Diabetes
    Diabetes is a chronic condition that affects how your body processes blood sugar (glucose).
    This tool uses several health metrics to assess the likelihood of diabetes.

    **Note:** This tool is for educational purposes only and should not replace professional medical advice.
    """)

    diabetes_example = {
        'Pregnancies': '6',
        'Glucose': '148',
        'BloodPressure': '72',
        'SkinThickness': '35',
        'Insulin': '0',
        'BMI': '33.6',
        'DiabetesPedigreeFunction': '0.627',
        'Age': '50'
    }

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button('Fill Example Values', key='diabetes_example'):
            for key, value in diabetes_example.items():
                st.session_state[key] = value

    col1, col2, col3 = st.columns(3)

    with col1:
       Pregnancies = st.text_input('Number of Pregnancies',
                                  key='Pregnancies',
                                  help='Number of times pregnant (0 for males)')

    with col2:
       Glucose = st.text_input('Glucose Level (mg/dL)',
                              key='Glucose',
                              help='Plasma glucose concentration after 2 hours in an oral glucose tolerance test. Normal fasting: 70-99 mg/dL')

    with col3:
       BloodPressure = st.text_input('Blood Pressure (mm Hg)',
                                    key='BloodPressure',
                                    help='Diastolic blood pressure. Normal: <80 mm Hg')

    with col1:
       SkinThickness = st.text_input('Skin Thickness (mm)',
                                    key='SkinThickness',
                                    help='Triceps skin fold thickness. Measures fat content')

    with col2:
       Insulin = st.text_input('Insulin Level (μU/ml)',
                              key='Insulin',
                              help='2-Hour serum insulin. Normal fasting: <25 μU/ml')

    with col3:
       BMI = st.text_input('BMI value',
                          key='BMI',
                          help='Body Mass Index. Normal: 18.5-24.9')

    with col1:
       DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function',
                                               key='DiabetesPedigreeFunction',
                                               help='Scores likelihood of diabetes based on family history')

    with col2:
       Age = st.text_input('Age (years)',
                          key='Age',
                          help='Age in years')

    diab_diagnosis = ''

    if st.button('Diabetes Test Result'):
        try:
            input_dict = {
                'Pregnancies': Pregnancies,
                'Glucose': Glucose,
                'BloodPressure': BloodPressure,
                'SkinThickness': SkinThickness,
                'Insulin': Insulin,
                'BMI': BMI,
                'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
                'Age': Age
            }

            input_df = pd.DataFrame([input_dict])

            for column in input_df.columns:
                input_df[column] = input_df[column].astype(float)

            diab_prediction = diabetes_model.predict(input_df)

            if diab_prediction[0] == 1:
                diab_diagnosis = 'The person is likely to have diabetes'
                st.warning("""
                **What is Diabetes?**
                Diabetes is a metabolic disease that causes high blood sugar levels. The hormone insulin
                moves sugar from the blood into your cells to be stored or used for energy. With diabetes,
                your body either doesn't make enough insulin or can't effectively use the insulin it makes.

                **Next Steps:**
                If you received a positive prediction, please consult with a healthcare provider for proper
                testing and diagnosis. This prediction is not a medical diagnosis.
                """)
            else:
                diab_diagnosis = 'The person is not likely to have diabetes'

            st.success(diab_diagnosis)

            try:
                glucose_value = float(Glucose)
                bmi_value = float(BMI)
                if glucose_value > 99 and diab_prediction[0] == 0:
                    st.info("""
                    **Note:** Your glucose level appears to be elevated, which can be a risk factor for prediabetes.
                    Consider discussing these results with your healthcare provider.
                    """)
                if bmi_value > 25 and diab_prediction[0] == 0:
                    st.info# -*- coding: utf-8 -*-

import pickle
import streamlit as st
import sys
import subprocess
import importlib

# Check if required packages are installed, if not install them
required_packages = {
    'streamlit_option_menu': 'streamlit-option-menu',
    'scikit-learn': 'scikit-learn',  
    'pandas': 'pandas'
}

for package_name in required_packages.values():
    try:
        if package_name == 'scikit-learn':
            importlib.import_module('sklearn')
            importlib.import_module('sklearn.model_selection')
        elif package_name == 'streamlit-option-menu':
            importlib.import_module('streamlit_option_menu')
        else:
            importlib.import_module(package_name)
    except ImportError:
        st.info(f"Installing required package: {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
        st.rerun()

# Now import after ensuring it's installed
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="")

# Loading the saved models
import pandas as pd

# Import sklearn modules with proper error handling
try:
    import sklearn
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier()

    try:
        sklearn_version = sklearn.__version__
        st.info(f"Using scikit-learn version: {sklearn_version}")
    except AttributeError:
        st.info("scikit-learn is successfully imported")

except ImportError as e:
    st.error(f"Error importing sklearn modules: {e}")
    st.info("Attempting to reinstall scikit-learn...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "scikit-learn"])
    try:
        import sklearn
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier()

        try:
            sklearn_version = sklearn.__version__
            st.info(f"Using scikit-learn version: {sklearn_version}")
        except AttributeError:
            st.info("scikit-learn is successfully imported after reinstallation")

    except ImportError as e:
        st.error(f"Still having issues with scikit-learn: {e}")
        st.error("Please try restarting the application or installing scikit-learn manually.")
        sys.exit(1)  
    st.rerun()

# Load and prepare diabetes data
diabetes_data = pd.read_csv('diabetes.csv')
X_diabetes = diabetes_data.drop('Outcome', axis=1)
y_diabetes = diabetes_data['Outcome']
diabetes_model = RandomForestClassifier()
diabetes_model.fit(X_diabetes, y_diabetes)

# Load and prepare heart disease data
heart_data = pd.read_csv('heart.csv')
X_heart = heart_data.iloc[:, :-1]  
y_heart = heart_data.iloc[:, -1]   
heart_disease_model = RandomForestClassifier()
heart_disease_model.fit(X_heart, y_heart)

# Load and prepare Parkinson's data
parkinsons_data = pd.read_csv('parkinsons.csv')
parkinsons_data = parkinsons_data.drop('name', axis=1)
X_parkinsons = parkinsons_data.drop('status', axis=1)
y_parkinsons = parkinsons_data['status']
parkinsons_model = RandomForestClassifier()
parkinsons_model.fit(X_parkinsons, y_parkinsons)

# Check if lung cancer dataset exists and load it
try:
    lung_cancer_data = pd.read_csv('lung_cancer.csv')
    X_lung_cancer = lung_cancer_data.drop('Cancer', axis=1)
    y_lung_cancer = lung_cancer_data['Cancer']
    lung_cancer_model = RandomForestClassifier()
    lung_cancer_model.fit(X_lung_cancer, y_lung_cancer)
    lung_cancer_model_ready = True
except FileNotFoundError:
    st.warning("Lung cancer dataset not found. Lung cancer prediction will be disabled.")
    lung_cancer_model_ready = False

# Check if hypothyroid dataset exists and load it
try:
    hypothyroid_data = pd.read_csv('hypothyroid.csv')
    X_hypothyroid = hypothyroid_data.drop('Hypothyroid', axis=1)
    y_hypothyroid = hypothyroid_data['Hypothyroid']
    hypothyroid_model = RandomForestClassifier()
    hypothyroid_model.fit(X_hypothyroid, y_hypothyroid)
    hypothyroid_model_ready = True
except FileNotFoundError:
    st.warning("Hypothyroid dataset not found. Hypothyroid prediction will be disabled.")
    hypothyroid_model_ready = False

#Sidebar for navigators
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction',
                            'Lung Cancer Prediction',
                            'Hypothyroid Prediction'],
                           icons = ['activity', 'heart', 'person', 'lungs', 'thermometer'],
                           default_index = 0)
    
# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):

    st.title('Diabetes Prediction Using ML')

    st.write("""
    ### About Diabetes
    Diabetes is a chronic condition that affects how your body processes blood sugar (glucose).
    This tool uses several health metrics to assess the likelihood of diabetes.

    **Note:** This tool is for educational purposes only and should not replace professional medical advice.
    """)

    diabetes_example = {
        'Pregnancies': '6',
        'Glucose': '148',
        'BloodPressure': '72',
        'SkinThickness': '35',
        'Insulin': '0',
        'BMI': '33.6',
        'DiabetesPedigreeFunction': '0.627',
        'Age': '50'
    }

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button('Fill Example Values', key='diabetes_example'):
            for key, value in diabetes_example.items():
                st.session_state[key] = value

    col1, col2, col3 = st.columns(3)

    with col1:
       Pregnancies = st.text_input('Number of Pregnancies',
                                  key='Pregnancies',
                                  help='Number of times pregnant (0 for males)')

    with col2:
       Glucose = st.text_input('Glucose Level (mg/dL)',
                              key='Glucose',
                              help='Plasma glucose concentration after 2 hours in an oral glucose tolerance test. Normal fasting: 70-99 mg/dL')

    with col3:
       BloodPressure = st.text_input('Blood Pressure (mm Hg)',
                                    key='BloodPressure',
                                    help='Diastolic blood pressure. Normal: <80 mm Hg')

    with col1:
       SkinThickness = st.text_input('Skin Thickness (mm)',
                                    key='SkinThickness',
                                    help='Triceps skin fold thickness. Measures fat content')

    with col2:
       Insulin = st.text_input('Insulin Level (μU/ml)',
                              key='Insulin',
                              help='2-Hour serum insulin. Normal fasting: <25 μU/ml')

    with col3:
       BMI = st.text_input('BMI value',
                          key='BMI',
                          help='Body Mass Index. Normal: 18.5-24.9')

    with col1:
       DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function',
                                               key='DiabetesPedigreeFunction',
                                               help='Scores likelihood of diabetes based on family history')

    with col2:
       Age = st.text_input('Age (years)',
                          key='Age',
                          help='Age in years')

    diab_diagnosis = ''

    if st.button('Diabetes Test Result'):
        try:
            input_dict = {
                'Pregnancies': Pregnancies,
                'Glucose': Glucose,
                'BloodPressure': BloodPressure,
                'SkinThickness': SkinThickness,
                'Insulin': Insulin,
                'BMI': BMI,
                'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
                'Age': Age
            }

            input_df = pd.DataFrame([input_dict])

            for column in input_df.columns:
                input_df[column] = input_df[column].astype(float)

            diab_prediction = diabetes_model.predict(input_df)

            if diab_prediction[0] == 1:
                diab_diagnosis = 'The person is likely to have diabetes'
                st.warning("""
                **What is Diabetes?**
                Diabetes is a metabolic disease that causes high blood sugar levels. The hormone insulin
                moves sugar from the blood into your cells to be stored or used for energy. With diabetes,
                your body either doesn't make enough insulin or can't effectively use the insulin it makes.

                **Next Steps:**
                If you received a positive prediction, please consult with a healthcare provider for proper
                testing and diagnosis. This prediction is not a medical diagnosis.
                """)
            else:
                diab_diagnosis = 'The person is not likely to have diabetes'

            st.success(diab_diagnosis)

            try:
                glucose_value = float(Glucose)
                bmi_value = float(BMI)
                if glucose_value > 99 and diab_prediction[0] == 0:
                    st.info("""
                    **Note:** Your glucose level appears to be elevated, which can be a risk factor for prediabetes.
                    Consider discussing these results with your healthcare provider.
                    """)
                if bmi_value > 25 and diab_prediction[0] == 0:
                    st.info# -*- coding: utf-8 -*-

import pickle
import streamlit as st
import sys
import subprocess
import importlib

# Check if required packages are installed, if not install them
required_packages = {
    'streamlit_option_menu': 'streamlit-option-menu',
    'scikit-learn': 'scikit-learn',  
    'pandas': 'pandas'
}

for package_name in required_packages.values():
    try:
        if package_name == 'scikit-learn':
            importlib.import_module('sklearn')
            importlib.import_module('sklearn.model_selection')
        elif package_name == 'streamlit-option-menu':
            importlib.import_module('streamlit_option_menu')
        else:
            importlib.import_module(package_name)
    except ImportError:
        st.info(f"Installing required package: {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
        st.rerun()

# Now import after ensuring it's installed
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Loading the saved models
import pandas as pd

# Import sklearn modules with proper error handling
try:
    import sklearn
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    try:
        sklearn_version = sklearn.__version__
        st.info(f"Using scikit-learn version: {sklearn_version}")
    except AttributeError:
        st.info("scikit-learn is successfully imported")

except ImportError as e:
    st.error(f"Error importing sklearn modules: {e}")
    st.info("Attempting to reinstall scikit-learn...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "scikit-learn"])
    try:
        import sklearn
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier

        try:
            sklearn_version = sklearn.__version__
            st.info(f"Using scikit-learn version: {sklearn_version}")
        except AttributeError:
            st.info("scikit-learn is successfully imported after reinstallation")

    except ImportError as e:
        st.error(f"Still having issues with scikit-learn: {e}")
        st.error("Please try restarting the application or installing scikit-learn manually.")
        sys.exit(1)  
    st.rerun()

# Load and prepare diabetes data
diabetes_data = pd.read_csv('diabetes.csv')
X_diabetes = diabetes_data.drop('Outcome', axis=1)
y_diabetes = diabetes_data['Outcome']
diabetes_model = RandomForestClassifier()
diabetes_model.fit(X_diabetes, y_diabetes)

# Load and prepare heart disease data
heart_data = pd.read_csv('heart.csv')
X_heart = heart_data.iloc[:, :-1]  
y_heart = heart_data.iloc[:, -1]   
heart_disease_model = RandomForestClassifier()
heart_disease_model.fit(X_heart, y_heart)

# Load and prepare Parkinson's data
parkinsons_data = pd.read_csv('parkinsons.csv')
parkinsons_data = parkinsons_data.drop('name', axis=1)
X_parkinsons = parkinsons_data.drop('status', axis=1)
y_parkinsons = parkinsons_data['status']
parkinsons_model = RandomForestClassifier()
parkinsons_model.fit(X_parkinsons, y_parkinsons)

# Check if lung cancer dataset exists and load it
try:
    lung_cancer_data = pd.read_csv('lung_cancer.csv')
    X_lung_cancer = lung_cancer_data.drop('Cancer', axis=1)
    y_lung_cancer = lung_cancer_data['Cancer']
    lung_cancer_model = RandomForestClassifier()
    lung_cancer_model.fit(X_lung_cancer, y_lung_cancer)
    lung_cancer_model_ready = True
except FileNotFoundError:
    st.warning("Lung cancer dataset not found. Lung cancer prediction will be disabled.")
    lung_cancer_model_ready = False

# Check if hypothyroid dataset exists and load it
try:
    hypothyroid_data = pd.read_csv('hypothyroid.csv')
    X_hypothyroid = hypothyroid_data.drop('Hypothyroid', axis=1)
    y_hypothyroid = hypothyroid_data['Hypothyroid']
    hypothyroid_model = RandomForestClassifier()
    hypothyroid_model.fit(X_hypothyroid, y_hypothyroid)
    hypothyroid_model_ready = True
except FileNotFoundError:
    st.warning("Hypothyroid dataset not found. Hypothyroid prediction will be disabled.")
    hypothyroid_model_ready = False

#Sidebar for navigators
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction',
                            'Lung Cancer Prediction',
                            'Hypothyroid Prediction'],
                           icons = ['activity', 'heart', 'person', 'lungs', 'thermometer'],
                           default_index = 0)
    
# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):

    st.title('Diabetes Prediction Using ML')

    st.write("""
    ### About Diabetes
    Diabetes is a chronic condition that affects how your body processes blood sugar (glucose).
    This tool uses several health metrics to assess the likelihood of diabetes.

    **Note:** This tool is for educational purposes only and should not replace professional medical advice.
    """)

    diabetes_example = {
        'Pregnancies': '6',
        'Glucose': '148',
        'BloodPressure': '72',
        'SkinThickness': '35',
        'Insulin': '0',
        'BMI': '33.6',
        'DiabetesPedigreeFunction': '0.627',
        'Age': '50'
    }

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button('Fill Example Values', key='diabetes_example'):
            for key, value in diabetes_example.items():
                st.session_state[key] = value

    col1, col2, col3 = st.columns(3)

    with col1:
       Pregnancies = st.text_input('Number of Pregnancies',
                                  key='Pregnancies',
                                  help='Number of times pregnant (0 for males)')

    with col2:
       Glucose = st.text_input('Glucose Level (mg/dL)',
                              key='Glucose',
                              help='Plasma glucose concentration after 2 hours in an oral glucose tolerance test. Normal fasting: 70-99 mg/dL')

    with col3:
       BloodPressure = st.text_input('Blood Pressure (mm Hg)',
                                    key='BloodPressure',
                                    help='Diastolic blood pressure. Normal: <80 mm Hg')

    with col1:
       SkinThickness = st.text_input('Skin Thickness (mm)',
                                    key='SkinThickness',
                                    help='Triceps skin fold thickness. Measures fat content')

    with col2:
       Insulin = st.text_input('Insulin Level (μU/ml)',
                              key='Insulin',
                              help='2-Hour serum insulin. Normal fasting: <25 μU/ml')

    with col3:
       BMI = st.text_input('BMI value',
                          key='BMI',
                          help='Body Mass Index. Normal: 18.5-24.9')

    with col1:
       DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function',
                                               key='DiabetesPedigreeFunction',
                                               help='Scores likelihood of diabetes based on family history')

    with col2:
       Age = st.text_input('Age (years)',
                          key='Age',
                          help='Age in years')

    diab_diagnosis = ''

    if st.button('Diabetes Test Result'):
        try:
            input_dict = {
                'Pregnancies': Pregnancies,
                'Glucose': Glucose,
                'BloodPressure': BloodPressure,
                'SkinThickness': SkinThickness,
                'Insulin': Insulin,
                'BMI': BMI,
                'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
                'Age': Age
            }

            input_df = pd.DataFrame([input_dict])

            for column in input_df.columns:
                input_df[column] = input_df[column].astype(float)

            diab_prediction = diabetes_model.predict(input_df)

            if diab_prediction[0] == 1:
                diab_diagnosis = 'The person is likely to have diabetes'
                st.warning("""
                **What is Diabetes?**
                Diabetes is a metabolic disease that causes high blood sugar levels. The hormone insulin
                moves sugar from the blood into your cells to be stored or used for energy. With diabetes,
                your body either doesn't make enough insulin or can't effectively use the insulin it makes.

                **Next Steps:**
                If you received a positive prediction, please consult with a healthcare provider for proper
                testing and diagnosis. This prediction is not a medical diagnosis.
                """)
            else:
                diab_diagnosis = 'The person is not likely to have diabetes'

            st.success(diab_diagnosis)

            try:
                glucose_value = float(Glucose)
                bmi_value = float(BMI)
                if glucose_value > 99 and diab_prediction[0] == 0:
                    st.info("""
                    **Note:** Your glucose level appears to be elevated, which can be a risk factor for prediabetes.
                    Consider discussing these results with your healthcare provider.
                    """)
                if bmi_value > 25 and diab_prediction[0# -*- coding: utf-8 -*-

import pickle
import streamlit as st
import sys
import subprocess
import importlib

# Check if required packages are installed, if not install them
required_packages = {
    'streamlit_option_menu': 'streamlit-option-menu',
    'scikit-learn': 'scikit-learn',  # Changed from 'sklearn' to 'scikit-learn'
    'pandas': 'pandas'
}

for package_name in required_packages.values():
    try:
        # Try to import the package to check if it's installed
        if package_name == 'scikit-learn':
            # For scikit-learn, we need to check sklearn module
            importlib.import_module('sklearn')
            # Also explicitly check for sklearn.model_selection
            importlib.import_module('sklearn.model_selection')
        elif package_name == 'streamlit-option-menu':
            importlib.import_module('streamlit_option_menu')
        else:
            importlib.import_module(package_name)
    except ImportError:
        st.info(f"Installing required package: {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
        st.rerun()

# Now import after ensuring it's installed
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Loading the saved models
# We need to train models first since we only have CSV data files
import pandas as pd

# Import sklearn modules with proper error handling
try:
    # First try to import the base sklearn module
    import sklearn
    # Then import specific modules
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    # Verify sklearn is properly installed by using it
    # Get version safely
    try:
        sklearn_version = sklearn.__version__
        st.info(f"Using scikit-learn version: {sklearn_version}")
    except AttributeError:
        # If __version__ is not available, just confirm it's imported
        st.info("scikit-learn is successfully imported")

except ImportError as e:
    st.error(f"Error importing sklearn modules: {e}")
    st.info("Attempting to reinstall scikit-learn...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "scikit-learn"])
    # After reinstalling, try to import again
    try:
        import sklearn
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier

        # Try to get version again
        try:
            sklearn_version = sklearn.__version__
            st.info(f"Using scikit-learn version: {sklearn_version}")
        except AttributeError:
            st.info("scikit-learn is successfully imported after reinstallation")

    except ImportError as e:
        st.error(f"Still having issues with scikit-learn: {e}")
        st.error("Please try restarting the application or installing scikit-learn manually.")
        sys.exit(1)  # Exit if sklearn cannot be imported after reinstall attempt
    st.rerun()

# Load and prepare diabetes data
diabetes_data = pd.read_csv('diabetes.csv')
X_diabetes = diabetes_data.drop('Outcome', axis=1)
y_diabetes = diabetes_data['Outcome']
diabetes_model = RandomForestClassifier()
diabetes_model.fit(X_diabetes, y_diabetes)

# Load and prepare heart disease data
heart_data = pd.read_csv('heart.csv')
X_heart = heart_data.iloc[:, :-1]  # All columns except the last one
y_heart = heart_data.iloc[:, -1]   # Last column is the target
heart_disease_model = RandomForestClassifier()
heart_disease_model.fit(X_heart, y_heart)

# Load and prepare Parkinson's data
parkinsons_data = pd.read_csv('parkinsons.csv')
# Drop the name column as it's not a feature
parkinsons_data = parkinsons_data.drop('name', axis=1)
# Extract features and target
X_parkinsons = parkinsons_data.drop('status', axis=1)
y_parkinsons = parkinsons_data['status']
parkinsons_model = RandomForestClassifier()
parkinsons_model.fit(X_parkinsons, y_parkinsons)

# Check if lung cancer dataset exists and load it
try:
    # Load and prepare lung cancer data
    lung_cancer_data = pd.read_csv('lung_cancer.csv')
    # Assuming the target column is named 'Cancer' (1 for cancer, 0 for no cancer)
    # Modify this according to your actual dataset
    X_lung_cancer = lung_cancer_data.drop('Cancer', axis=1)
    y_lung_cancer = lung_cancer_data['Cancer']
    lung_cancer_model = RandomForestClassifier()
    lung_cancer_model.fit(X_lung_cancer, y_lung_cancer)
    lung_cancer_model_ready = True
except FileNotFoundError:
    st.warning("Lung cancer dataset not found. Lung cancer prediction will be disabled.")
    lung_cancer_model_ready = False

# Check if hypothyroid dataset exists and load it
try:
    # Load and prepare hypothyroid data
    hypothyroid_data = pd.read_csv('hypothyroid.csv')
    # Assuming the target column is named 'Hypothyroid' (1 for hypothyroid, 0 for normal)
    # Modify this according to your actual dataset
    X_hypothyroid = hypothyroid_data.drop('Hypothyroid', axis=1)
    y_hypothyroid = hypothyroid_data['Hypothyroid']
    hypothyroid_model = RandomForestClassifier()
    hypothyroid_model.fit(X_hypothyroid, y_hypothyroid)
    hypothyroid_model_ready = True
except FileNotFoundError:
    st.warning("Hypothyroid dataset not found. Hypothyroid prediction will be disabled.")
    hypothyroid_model_ready = False

# Sidebar for navigators
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',

                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction',
                            'Lung Cancer Prediction',
                            'Hypothyroid Prediction'],
                           icons=['activity', 'heart', 'person', 'lungs', 'thermometer'],
                           default_index=0)

# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):

    # page title
    st.title('Diabetes Prediction Using ML')

    st.write("""
    ### About Diabetes
    Diabetes is a chronic condition that affects how your body processes blood sugar (glucose).
    This tool uses several health metrics to assess the likelihood of diabetes.

    **Note:** This tool is for educational purposes only and should not replace professional medical advice.
    """)

    # Example values for diabetes
    diabetes_example = {
        'Pregnancies': '6',
        'Glucose': '148',
        'BloodPressure': '72',
        'SkinThickness': '35',
        'Insulin': '0',
        'BMI': '33.6',
        'DiabetesPedigreeFunction': '0.627',
        'Age': '50'
    }

    # Add example values button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button('Fill Example Values', key='diabetes_example'):
            for key, value in diabetes_example.items():
                st.session_state[key] = value

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies',
                                  key='Pregnancies',
                                  help='Number of times pregnant (0 for males)')

    with col2:
        Glucose = st.text_input('Glucose Level (mg/dL)',
                              key='Glucose',
                              help='Plasma glucose concentration after 2 hours in an oral glucose tolerance test. Normal fasting: 70-99 mg/dL')

    with col3:
        BloodPressure = st.text_input('Blood Pressure (mm Hg)',
                                    key='BloodPressure',
                                    help='Diastolic blood pressure. Normal: <80 mm Hg')

    with col1:
        SkinThickness = st.text_input('Skin Thickness (mm)',
                                    key='SkinThickness',
                                    help='Triceps skin fold thickness. Measures fat content')

    with col2:
        Insulin = st.text_input('Insulin Level (μU/ml)',
                              key='Insulin',
                              help='2-Hour serum insulin. Normal fasting: <25 μU/ml')

    with col3:
        BMI = st.text_input('BMI value',
                          key='BMI',
                          help='Body Mass Index. Normal: 18.5-24.9')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function',
                                               key='DiabetesPedigreeFunction',
                                               help='Scores likelihood of diabetes based on family history')

    with col2:
        Age = st.text_input('Age (years)',
                          key='Age',
                          help='Age in years')

    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction
    if st.button('Diabetes Test Result'):
        try:
            # Create a dictionary with feature names matching the training data
            input_dict = {
                'Pregnancies': Pregnancies,
                'Glucose': Glucose,
                'BloodPressure': BloodPressure,
                'SkinThickness': SkinThickness,
                'Insulin': Insulin