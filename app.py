import os
import pickle
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from streamlit_option_menu import option_menu
import pandas as pd
import streamlit.components.v1 as components

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Load saved models
working_dir = os.path.dirname(os.path.abspath(__file__))
diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))

# Function for input validation

def validate_input(value, feature_name, min_value=None, max_value=None):
    try:
        value = float(value)
    except ValueError:
        st.warning(f"Please enter a valid number for {feature_name}.")
        return False
    
    if min_value is not None and value < min_value:
        st.warning(f"{feature_name} cannot be less than {min_value}.")
        return False
    if max_value is not None and value > max_value:
        st.warning(f"{feature_name} cannot be greater than {max_value}.")
        return False
    
    st.write(f"Validated {feature_name}: {value}")  # Debugging statement
    return value

# Function for feature scaling
def scale_input(features):
    scaler = MinMaxScaler()
    scaler.fit(features)
    scaled_features = scaler.transform(features)
    return scaled_features

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            "Parkinson's Prediction"],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies',placeholder='range of 0 to 20')

    with col2:
        Glucose = st.text_input('Glucose Level',placeholder='range of 0 to 250')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value',placeholder='range of 0 to 200')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value',placeholder='range of 0 to 150')

    with col2:
        Insulin = st.text_input('Insulin Level',placeholder='range of 0 to 900')

    with col3:
        BMI = st.text_input('BMI value',placeholder='range of 0 to 80')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value',placeholder='range of 0 to 3')

    with col2:
        Age = st.text_input('Age of the Person',placeholder='range of 0 to 150')

    diab_prediction = None

    if st.button('Diabetes Test Result'):
        # Validate input
        valid_input = True
        if not (Pregnancies.strip().replace('.', '', 1).isdigit() and 0 <= float(Pregnancies) <= 20):
            st.warning("Please enter a valid number for Number of Pregnancies within the range of 0 to 20.")
            valid_input = False
        if not (Glucose.strip().replace('.', '', 1).isdigit() and 0 <= float(Glucose) <= 250):
            st.warning("Please enter a valid number for Glucose Level within the range of 0 to 250.")
            valid_input = False
        if not (BloodPressure.strip().replace('.', '', 1).isdigit() and 0 <= float(BloodPressure) <= 200):
            st.warning("Please enter a valid number for Blood Pressure value within the range of 0 to 200.")
            valid_input = False
        if not (SkinThickness.strip().replace('.', '', 1).isdigit() and 0 <= float(SkinThickness) <= 150):
            st.warning("Please enter a valid number for Skin Thickness value within the range of 0 to 150.")
            valid_input = False
        if not (Insulin.strip().replace('.', '', 1).isdigit() and 0 <= float(Insulin) <= 900):
            st.warning("Please enter a valid number for Insulin Level within the range of 0 to 900.")
            valid_input = False
        if not (BMI.strip().replace('.', '', 1).isdigit() and 0 <= float(BMI) <= 80):
            st.warning("Please enter a valid number for BMI value within the range of 0 to 80.")
            valid_input = False
        if not (DiabetesPedigreeFunction.strip().replace('.', '', 1).isdigit() and 0 <= float(DiabetesPedigreeFunction) <= 3):
            st.warning("Please enter a valid number for Diabetes Pedigree Function value within the range of 0 to 3.")
            valid_input = False
        if not (Age.strip().replace('.', '', 1).isdigit() and 0 <= float(Age) <= 150):
            st.warning("Please enter a valid number for Age of the Person within the range of 0 to 150.")
            valid_input = False
        
        if valid_input:
            user_input = [float(x) for x in [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                        BMI, DiabetesPedigreeFunction, Age]]

            diab_prediction = diabetes_model.predict([user_input])

            # Display output message
        if diab_prediction is not None:
            if diab_prediction[0] == 1:
                st.success('The person is diabetic')
            else:
                st.success('The person is not diabetic')

            if valid_input:
                user_input = [float(x) for x in [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                            BMI, DiabetesPedigreeFunction, Age]]

                if diab_prediction[0] == 1:
                    diab_diagnosis = 'The person is diabetic'
                else:
                    diab_diagnosis = 'The person is not diabetic'

                # Create a DataFrame to store the entered values and the output
                df_report = pd.DataFrame({'Number of Pregnancies': [Pregnancies],
                                        'Glucose Level': [Glucose],
                                        'Blood Pressure value': [BloodPressure],
                                        'Skin Thickness value': [SkinThickness],
                                        'Insulin Level': [Insulin],
                                        'BMI value': [BMI],
                                        'Diabetes Pedigree Function value': [DiabetesPedigreeFunction],
                                        'Age of the Person': [Age],
                                        'Diabetes Prediction': [diab_diagnosis]})
                # Display the DataFrame
                st.write("### Report:")
                st.write(df_report)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age',placeholder='range of 0 to 150')

    with col2:
        sex = st.text_input('Sex',placeholder='range of 0(Male) to 1(Female)')

    with col3:
        cp = st.text_input('Chest Pain types',placeholder='range of 0 to 3')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure',placeholder='range of 0 to 250')

    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl',placeholder='range of 0 to 900')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl',placeholder='range of 0 to 2')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results',placeholder='range of 0 to 3')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved',placeholder='range of 0 to 250')

    with col3:
        exang = st.text_input('Exercise Induced Angina',placeholder='range of 0 to 2')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise',placeholder='range of 0 to 7')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment',placeholder='range of 0 to 3')

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy',placeholder='range of 0 to 5')

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect',placeholder='range of 0 to 5')

    # code for Prediction
    heart_diagnosis = None
    heart_prediction = None

    if st.button('Heart Disease Test Result'):
    # Validate input
        valid_input = True
        if not (age.strip().replace('.', '', 1).isdigit() and 0 <= float(age) <= 150):
            st.warning("Please enter a valid number for Number of age within the range of 0 to 150.")
            valid_input = False
        if not (sex.strip().replace('.', '', 1).isdigit() and 0 <= float(sex) <= 1):
            st.warning("Please enter a valid number for sex within the range of 0(Male) to 1(Female).")
            valid_input = False
        if not (cp.strip().replace('.', '', 1).isdigit() and 0 <= float(cp) <= 3):
            st.warning("Please enter a valid number for Chest Pain type value within the range of 0 to 3.")
            valid_input = False
        if not (trestbps.strip().replace('.', '', 1).isdigit() and 0 <= float(trestbps) <= 250):
            st.warning("Please enter a valid number for Resting Blood Pressure value within the range of 0 to 250.")
            valid_input = False
        if not (chol.strip().replace('.', '', 1).isdigit() and 0 <= float(chol) <= 900):
            st.warning("Please enter a valid number for Serum Cholestoral in mg/dl within the range of 0 to 900.")
            valid_input = False
        if not (fbs.strip().replace('.', '', 1).isdigit() and 0 <= float(fbs) <= 2):
            st.warning("Please enter a valid number for Fasting Blood Sugar value within the range of 0 to 2.")
            valid_input = False
        if not (restecg.strip().replace('.', '', 1).isdigit() and 0 <= float(restecg) <= 3):
            st.warning("Please enter a valid number for Resting Electrocardiographic results value within the range of 0 to 3.")
            valid_input = False
        if not (thalach.strip().replace('.', '', 1).isdigit() and 0 <= float(thalach) <= 250):
            st.warning("Please enter a valid number for Maximum Heart Rate achieved of the Person within the range of 0 to 250.")
            valid_input = False
        if not (exang.strip().replace('.', '', 1).isdigit() and 0 <= float(exang) <= 2):
            st.warning("Please enter a valid number for Exercise Induced Angina value within the range of 0 to 2.")
            valid_input = False

        if not (oldpeak.strip().replace('.', '', 1).isdigit() and 0 <= float(oldpeak) <= 7):
            st.warning("Please enter a valid number for ST depression induced by exercise value within the range of 0 to 7.")
            valid_input = False
        if not (slope.strip().replace('.', '', 1).isdigit() and 0 <= float(slope) <= 3):
            st.warning("Please enter a valid number for Slope of the peak exercise ST segment value within the range of 0 to 3.")
            valid_input = False
        if not (ca.strip().replace('.', '', 1).isdigit() and 0 <= float(ca) <= 5):
            st.warning("Please enter a valid number for Major vessels colored by flourosopy value within the range of 0 to 5.")
            valid_input = False
        if not (thal.strip().replace('.', '', 1).isdigit() and 0 <= float(thal) <= 5):
            st.warning("Please enter a valid number for thal value within the range of 0 to 5.")
            valid_input = False

        if valid_input:
            user_input = [float(x) for x in [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]

            heart_prediction = heart_disease_model.predict([user_input])

                # Display output message
        if heart_prediction is not None:
            if heart_prediction[0] == 1:
                st.success('The person is having heart disease')
            else:
                st.success('The person does not have any heart disease')

            if valid_input:
                user_input = [float(x) for x in [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]

                if heart_prediction[0] == 1:
                    heart_diagnosis = 'The person is having heart disease'
                else:
                    heart_diagnosis = 'The person does not have any heart disease'

                # Create a DataFrame to store the entered values and the output
                df_report = pd.DataFrame({'Age': [age],
                              'Sex': [sex],
                              'Chest Pain types': [cp],
                              'Resting Blood Pressure': [trestbps],
                              'Serum Cholestoral in mg/dl': [chol],
                              'Fasting Blood Sugar > 120 mg/dl': [fbs],
                              'Resting Electrocardiographic results': [restecg],
                              'Maximum Heart Rate achieved': [thalach],
                              'Exercise Induced Angina': [exang],
                              'ST depression induced by exercise': [oldpeak],
                              'Slope of the peak exercise ST segment': [slope],
                              'Major vessels colored by flourosopy': [ca],
                              'Thal': [thal]
                              ,'Heart Disease Prediction':[heart_diagnosis]})
                # Display the DataFrame
                st.write("### Report:")
                st.write(df_report)

# Parkinson's Prediction Page
if selected == "Parkinson's Prediction":

    # page title
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP Fo(Hz)', placeholder='range of 0 to 300')

    with col2:
        fhi = st.text_input('MDVP Fhi(Hz)', placeholder='range of 0 to 650')

    with col3:
        flo = st.text_input('MDVP Flo(Hz)', placeholder='range of 0 to 300')

    with col4:
        Jitter_percent = st.text_input('MDVP Jitter(%)', placeholder='range of 0 to 1')

    with col5:
        Jitter_Abs = st.text_input('MDVP Jitter(Abs)', placeholder='range of 0 to 1')

    with col1:
        RAP = st.text_input('MDVP RAP', placeholder='range of 0 to 1')

    with col2:
        PPQ = st.text_input('MDVP PPQ', placeholder='range of 0 to 1')

    with col3:
        DDP = st.text_input('Jitter DDP', placeholder='range of 0 to 1')

    with col4:
        Shimmer = st.text_input('MDVP Shimmer', placeholder='range of 0 to 1')

    with col5:
        Shimmer_dB = st.text_input('MDVP Shimmer(dB)', placeholder='range of 0 to 2')

    with col1:
        APQ3 = st.text_input('Shimmer APQ3', placeholder='range of 0 to 1')

    with col2:
        APQ5 = st.text_input('Shimmer APQ5', placeholder='range of 0 to 1')

    with col3:
        APQ = st.text_input('MDVP APQ', placeholder='range of 0 to 1')

    with col4:
        DDA = st.text_input('Shimmer DDA', placeholder='range of 0 to 1')

    with col5:
        NHR = st.text_input('NHR', placeholder='range of 0 to 1')

    with col1:
        HNR = st.text_input('HNR', placeholder='range of 0 to 40')

    with col2:
        RPDE = st.text_input('RPDE', placeholder='range of 0 to 1')

    with col3:
        DFA = st.text_input('DFA', placeholder='range of 0 to 1')

    with col4:
        spread1 = st.text_input('spread1', placeholder='range of -6 to 0')

    with col5:
        spread2 = st.text_input('spread2', placeholder='range of 0 to 1')

    with col1:
        D2 = st.text_input('D2', placeholder='range of 0 to 5')

    with col2:
        PPE = st.text_input('PPE', placeholder='range of 0 to 1')

    # code for Prediction
    parkinsons_diagnosis = None
    parkinsons_prediction = None

    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        # Validate input
        valid_input = True
        if not (fo.strip().replace('.', '', 1).isdigit() and 0 <= float(fo) <= 300):
            st.warning("Please enter a valid number for MDVP:Fo(Hz) within the range of 0 to 300.")
            valid_input = False
        if not (fhi.strip().replace('.', '', 1).isdigit() and 0 <= float(fhi) <= 650):
            st.warning("Please enter a valid number for MDVP:Fhi(Hz) within the range of 0 to 650.")
            valid_input = False
        if not (flo.strip().replace('.', '', 1).isdigit() and 0 <= float(flo) <= 300):
            st.warning("Please enter a valid number for MDVP:Flo(Hz) value within the range of 0 to 300.")
            valid_input = False
        if not (Jitter_percent.strip().replace('.', '', 1).isdigit() and 0 <= float(Jitter_percent) <= 1):
            st.warning("Please enter a valid number for MDVP:Jitter(%) value within the range of 0 to 1.")
            valid_input = False
        if not (Jitter_Abs.strip().replace('.', '', 1).isdigit() and 0 <= float(Jitter_Abs) <= 1):
            st.warning("Please enter a valid number for MDVP:Jitter(Abs) value within the range of 0 to 1.")
            valid_input = False
        if not (RAP.strip().replace('.', '', 1).isdigit() and 0 <= float(RAP) <= 1):
            st.warning("Please enter a valid number for MDVP:RAP value within the range of 0 to 1.")
            valid_input = False
        if not (PPQ.strip().replace('.', '', 1).isdigit() and 0 <= float(PPQ) <= 1):
            st.warning("Please enter a valid number for MDVP:PPQ within the range of 0 to 1.")
            valid_input = False
        if not (DDP.strip().replace('.', '', 1).isdigit() and 0 <= float(DDP) <= 1):
            st.warning("Please enter a valid number for Jitter:DDP within the range of 0 to 1.")
            valid_input = False
        if not (Shimmer.strip().replace('.', '', 1).isdigit() and 0 <= float(Shimmer) <= 1):
            st.warning("Please enter a valid number for MDVP:Shimmer value within the range of 0 to 1.")
            valid_input = False
        if not (Shimmer_dB.strip().replace('.', '', 1).isdigit() and 0 <= float(Shimmer_dB) <= 2):
            st.warning("Please enter a valid number for MDVP:Shimmer(dB) value within the range of 0 to 2.")
            valid_input = False
        if not (APQ3.strip().replace('.', '', 1).isdigit() and 0 <= float(APQ3) <= 1):
            st.warning("Please enter a valid number for Shimmer:APQ3 within the range of 0 to 1.")
            valid_input = False
        if not (APQ5.strip().replace('.', '', 1).isdigit() and 0 <= float(APQ5) <= 1):
            st.warning("Please enter a valid number for Shimmer:APQ5 within the range of 0 to 1.")
            valid_input = False
        if not (APQ.strip().replace('.', '', 1).isdigit() and 0 <= float(APQ) <= 1):
            st.warning("Please enter a valid number for MDVP:APQ value within the range of 0 to 1.")
            valid_input = False
        if not (DDA.strip().replace('.', '', 1).isdigit() and 0 <= float(DDA) <= 1):
            st.warning("Please enter a valid number for Shimmer:DDA value within the range of 0 to 1.")
            valid_input = False
        if not (NHR.strip().replace('.', '', 1).isdigit() and 0 <= float(NHR) <= 1):
            st.warning("Please enter a valid number for NHR within the range of 0 to 1.")
            valid_input = False
        if not (HNR.strip().replace('.', '', 1).isdigit() and 0 <= float(HNR) <= 40):
            st.warning("Please enter a valid number for HNR value within the range of 0 to 40.")
            valid_input = False
        if not (RPDE.strip().replace('.', '', 1).isdigit() and 0 <= float(RPDE) <= 1):
            st.warning("Please enter a valid number for RPDE value within the range of 0 to 1.")
            valid_input = False
        if not (DFA.strip().replace('.', '', 1).isdigit() and 0 <= float(DFA) <= 1):
            st.warning("Please enter a valid number for DFA within the range of 0 to 1.")
            valid_input = False        
        if not (spread1.strip().replace('.', '', 1).replace('-', '', 1).isdigit() and -6 <= float(spread1) <= 0):
            st.warning("Please enter a valid number for spread1 within the range of -6 to 0.")

            valid_input = False
        if not (spread2.strip().replace('.', '', 1).isdigit() and 0 <= float(spread2) <= 1):
            st.warning("Please enter a valid number for spread2 value within the range of 0 to 1.")
            valid_input = False
        if not (D2.strip().replace('.', '', 1).isdigit() and 0 <= float(D2) <= 5):
            st.warning("Please enter a valid number for D2 value within the range of 0 to 5.")
            valid_input = False
        if not (PPE.strip().replace('.', '', 1).isdigit() and 0 <= float(PPE) <= 1):
            st.warning("Please enter a valid number for PPE within the range of 0 to 1.")
            valid_input = False
        
        if valid_input:
            user_input = [float(x) for x in [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                        RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5,
                        APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]]

            parkinsons_prediction = parkinsons_model.predict([user_input])

            # Display output message
        if parkinsons_prediction is not None:
            if parkinsons_prediction[0] == 1:
                st.success("The person has Parkinson's Disease")
            else:
                st.success("The person does not have Parkinson's Disease")

            if valid_input:
                user_input = [float(x) for x in [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                        RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5,
                        APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]]

                if parkinsons_prediction[0] == 1:
                    parkinsons_diagnosis = "The person has Parkinson's Disease"
                else:
                    parkinsons_diagnosis = "The person does not have Parkinson's Disease"

                # Create a DataFrame to store the entered values and the output
                df_report = pd.DataFrame({'MDVP:Fo(Hz)': [fo],
                                          'MDVP:Fhi(Hz)': [fhi],
                                          'MDVP:Flo(Hz)': [flo],
                                          'MDVP:Jitter(%)': [Jitter_percent],
                                          'MDVP:Jitter(Abs)': [Jitter_Abs],
                                          'MDVP:RAP': [RAP],
                                          'MDVP:PPQ': [PPQ],
                                          'Jitter:DDP': [DDP],
                                          'MDVP:Shimmer': [Shimmer],
                                          'MDVP:Shimmer(dB)': [Shimmer_dB],
                                          'Shimmer:APQ3': [APQ3],
                                          'Shimmer:APQ5': [APQ5],
                                          'MDVP:APQ': [APQ],
                                          'Shimmer:DDA': [DDA],
                                          'NHR': [NHR],
                                          'HNR': [HNR],
                                          'RPDE': [RPDE],
                                          'DFA': [DFA],
                                          'spread1': [spread1],
                                          'spread2': [spread2],
                                          'D2': [D2],
                                          'PPE': [PPE],
                                          "Parkinson's Disease Prediction": [parkinsons_diagnosis]})
                # Display the DataFrame
                st.write("### Report:")
                st.write(df_report) 