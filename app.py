import streamlit as st
import pickle
import numpy as np

# Page configuration
st.set_page_config(page_title="AI Multi-Disease Predictor", layout="wide", page_icon="🏥")

# Sidebar for navigation
with st.sidebar:
    st.title("Main Menu")
    selection = st.radio("Select a Project:", ["Chronic Disease Prediction", "Credit Card Fraud Detection"])

# --- 1. Chronic Disease Prediction Page ---
if selection == "Chronic Disease Prediction":
    st.title("🏥 Chronic Disease (Diabetes) Prediction")
    st.write("Please enter the patient's clinical metrics below:")

    # Load the Disease Model
    try:
        disease_model = pickle.load(open('chronic_disease_model.sav', 'rb'))
    except FileNotFoundError:
        st.error("Error: 'chronic_disease_model.sav' file not found!")

    # UI setup with 2 columns
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input('Age', min_value=1, step=1, value=25)
        bmi = st.number_input('BMI', value=22.0)
        phys = st.number_input('Physical Activity (hours/week)', value=2.0)
        sleep = st.number_input('Sleep Hours', value=7.0)
        bp = st.number_input('Blood Pressure', value=80.0)

    with col2:
        chol = st.number_input('Cholesterol', value=150.0)
        glucose = st.number_input('Glucose Level', value=90.0)
        stress = st.number_input('Stress Level (1-10)', value=5.0)
        gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
        smoking = st.selectbox('Smoking Status', ['No', 'Yes'])
        alcohol = st.selectbox('Alcohol Intake', ['Low', 'Moderate', 'High'])
        diet = st.selectbox('Diet Quality', ['Good', 'Average', 'Poor'])
        fam_hist = st.selectbox('Family History of Diabetes', ['No', 'Yes'])

    if st.button("Predict Disease Status"):
        # 1. Initialize all 21 features with 0.0
        input_data = [0.0] * 21
        
        # 2. Map Numerical Features (Indices 0 to 7)
        input_data[0] = float(age)
        input_data[1] = float(bmi)
        input_data[2] = float(phys)
        input_data[3] = float(sleep)
        input_data[4] = float(bp)
        input_data[5] = float(chol)
        input_data[6] = float(glucose)
        input_data[7] = float(stress)
        
        # 3. Map Categorical Features (One-Hot Encoding logic)
        # Gender
        if gender == 'Female': input_data[8] = 1.0
        elif gender == 'Male': input_data[9] = 1.0
        else: input_data[10] = 1.0
        
        # Smoking
        if smoking == 'No': input_data[11] = 1.0
        else: input_data[12] = 1.0
        
        # Alcohol
        if alcohol == 'High': input_data[13] = 1.0
        elif alcohol == 'Low': input_data[14] = 1.0
        else: input_data[15] = 1.0
        
        # Diet
        if diet == 'Average': input_data[16] = 1.0
        elif diet == 'Good': input_data[17] = 1.0
        else: input_data[18] = 1.0
        
        # Family History
        if fam_hist == 'No': input_data[19] = 1.0
        else: input_data[20] = 1.0

        # 4. Final Prediction
        prediction = disease_model.predict([input_data])
        
        if prediction[0] == 1:
            st.warning("⚠️ High Risk: The person is likely to have Chronic Disease.")
        else:
            st.success("🎉 Low Risk: The person is Healthy.")
    
            
            
# --- 2. Credit Card Fraud Detection Page ---
elif selection == "Credit Card Fraud Detection":
    st.title("🚨 Credit Card Fraud Detection")
    st.write("Enter transaction details to analyze fraud risk.")

    # Load the Fraud Model
    try:
        fraud_model = pickle.load(open('credit_card_fraud_model.sav', 'rb'))
    except FileNotFoundError:
        st.error("Error: 'credit_card_fraud_model.sav' file not found in repository!")

    v1 = st.number_input("Feature V1")
    v2 = st.number_input("Feature V2")
    amount = st.number_input("Transaction Amount")

    if st.button("Detect Fraud"):
        # We need 30 features for the model (rest 27 are set to 0)
        features = np.zeros(30)
        features[1] = v1
        features[2] = v2
        features[29] = amount
        
        fraud_prediction = fraud_model.predict([features])
        
        if fraud_prediction[0] == 1:
            st.error("🚨 ALERT: This is a Fraudulent Transaction!")
        else:
            st.success("✅ Safe: This is a Normal Transaction.")
            
