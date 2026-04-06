import streamlit as st
import pandas as pd
import joblib

# Load the saved compressed pipeline
pipeline = joblib.load('full_pipeline_compressed.sav')

st.title("Chronic Disease Prediction App")
st.write("Enter patient details to predict the likelihood of chronic disease.")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=30)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
smoking = st.selectbox("Smoking", ["No", "Yes"])
alcohol = st.selectbox("Alcohol Intake", ["Low", "Moderate", "High"])
activity = st.slider("Physical Activity (Hours/Week)", 0.0, 10.0, 3.0)
diet = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
sleep = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
bp = st.number_input("Blood Pressure (Systolic)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol", 100, 400, 200)
glucose = st.number_input("Glucose Level", 50, 300, 100)
family_hist = st.selectbox("Family History", ["No", "Yes"])
stress = st.slider("Stress Level (1-10)", 1, 10, 5)

# Create input dataframe
input_data = pd.DataFrame([[
    age, gender, bmi, smoking, alcohol, activity, diet, sleep, bp, cholesterol, glucose, family_hist, stress
]], columns=['Age', 'Gender', 'BMI', 'Smoking', 'AlcoholIntake', 'PhysicalActivity', 'DietQuality', 'SleepHours', 'BloodPressure', 'Cholesterol', 'Glucose', 'FamilyHistory', 'StressLevel'])

if st.button("Predict"):
    prediction = pipeline.predict(input_data)
    result = "Chronic Disease Detected" if prediction[0] == 1 else "No Chronic Disease"
    st.header(f"Result: {result}")
