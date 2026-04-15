import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
from fpdf import FPDF

# ==========================================
# 1. FIXED PDF Function (With Units & Clinical Values)
# ==========================================
def create_pdf(name, age, gender, result, prob, recs, medical_data):
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_fill_color(44, 62, 80)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 20, txt="CHRONIC DISEASE DIAGNOSTIC REPORT", ln=True, align='C', fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    
    # Patient Info
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=f"Patient: {name} | Age: {age} | Gender: {gender}", ln=True, border='B')
    pdf.ln(5)

    # Clinical Metrics Table (This will now show all values)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=" CLINICAL MEASUREMENTS (Recorded Points)", ln=True, fill=True)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(95, 10, "Metric Description", border=1, align='C')
    pdf.cell(95, 10, "Recorded Value", border=1, align='C', ln=True)
    
    pdf.set_font("Arial", size=10)
    # Loop through medical_data to fill the table
    for key, value in medical_data.items():
        pdf.cell(95, 8, f" {key}", border=1)
        pdf.cell(95, 8, f" {value}", border=1, ln=True)
    pdf.ln(5)

    # Risk Result
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=f"Diagnosis: {result} (Probability: {prob:.1f}%)", ln=True)
    pdf.ln(5)

    # Recommendations
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=" DOCTOR'S RECOMMENDATIONS", ln=True, fill=True)
    pdf.set_font("Arial", size=10)
    for r in recs:
        pdf.multi_cell(0, 8, txt=r)
    
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 2. Main App Logic
# ==========================================
st.set_page_config(page_title="Professional AI Health", layout="wide")
pipeline = joblib.load('full_pipeline_compressed.sav')

st.title("🏥 Professional Health Prediction & Analytics")

col1, col2 = st.columns(2)
with col1:
    patient_name = st.text_input("Patient Full Name", "Enter Name")
    age = st.number_input("Age", 1, 120, 45)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    bmi = st.number_input("BMI (kg/m2)", 10.0, 50.0, 25.0)
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    activity = st.slider("Activity (Hrs/Week)", 0.0, 10.0, 3.0)

with col2:
    bp = st.number_input("Systolic BP (mmHg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 400, 200)
    glucose = st.number_input("Glucose Level (mg/dL)", 50, 300, 100)
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)
    diet = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
    family_hist = st.selectbox("Family History", ["No", "Yes"])

# --- This is the key fix: Mapping the data with units ---
clinical_summary = {
    "Systolic Blood Pressure": f"{bp} mmHg",
    "Blood Glucose Level": f"{glucose} mg/dL",
    "Total Cholesterol": f"{cholesterol} mg/dL",
    "Body Mass Index (BMI)": f"{bmi:.1f}",
    "Physical Activity": f"{activity} Hours/Week",
    "Stress Level Score": f"{stress}/10"
}

# Model Prediction
features = ['Age', 'Gender', 'BMI', 'Smoking', 'AlcoholIntake', 'PhysicalActivity', 'DietQuality', 'SleepHours', 'BloodPressure', 'Cholesterol', 'Glucose', 'FamilyHistory', 'StressLevel']
input_df = pd.DataFrame([[age, gender, bmi, smoking, "Low", activity, diet, 7.0, bp, cholesterol, glucose, family_hist, stress]], columns=features)

if st.button("Generate Diagnostic Report"):
    prob = pipeline.predict_proba(input_df)[0][1] * 100
    res_text = "Chronic Disease Detected" if prob > 50 else "No Chronic Disease"

    # Logic-based Recommendations
    recs = []
    if bp > 135: recs.append("-> BP is high. Reduce salt and monitor systolic levels.")
    if glucose > 140: recs.append("-> Glucose is high. Limit sugar and processed carbs.")
    if cholesterol > 240: recs.append("-> High Cholesterol. Avoid trans-fats.")
    
    st.markdown("---")
    st.subheader(f"Status: {res_text} ({prob:.1f}%)")

    # Generate PDF with the fixed clinical_summary
    pdf_output = create_pdf(patient_name, age, gender, res_text, prob, recs, clinical_summary)
    st.download_button(label="📥 Download Detailed PDF Report",
                       data=pdf_output,
                       file_name=f"Report_{patient_name}.pdf",
                       mime="application/pdf")
