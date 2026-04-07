import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

# ==========================================
# 1. UPDATED PDF Function with Detailed Recommendations
# ==========================================
def create_pdf(name, age, gender, result, prob, recs, medical_data):
    pdf = FPDF()
    pdf.add_page()
    
    # Title Header
    pdf.set_fill_color(44, 62, 80)  # Dark Blue Header
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 20, txt="CHRONIC DISEASE ASSESSMENT REPORT", ln=True, align='C', fill=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    
    # Patient Info Section
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 10, txt=" PATIENT INFORMATION", ln=True, fill=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(95, 10, txt=f"Name: {name}", border='B')
    pdf.cell(95, 10, txt=f"Age: {age}", border='B', ln=True)
    pdf.cell(95, 10, txt=f"Gender: {gender}", border='B')
    pdf.cell(95, 10, txt=f"Date: {pd.Timestamp.now().strftime('%d-%m-%Y')}", border='B', ln=True)
    pdf.ln(5)

    # Clinical Metrics Table
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=" CLINICAL MEASUREMENTS", ln=True, fill=True)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(95, 10, "Metric Description", border=1, align='C')
    pdf.cell(95, 10, "Recorded Value", border=1, align='C', ln=True)
    
    pdf.set_font("Arial", size=10)
    for key, value in medical_data.items():
        pdf.cell(95, 8, f" {key}", border=1)
        pdf.cell(95, 8, f" {value}", border=1, ln=True)
    pdf.ln(5)

    # Risk Result
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=" DIAGNOSTIC SUMMARY", ln=True, fill=True)
    pdf.set_font("Arial", 'B', 11)
    color = (200, 0, 0) if "Detected" in result else (0, 128, 0)
    pdf.set_text_color(*color)
    pdf.cell(0, 10, txt=f"Status: {result} (Probability: {prob:.1f}%)", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # DETAILED DOCTOR RECOMMENDATIONS
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=" DOCTOR'S CLINICAL RECOMMENDATIONS", ln=True, fill=True)
    pdf.set_font("Arial", size=10)
    pdf.ln(2)
    
    if not recs:
        pdf.multi_cell(0, 8, txt="Patient exhibits healthy clinical markers. Maintain current lifestyle, balanced diet, and regular annual check-ups.")
    else:
        for r in recs:
            pdf.multi_cell(0, 8, txt=f" {r}")
    
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 10, txt="Note: This is an AI-generated assessment. Please consult a registered medical practitioner for clinical validation.", align='C')

    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 2. Main Streamlit App
# ==========================================
st.set_page_config(page_title="AI Health Expert", layout="wide")
pipeline = joblib.load('full_pipeline_compressed.sav')

st.title("🏥 Clinical Disease Prediction & Reporting")

col1, col2 = st.columns(2)
with col1:
    patient_name = st.text_input("Patient Full Name", "Enter Name")
    age = st.number_input("Age", 1, 120, 30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    bmi = st.number_input("BMI (Body Mass Index)", 10.0, 50.0, 25.0)
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    activity = st.slider("Activity (Hrs/Week)", 0.0, 10.0, 3.0)

with col2:
    bp = st.number_input("Blood Pressure (Systolic)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 400, 200)
    glucose = st.number_input("Glucose Level (mg/dL)", 50, 300, 100)
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)
    diet = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
    family_hist = st.selectbox("Family History", ["No", "Yes"])

# Prepare Clinical Data for PDF
clinical_summary = {
    "Body Mass Index": f"{bmi:.1f} kg/m2",
    "Systolic Blood Pressure": f"{bp} mmHg",
    "Total Cholesterol": f"{cholesterol} mg/dL",
    "Blood Glucose Level": f"{glucose} mg/dL",
    "Weekly Physical Activity": f"{activity} Hours",
    "Stress Assessment Score": f"{stress}/10"
}

# Features for Model
features = ['Age', 'Gender', 'BMI', 'Smoking', 'AlcoholIntake', 'PhysicalActivity', 'DietQuality', 'SleepHours', 'BloodPressure', 'Cholesterol', 'Glucose', 'FamilyHistory', 'StressLevel']
# Adding dummy values for alcohol/sleep which are in model but not critical for custom recommendations
input_df = pd.DataFrame([[age, gender, bmi, smoking, "Low", activity, diet, 7.0, bp, cholesterol, glucose, family_hist, stress]], columns=features)

if st.button("Generate Diagnostic Report"):
    prob = pipeline.predict_proba(input_df)[0][1] * 100
    res_text = "Chronic Disease Detected" if prob > 50 else "No Chronic Disease"

    # Logic-based Recommendations (The missing part)
    recs = []
    if bmi > 25: 
        recs.append("-> WEIGHT MANAGEMENT: BMI is above normal range. Clinically advised to follow a calorie-restricted diet and increase physical activity.")
    if bp > 135: 
        recs.append("-> HYPERTENSION ADVISORY: Elevated Blood Pressure detected. Reduce sodium (salt) intake and monitor BP twice daily.")
    if glucose > 140: 
        recs.append("-> DIABETIC PRECAUTION: High Glucose levels observed. Immediate reduction in refined sugars and carbohydrate monitoring is recommended.")
    if cholesterol > 240:
        recs.append("-> CHOLESTEROL CONTROL: High cholesterol detected. Focus on Omega-3 rich foods and avoid trans-fats.")
    if smoking == "Yes": 
        recs.append("-> CESSATION ADVICE: Smoking is a primary risk factor. Strongly recommend clinical support for tobacco cessation.")
    if stress > 7:
        recs.append("-> MENTAL WELLNESS: High stress score may impact cardiovascular health. Consider mindfulness or professional counseling.")

    st.markdown("---")
    st.subheader(f"Status: {res_text}")
    
    # Generate PDF
    pdf_output = create_pdf(patient_name, age, gender, res_text, prob, recs, clinical_summary)
    st.download_button(label="📥 Download Clinical PDF Report",
                       data=pdf_output,
                       file_name=f"Report_{patient_name}.pdf",
                       mime="application/pdf")
