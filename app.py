import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
from fpdf import FPDF

# ==========================================
# 1. FIXED PDF FUNCTION (Error-Free)
# ==========================================
def create_pdf(name, age, gender, result, prob, medical_data):
    pdf = FPDF()
    pdf.add_page()
    
    # Header Section
    pdf.set_fill_color(44, 62, 80)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 20, txt="CLINICAL ASSESSMENT REPORT", ln=True, align='C', fill=True)
    
    # Patient Details
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=f"Patient: {name if name else 'N/A'} | Age: {age} | Gender: {gender}", ln=True, border='B')
    
    # Clinical Data Table
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=" CLINICAL MEASUREMENTS", ln=True, fill=True)
    pdf.set_font("Arial", size=10)
    for key, value in medical_data.items():
        pdf.cell(95, 8, f" {key}", border=1)
        pdf.cell(95, 8, f" {value}", border=1, ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=f"Final Diagnosis: {result} (Risk Score: {prob:.1f}%)", ln=True)
    
    # Logic to handle different FPDF versions output
    pdf_output = pdf.output(dest='S')
    if isinstance(pdf_output, str):
        return pdf_output.encode('latin-1')
    return pdf_output

# ==========================================
# 2. APP UI & INPUTS
# ==========================================
st.set_page_config(page_title="Professional AI Health", layout="wide")

try:
    pipeline = joblib.load('full_pipeline_compressed.sav')
except:
    st.error("Model file not found! Please check 'full_pipeline_compressed.sav'")

st.title("🏥 Professional Health Dashboard & AI Analytics")

col1, col2 = st.columns(2)
with col1:
    st.subheader("👤 Patient Profile")
    # FIX: Default name empty, placeholder added
    patient_name = st.text_input("Patient Full Name", value="", placeholder="Enter Name")
    age = st.number_input("Age", 1, 120, 30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    bmi = st.number_input("BMI (kg/m2)", 10.0, 50.0, 25.0)
    smoking = st.selectbox("Smoking Status", ["No", "Yes"])
    activity = st.slider("Activity (Hrs/Week)", 0.0, 10.0, 3.0)

with col2:
    st.subheader("🩸 Clinical Metrics")
    bp = st.number_input("Systolic BP (mmHg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 400, 200)
    glucose = st.number_input("Glucose Level (mg/dL)", 50, 300, 100)
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)
    diet = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
    family_hist = st.selectbox("Family History", ["No", "Yes"])

st.markdown("---")

# ==========================================
# 3. ANALYSIS & SIMULATOR
# ==========================================
if st.button("🚀 Generate Full Health Analysis"):
    # Data Prep
    features = ['Age', 'Gender', 'BMI', 'Smoking', 'AlcoholIntake', 'PhysicalActivity', 'DietQuality', 'SleepHours', 'BloodPressure', 'Cholesterol', 'Glucose', 'FamilyHistory', 'StressLevel']
    input_df = pd.DataFrame([[age, gender, bmi, smoking, "Low", activity, diet, 7.0, bp, cholesterol, glucose, family_hist, stress]], columns=features)
    
    prob = pipeline.predict_proba(input_df)[0][1] * 100
    res_text = "Chronic Disease Likely" if prob > 50 else "No Chronic Disease"

    # Results Header
    if prob > 50:
        st.error(f"### Status: {res_text} ({prob:.1f}%)")
    else:
        st.success(f"### Status: {res_text} ({prob:.1f}%)")

    # Metric Cards
    m1, m2, m3 = st.columns(3)
    m1.metric("BP Points", f"{bp}", delta=f"{bp-120}", delta_color="inverse")
    m2.metric("Glucose Points", f"{glucose}", delta=f"{glucose-100}", delta_color="inverse")
    m3.metric("BMI Points", f"{bmi:.1f}", delta=f"{bmi-22.0:.1f}", delta_color="inverse")

    # What-If Simulator
    st.markdown("---")
    st.subheader("🏁 Full Clinical Improvement Simulator")
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        t_bp = st.slider("Target BP", 80, 200, int(bp))
        t_glu = st.slider("Target Glucose", 50, 300, int(glucose))
    with sim_col2:
        t_cho = st.slider("Target Cholesterol", 100, 400, int(cholesterol))
        t_bmi_sim = st.slider("Target BMI", 10.0, 50.0, float(bmi))

    # FIX: Using .at[0, col] to avoid Index/Scalar ValueError
    sim_df = input_df.copy()
    sim_df.at[0, 'BloodPressure'] = t_bp
    sim_df.at[0, 'Glucose'] = t_glu
    sim_df.at[0, 'Cholesterol'] = t_cho
    sim_df.at[0, 'BMI'] = t_bmi_sim
    
    sim_prob = pipeline.predict_proba(sim_df)[0][1] * 100
    st.metric("Simulated Risk Score", f"{sim_prob:.1f}%", delta=f"{sim_prob-prob:.1f}%", delta_color="inverse")

    # PDF Download
    st.markdown("---")
    summary_data = {"BP": f"{bp} mmHg", "Glucose": f"{glucose} mg/dL", "Cholesterol": f"{cholesterol} mg/dL", "BMI": f"{bmi:.1f}"}
    
    try:
        pdf_bytes = create_pdf(patient_name, age, gender, res_text, prob, summary_data)
        st.download_button(
            label="📥 Download Clinical Report (PDF)",
            data=pdf_bytes,
            file_name=f"Report_{patient_name if patient_name else 'Patient'}.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"PDF Error: {e}")
