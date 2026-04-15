import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
from fpdf import FPDF

# ==========================================
# 1. PROFESSIONAL PDF FUNCTION
# ==========================================
def create_pdf(name, age, gender, result, prob, score, medical_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(44, 62, 80)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 20, txt="CHRONIC DISEASE ASSESSMENT REPORT", ln=True, align='C', fill=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=f"Patient: {name if name else 'N/A'} | Age: {age} | Gender: {gender}", ln=True, border='B')
    
    pdf.ln(5)
    pdf.cell(0, 10, txt=f"Lifestyle Health Score: {score}/100", ln=True)
    pdf.cell(0, 10, txt=f"Final Diagnosis: {result} (Risk: {prob:.1f}%)", ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=" CLINICAL MEASUREMENTS", ln=True, fill=True)
    pdf.set_font("Arial", size=10)
    for key, value in medical_data.items():
        pdf.cell(95, 8, f" {key}", border=1)
        pdf.cell(95, 8, f" {value}", border=1, ln=True)
    
    pdf_output = pdf.output(dest='S')
    if isinstance(pdf_output, str):
        return pdf_output.encode('latin-1')
    return pdf_output

# ==========================================
# 2. MAIN APP & INPUTS
# ==========================================
st.set_page_config(page_title="Professional AI Health", layout="wide")
pipeline = joblib.load('full_pipeline_compressed.sav')

st.title("🏥 Professional Health Dashboard & AI Analytics")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.subheader("👤 Patient Profile")
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

clinical_summary = {
    "BP": f"{bp} mmHg",
    "Glucose": f"{glucose} mg/dL",
    "Cholesterol": f"{cholesterol} mg/dL",
    "BMI": f"{bmi:.1f}"
}

# ==========================================
# 3. ANALYSIS & NEW FEATURES
# ==========================================
if st.button("🚀 Generate Full Health Analysis"):
    # A. PREDICTION
    features = ['Age', 'Gender', 'BMI', 'Smoking', 'AlcoholIntake', 'PhysicalActivity', 'DietQuality', 'SleepHours', 'BloodPressure', 'Cholesterol', 'Glucose', 'FamilyHistory', 'StressLevel']
    input_df = pd.DataFrame([[age, gender, bmi, smoking, "Low", activity, diet, 7.0, bp, cholesterol, glucose, family_hist, stress]], columns=features)
    
    prob = pipeline.predict_proba(input_df)[0][1] * 100
    res_text = "Chronic Disease Likely" if prob > 50 else "No Chronic Disease"

    st.markdown("---")
    
    # B. FEATURE 6: LIFESTYLE SCORE
    st.subheader("🎯 Lifestyle Health Score")
    l_score = 100
    if smoking == "Yes": l_score -= 30
    if diet == "Poor": l_score -= 20
    if activity < 2: l_score -= 20
    if stress > 7: l_score -= 15
    l_score = max(0, min(100, l_score))
    
    s_col1, s_col2 = st.columns([2, 1])
    with s_col1:
        st.progress(l_score / 100)
    with s_col2:
        score_color = "green" if l_score > 70 else "orange" if l_score > 40 else "red"
        st.markdown(f"Score: :{score_color}[**{l_score}/100**]")

    # C. WHO CLASSIFICATION (Feature 7)
    st.subheader("🔬 Clinical Classification (WHO Standards)")
    c1, c2 = st.columns(2)
    with c1:
        if bp < 120: st.info("BP Status: Normal")
        elif 120 <= bp < 140: st.warning("BP Status: Pre-hypertension")
        else: st.error("BP Status: Hypertension (High BP)")
    with c2:
        if glucose < 100: st.info("Glucose: Normal")
        elif 100 <= glucose < 126: st.warning("Glucose: Prediabetic")
        else: st.error("Glucose: Diabetic Range")

    # D. IMPROVEMENT SIMULATOR
    st.markdown("---")
    st.subheader("🏁 Full Clinical Improvement Simulator")
    t_bp = st.slider("Target BP", 80, 200, int(bp))
    sim_df = input_df.copy()
    sim_df.at[0, 'BloodPressure'] = t_bp
    sim_prob = pipeline.predict_proba(sim_df)[0][1] * 100
    st.metric("Simulated Risk Score", f"{sim_prob:.1f}%", delta=f"{sim_prob-prob:.1f}%", delta_color="inverse")

    # PDF DOWNLOAD
    pdf_bytes = create_pdf(patient_name, age, gender, res_text, prob, l_score, clinical_summary)
    st.download_button("📥 Download Analytical Report (PDF)", pdf_bytes, f"Report_{patient_name}.pdf", "application/pdf")
