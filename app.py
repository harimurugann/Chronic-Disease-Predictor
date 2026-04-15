import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
from fpdf import FPDF

# ==========================================
# 1. UPDATED PDF FUNCTION (Error-Free)
# ==========================================
def create_pdf(name, age, gender, result, prob, recs, medical_data, bmi_val):
    pdf = FPDF()
    pdf.add_page()
    
    # PDF-la Tamil encoding error thavirkka English-la headers vachukalam
    pdf.set_fill_color(44, 62, 80)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 20, txt="CLINICAL HEALTH ASSESSMENT REPORT", ln=True, align='C', fill=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 11)
    
    # Simple logic to handle results in English for PDF stability
    status = "Chronic Disease Risk Detected" if prob > 50 else "Low Risk / Healthy"
    
    pdf.cell(0, 10, txt=f"Patient: {name} | Age: {age} | Gender: {gender}", ln=True, border='B')
    pdf.ln(5)
    pdf.cell(0, 10, txt=f"Diagnosis: {status} (Score: {prob:.1f}%)", ln=True)
    pdf.cell(0, 10, txt=f"Body Mass Index (BMI): {bmi_val:.1f}", ln=True)
    pdf.ln(5)

    pdf.cell(0, 10, txt=" MEASURED CLINICAL VALUES", ln=True, fill=True)
    pdf.set_font("Arial", size=10)
    for k, v in medical_data.items():
        # ASCII encoding safe check
        clean_k = str(k).encode('ascii', 'ignore').decode('ascii')
        pdf.cell(95, 8, txt=f" {clean_k if clean_k else 'Metric'}", border=1)
        pdf.cell(95, 8, txt=f" {v}", border=1, ln=True)
    
    return pdf.output(dest='S').encode('latin-1', 'ignore')

# ==========================================
# 2. APP CONFIG & TRANSLATION
# ==========================================
st.set_page_config(page_title="Health AI Pro", layout="wide")
pipeline = joblib.load('full_pipeline_compressed.sav')

lang_dict = {
    "English": {
        "title": "🏥 Clinical Dashboard & AI Analyst",
        "profile": "👤 Patient Profile", "vitals": "🩸 Clinical Vitals",
        "btn": "🚀 Run Full Analysis", "bmi_cat": "BMI Category",
        "sim": "🏁 Clinical Improvement Simulator"
    },
    "Tamil": {
        "title": "🏥 மருத்துவ நலப் பரிசோதனை மற்றும் பகுப்பாய்வு",
        "profile": "👤 நோயாளியின் விவரங்கள்", "vitals": "🩸 மருத்துவ அளவீடுகள்",
        "btn": "🚀 முழுமையான பரிசோதனையைத் தொடங்கு", "bmi_cat": "உடல் எடை வகை (BMI)",
        "sim": "🏁 உடல்நிலை முன்னேற்ற சிமுலேட்டர்"
    }
}

sel_lang = st.sidebar.selectbox("🌐 Choose Language / மொழியைத் தேர்ந்தெடுக்கவும்", ["English", "Tamil"])
L = lang_dict[sel_lang]

st.title(L["title"])

# ==========================================
# 3. INPUT SECTION
# ==========================================
col1, col2 = st.columns(2)
with col1:
    st.subheader(L["profile"])
    p_name = st.text_input("Name / பெயர்", "Palanisamy")
    age = st.number_input("Age / வயது", 1, 120, 45)
    gender = st.selectbox("Gender / பாலினம்", ["Male", "Female"])
    weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)
    height = st.number_input("Height (cm)", 100.0, 250.0, 170.0)
    # BMI Calculation
    bmi = weight / ((height/100)**2)
    smoking = st.selectbox("Smoking", ["No", "Yes"])

with col2:
    st.subheader(L["vitals"])
    bp = st.number_input("Systolic BP (mmHg)", 80, 200, 120)
    glucose = st.number_input("Glucose (mg/dL)", 50, 300, 100)
    chol = st.number_input("Cholesterol (mg/dL)", 100, 400, 200)
    activity = st.slider("Activity (Hrs/Week)", 0.0, 10.0, 3.0)
    diet = st.selectbox("Diet", ["Good", "Average", "Poor"])

# BMI Categorization Logic
if bmi < 18.5: bmi_msg = "Underweight"
elif 18.5 <= bmi < 25: bmi_msg = "Normal"
elif 25 <= bmi < 30: bmi_msg = "Overweight"
else: bmi_msg = "Obese"

# ==========================================
# 4. ANALYSIS TRIGGER
# ==========================================
if st.button(L["btn"]):
    # Model Input
    features = ['Age', 'Gender', 'BMI', 'Smoking', 'AlcoholIntake', 'PhysicalActivity', 'DietQuality', 'SleepHours', 'BloodPressure', 'Cholesterol', 'Glucose', 'FamilyHistory', 'StressLevel']
    input_df = pd.DataFrame([[age, gender, bmi, smoking, "Low", activity, diet, 7.0, bp, chol, glucose, "No", 5]], columns=features)
    
    prob = pipeline.predict_proba(input_df)[0][1] * 100
    res = "Chronic Disease Detected" if prob > 50 else "No Disease Detected"

    st.markdown("---")
    # Result & BMI Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Risk Score", f"{prob:.1f}%")
    m2.metric("BMI Value", f"{bmi:.1f}")
    m3.metric(L["bmi_cat"], bmi_msg)
    m4.metric("BP Points", f"{bp}", delta=f"{bp-120}", delta_color="inverse")

    # 

[Image of BMI categories chart]

    
    # Improvement Simulator
    st.subheader(L["sim"])
    t_bp = st.slider("Target BP", 80, 180, int(bp))
    sim_df = input_df.copy(); sim_df['BloodPressure'] = t_bp
    s_prob = pipeline.predict_proba(sim_df)[0][1] * 100
    st.metric("New Risk Score", f"{s_prob:.1f}%", delta=f"{s_prob-prob:.1f}%", delta_color="inverse")

    # PDF Download
    summary = {"Blood Pressure": f"{bp} mmHg", "Glucose": f"{glucose} mg/dL", "Cholesterol": f"{chol} mg/dL"}
    pdf_bytes = create_pdf(p_name, age, gender, res, prob, [], summary, bmi)
    st.download_button("📥 Download Report (English PDF)", pdf_bytes, f"Report_{p_name}.pdf")
