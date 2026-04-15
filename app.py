import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
from fpdf import FPDF

# ==========================================
# 1. MULTI-LANGUAGE DICTIONARY
# ==========================================
lang_data = {
    "English": {
        "title": "🏥 Clinical Health Dashboard",
        "profile": "👤 Patient Profile",
        "name": "Full Name",
        "age": "Age",
        "gender": "Gender",
        "vitals": "🩸 Clinical Vitals",
        "bp": "Systolic BP (mmHg)",
        "glucose": "Glucose Level (mg/dL)",
        "chol": "Cholesterol (mg/dL)",
        "run_btn": "🚀 Run Comprehensive Analysis",
        "result_detected": "Chronic Disease Detected",
        "result_safe": "No Chronic Disease",
        "sim_title": "🏁 Full Clinical Improvement Simulator",
        "target": "Target",
        "download": "📥 Download Clinical PDF Report",
        "report_title": "HEALTH ASSESSMENT REPORT"
    },
    "Tamil": {
        "title": "🏥 மருத்துவ நலப் பரிசோதனை மற்றும் பகுப்பாய்வு",
        "profile": "👤 நோயாளியின் விவரங்கள்",
        "name": "முழு பெயர்",
        "age": "வயது",
        "gender": "பாலினம்",
        "vitals": "🩸 மருத்துவ அளவீடுகள்",
        "bp": "இரத்த அழுத்தம் (BP)",
        "glucose": "சர்க்கரை அளவு (Glucose)",
        "chol": "கொலஸ்ட்ரால் (Cholesterol)",
        "run_btn": "🚀 முழுமையான பரிசோதனையைத் தொடங்கு",
        "result_detected": "தீராத நோய் கண்டறியப்பட்டது",
        "result_safe": "நோய் பாதிப்பு இல்லை",
        "sim_title": "🏁 உடல்நிலை முன்னேற்ற சிமுலேட்டர்",
        "target": "இலக்கு",
        "download": "📥 மருத்துவ அறிக்கையைப் பதிவிறக்கம் செய் (PDF)",
        "report_title": "மருத்துவப் பரிசோதனை அறிக்கை"
    }
}

# ==========================================
# 2. PDF FUNCTION (With Language Support)
# ==========================================
def create_pdf(name, age, gender, result, prob, recs, medical_data, lang_choice):
    pdf = FPDF()
    pdf.add_page()
    l = lang_data[lang_choice]
    
    pdf.set_fill_color(44, 62, 80)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 20, txt=l["report_title"], ln=True, align='C', fill=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 10, txt=f"Patient: {name} | Age: {age} | Gender: {gender}", ln=True, border='B')
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 10, txt=f"Analysis: {result} ({prob:.1f}%)", ln=True)
    pdf.ln(5)

    pdf.cell(0, 10, txt=" CLINICAL VALUES", ln=True, fill=True)
    pdf.set_font("Arial", size=10)
    for k, v in medical_data.items():
        pdf.cell(95, 8, txt=f" {k}", border=1)
        pdf.cell(95, 8, txt=f" {v}", border=1, ln=True)
    
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 3. MAIN APP SETUP
# ==========================================
st.set_page_config(page_title="Health AI Pro", layout="wide")
pipeline = joblib.load('full_pipeline_compressed.sav')

# --- Language Selection ---
sel_lang = st.sidebar.selectbox("🌐 Choose Language / மொழியைத் தேர்ந்தெடுக்கவும்", ["English", "Tamil"])
L = lang_data[sel_lang]

st.title(L["title"])
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.subheader(L["profile"])
    patient_name = st.text_input(L["name"], "Palanisamy")
    age = st.number_input(L["age"], 1, 120, 45)
    gender = st.selectbox(L["gender"], ["Male", "Female", "Other"])
    bmi = st.number_input("BMI (kg/m2)", 10.0, 50.0, 25.0)
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    activity = st.slider("Activity (Hrs/Week)", 0.0, 10.0, 3.0)

with col2:
    st.subheader(L["vitals"])
    bp = st.number_input(L["bp"], 80, 200, 120)
    cholesterol = st.number_input(L["chol"], 100, 400, 200)
    glucose = st.number_input(L["glucose"], 50, 300, 100)
    stress = st.slider("Stress (1-10)", 1, 10, 5)
    diet = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
    family_hist = st.selectbox("Family History", ["No", "Yes"])

features = ['Age', 'Gender', 'BMI', 'Smoking', 'AlcoholIntake', 'PhysicalActivity', 'DietQuality', 'SleepHours', 'BloodPressure', 'Cholesterol', 'Glucose', 'FamilyHistory', 'StressLevel']
input_df = pd.DataFrame([[age, gender, bmi, smoking, "Low", activity, diet, 7.0, bp, cholesterol, glucose, family_hist, stress]], columns=features)

if st.button(L["run_btn"]):
    prob = pipeline.predict_proba(input_df)[0][1] * 100
    res_text = L["result_detected"] if prob > 50 else L["result_safe"]
    
    if prob > 50: st.error(f"### {res_text} ({prob:.1f}%)")
    else: st.success(f"### {res_text} ({prob:.1f}%)")

    # Metrics Display
    m1, m2, m3 = st.columns(3)
    m1.metric(L["bp"], f"{bp} mmHg", delta=f"{bp-120}", delta_color="inverse")
    m2.metric(L["glucose"], f"{glucose} mg/dL", delta=f"{glucose-100}", delta_color="inverse")
    m3.metric(L["chol"], f"{cholesterol} mg/dL", delta=f"{cholesterol-200}", delta_color="inverse")

    # Simplified Simulator
    st.markdown("---")
    st.subheader(L["sim_title"])
    t_bp = st.slider(f"{L['target']} {L['bp']}", 80, 200, int(bp))
    sim_df = input_df.copy(); sim_df['BloodPressure'] = t_bp
    s_prob = pipeline.predict_proba(sim_df)[0][1] * 100
    st.metric("Simulated Risk", f"{s_prob:.1f}%", delta=f"{s_prob-prob:.1f}%", delta_color="inverse")

    # PDF Report
    med_summary = {L["bp"]: f"{bp} mmHg", L["glucose"]: f"{glucose} mg/dL"}
    pdf_out = create_pdf(patient_name, age, gender, res_text, prob, [], med_summary, sel_lang)
    st.download_button(L["download"], pdf_out, f"Report_{patient_name}.pdf")
