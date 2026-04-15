import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
from fpdf import FPDF

# ==========================================
# 1. LANGUAGE DICTIONARY
# ==========================================
lang_data = {
    "English": {
        "title": "🏥 Professional Health Prediction & Analytics",
        "sidebar": "🌐 Settings",
        "lang_sel": "Choose Language",
        "profile": "👤 Patient Profile",
        "name": "Patient Full Name",
        "placeholder": "Enter Name",
        "age": "Age",
        "vitals": "🩸 Clinical Metrics",
        "bp": "Systolic BP (mmHg)",
        "glucose": "Glucose Level (mg/dL)",
        "run_btn": "🚀 Generate Full Health Analysis",
        "res_likely": "Chronic Disease Likely",
        "res_safe": "No Chronic Disease",
        "sim_title": "🏁 Full Clinical Improvement Simulator",
        "download": "📥 Download Clinical Report (PDF)"
    },
    "Tamil": {
        "title": "🏥 மருத்துவ நலப் பரிசோதனை மற்றும் பகுப்பாய்வு",
        "sidebar": "🌐 அமைப்புகள்",
        "lang_sel": "மொழியைத் தேர்ந்தெடுக்கவும்",
        "profile": "👤 நோயாளியின் விவரங்கள்",
        "name": "நோயாளியின் முழு பெயர்",
        "placeholder": "பெயரை உள்ளிடவும்",
        "age": "வயது",
        "vitals": "🩸 மருத்துவ அளவீடுகள்",
        "bp": "இரத்த அழுத்தம் (BP)",
        "glucose": "சர்க்கரை அளவு (Glucose)",
        "run_btn": "🚀 முழுமையான பரிசோதனையைத் தொடங்கு",
        "res_likely": "நோய் பாதிப்பு இருக்க வாய்ப்புள்ளது",
        "res_safe": "நோய் பாதிப்பு இல்லை",
        "sim_title": "🏁 உடல்நிலை முன்னேற்ற சிமுலேட்டர்",
        "download": "📥 மருத்துவ அறிக்கையைப் பதிவிறக்கம் செய் (PDF)"
    }
}

# ==========================================
# 2. FIXED PDF FUNCTION
# ==========================================
def create_pdf(name, age, gender, result, prob, medical_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(44, 62, 80)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 20, txt="CLINICAL ASSESSMENT REPORT", ln=True, align='C', fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=f"Patient: {name if name else 'N/A'} | Age: {age} | Gender: {gender}", ln=True, border='B')
    pdf.ln(5)
    for key, value in medical_data.items():
        pdf.cell(95, 8, f" {key}", border=1)
        pdf.cell(95, 8, f" {value}", border=1, ln=True)
    pdf_output = pdf.output(dest='S')
    if isinstance(pdf_output, str): return pdf_output.encode('latin-1')
    return pdf_output

# ==========================================
# 3. MAIN APP SETUP
# ==========================================
st.set_page_config(page_title="Professional AI Health", layout="wide")

# Language Selector in Sidebar
st.sidebar.title(lang_data["English"]["sidebar"])
sel_lang = st.sidebar.selectbox("Language / மொழி", ["English", "Tamil"])
L = lang_data[sel_lang]

try:
    pipeline = joblib.load('full_pipeline_compressed.sav')
except:
    st.error("Model file not found!")

st.title(L["title"])
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.subheader(L["profile"])
    patient_name = st.text_input(L["name"], value="", placeholder=L["placeholder"])
    age = st.number_input(L["age"], 1, 120, 30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    bmi = st.number_input("BMI (kg/m2)", 10.0, 50.0, 25.0)
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    activity = st.slider("Activity (Hrs/Week)", 0.0, 10.0, 3.0)

with col2:
    st.subheader(L["vitals"])
    bp = st.number_input(L["bp"], 80, 200, 120)
    cholesterol = st.number_input("Cholesterol", 100, 400, 200)
    glucose = st.number_input(L["glucose"], 50, 300, 100)
    stress = st.slider("Stress (1-10)", 1, 10, 5)
    diet = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
    family_hist = st.selectbox("Family History", ["No", "Yes"])

# ==========================================
# 4. BUTTON TRIGGER & LOGIC
# ==========================================
if st.button(L["run_btn"]):
    features = ['Age', 'Gender', 'BMI', 'Smoking', 'AlcoholIntake', 'PhysicalActivity', 'DietQuality', 'SleepHours', 'BloodPressure', 'Cholesterol', 'Glucose', 'FamilyHistory', 'StressLevel']
    input_df = pd.DataFrame([[age, gender, bmi, smoking, "Low", activity, diet, 7.0, bp, cholesterol, glucose, family_hist, stress]], columns=features)
    
    prob = pipeline.predict_proba(input_df)[0][1] * 100
    res_text = L["res_likely"] if prob > 50 else L["res_safe"]

    if prob > 50: st.error(f"### Status: {res_text} ({prob:.1f}%)")
    else: st.success(f"### Status: {res_text} ({prob:.1f}%)")

    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("BP", f"{bp}", delta=f"{bp-120}", delta_color="inverse")
    m2.metric("Glucose", f"{glucose}", delta=f"{glucose-100}", delta_color="inverse")
    m3.metric("BMI", f"{bmi:.1f}", delta=f"{bmi-22.0:.1f}", delta_color="inverse")

    # Simulator
    st.markdown("---")
    st.subheader(L["sim_title"])
    t_bp = st.slider("Target BP", 80, 200, int(bp))
    sim_df = input_df.copy()
    sim_df.at[0, 'BloodPressure'] = t_bp
    sim_prob = pipeline.predict_proba(sim_df)[0][1] * 100
    st.metric("Simulated Risk", f"{sim_prob:.1f}%", delta=f"{sim_prob-prob:.1f}%", delta_color="inverse")

    # PDF Download
    st.markdown("---")
    summary = {"BP": f"{bp} mmHg", "Glucose": f"{glucose} mg/dL"}
    try:
        pdf_bytes = create_pdf(patient_name, age, gender, res_text, prob, summary)
        st.download_button(label=L["download"], data=pdf_bytes, file_name=f"Report_{patient_name}.pdf", mime="application/pdf")
    except Exception as e:
        st.error(f"PDF Error: {e}")
