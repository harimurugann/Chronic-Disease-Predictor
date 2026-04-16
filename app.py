import streamlit as st
import pandas as pd
import joblib
import numpy as np
from fpdf import FPDF

# ==========================================
# 1. LANGUAGE DATA
# ==========================================
lang_data = {
    "English": {
        "title": "🏥 Advanced AI Health Diagnostic Pro",
        "profile": "👤 Patient Profile",
        "vitals": "🩸 Clinical Metrics",
        "run_btn": "🚀 Run Full Diagnostic Analysis",
        "advice_title": "📋 Clinical Recommendations",
        "download_btn": "📥 Download Clinical Report (PDF)",
    },
    "Tamil": {
        "title": "🏥 உயர்தர AI மருத்துவப் பரிசோதனை மையம்",
        "profile": "👤 நோயாளியின் விவரங்கள்",
        "vitals": "🩸 மருத்துவ அளவீடுகள்",
        "run_btn": "🚀 முழுமையான பரிசோதனையைத் தொடங்கு",
        "advice_title": "📋 மருத்துவ பரிந்துரைகள்",
        "download_btn": "📥 மருத்துவ அறிக்கையைப் பதிவிறக்கம் செய்",
    }
}

# ==========================================
# 2. RECOMMENDATION LOGIC (Dual-Text for PDF Safety)
# ==========================================
def get_recommendations(bp, glu, bmi, smoking, lang):
    recs = []
    if lang == "English":
        if bp > 140: recs.append(("⚠️ BP is High: Reduce salt intake.", "BP is High: Reduce salt intake."))
        if glu > 150: recs.append(("⚠️ Glucose is High: Avoid sugary foods.", "Glucose is High: Avoid sugary foods."))
        if bmi > 25: recs.append(("🏃 BMI High: Daily 30-min brisk walk recommended.", "BMI High: Daily 30-min brisk walk recommended."))
        if smoking == "Yes": recs.append(("🚭 Smoking: Quit smoking to lower cardiac risk.", "Smoking: Quit smoking to lower cardiac risk."))
    else:
        if bp > 140: recs.append(("⚠️ இரத்த அழுத்தம் அதிகம்: உப்பைக் குறைக்கவும்.", "BP is High: Reduce salt intake."))
        if glu > 150: recs.append(("⚠️ சர்க்கரை அளவு அதிகம்: இனிப்பைத் தவிர்க்கவும்.", "Glucose is High: Avoid sugary foods."))
        if bmi > 25: recs.append(("🏃 எடை அதிகம்: தினமும் நடைப்பயிற்சி செய்யவும்.", "BMI High: Daily 30-min walk recommended."))
        if smoking == "Yes": recs.append(("🚭 புகைப்பிடித்தலைத் தவிர்க்கவும்.", "Smoking: Quit smoking recommended."))
    return recs

# ==========================================
# 3. FIXED PDF FUNCTION
# ==========================================
def create_pdf(name, age, gender, result, prob, medical_data, recommendations):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(44, 62, 80)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 15, txt="MEDICAL ASSESSMENT REPORT", ln=True, align='C', fill=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, txt=f"Patient: {name if name else 'N/A'} | Age: {age} | Gender: {gender}", ln=True)
    pdf.ln(5)

    # Vitals Table
    pdf.cell(0, 8, txt="CLINICAL VITALS:", ln=True)
    for key, value in medical_data.items():
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(60, 8, f" {key}", border=1)
        pdf.set_font("Arial", size=10)
        pdf.cell(100, 8, f" {value}", border=1, ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=f"AI Risk Assessment: {result} ({prob:.1f}%)", ln=True)

    # Safe Recommendations (Clean Text Only)
    if recommendations:
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, txt="CLINICAL ADVICE & NEXT STEPS:", ln=True)
        pdf.set_font("Arial", size=10)
        for r_pair in recommendations:
            clean_text = r_pair[1].encode('ascii', 'ignore').decode('ascii')
            pdf.multi_cell(0, 7, txt=f"- {clean_text}")

    pdf_out = pdf.output(dest='S')
    if isinstance(pdf_out, str):
        return pdf_out.encode('latin-1', 'ignore')
    return pdf_out

# ==========================================
# 4. MAIN APP UI
# ==========================================
st.set_page_config(page_title="Health AI Pro", layout="wide")

st.sidebar.title("⚙️ Settings")
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
    p_name = st.text_input("Name", value="", placeholder="Enter Name")
    age = st.number_input("Age", 1, 120, 30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    bmi = st.number_input("BMI (kg/m2)", 10.0, 50.0, 24.5)
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    activity = st.slider("Activity (Hrs/Week)", 0.0, 14.0, 3.5)

with col2:
    st.subheader(L["vitals"])
    bp = st.number_input("Systolic BP", 80, 200, 120)
    glu = st.number_input("Glucose", 50, 300, 100)
    cho = st.number_input("Cholesterol", 100, 400, 200)
    stress = st.slider("Stress (1-10)", 1, 10, 5)
    diet = st.selectbox("Diet Quality", ["Good", "Average", "Poor"])
    family = st.selectbox("Family History", ["No", "Yes"])

if st.button(L["run_btn"]):
    # Prediction
    features = ['Age', 'Gender', 'BMI', 'Smoking', 'AlcoholIntake', 'PhysicalActivity', 'DietQuality', 'SleepHours', 'BloodPressure', 'Cholesterol', 'Glucose', 'FamilyHistory', 'StressLevel']
    input_df = pd.DataFrame([[age, gender, bmi, smoking, "Low", activity, diet, 7.0, bp, cho, glu, family, stress]], columns=features)
    prob = pipeline.predict_proba(input_df)[0][1] * 100
    res_text = "Risk Detected" if prob > 50 else "Healthy Range"

    # Metrics
    st.markdown(f"### AI Diagnosis: {res_text} ({prob:.1f}%)")
    m1, m2, m3 = st.columns(3)
    m1.metric("BP", f"{bp}", f"{bp-120}", delta_color="inverse")
    m2.metric("Glucose", f"{glu}", f"{glu-100}", delta_color="inverse")
    m3.metric("BMI", f"{bmi:.1f}", f"{bmi-22.5:.1f}", delta_color="inverse")

    # Recommendations (Dashboard)
    patient_recs = get_recommendations(bp, glu, bmi, smoking, sel_lang)
    st.markdown("---")
    st.subheader(L["advice_title"])
    for r_pair in patient_recs:
        st.write(r_pair[0])

    # PDF Download
    st.markdown("---")
    summary = {"BP": f"{bp} mmHg", "Glucose": f"{glu} mg/dL", "BMI": f"{bmi:.1f}"}
    try:
        pdf_bytes = create_pdf(p_name, age, gender, res_text, prob, summary, patient_recs)
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            st.download_button(L["download_btn"], data=pdf_bytes, file_name=f"Report_{p_name}.pdf", mime="application/pdf", use_container_width=True)
    except Exception as e:
        st.warning(f"PDF Rendering Note: {e}")
