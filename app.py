import streamlit as st
import pandas as pd
import joblib
import numpy as np
from fpdf import FPDF

# ==========================================
# 1. LANGUAGE & RECOMMENDATION DATA
# ==========================================
lang_data = {
    "English": {
        "title": "🏥 Advanced AI Health Diagnostic Pro",
        "profile": "👤 Patient Profile",
        "vitals": "🩸 Clinical Metrics",
        "run_btn": "🚀 Run Full Diagnostic Analysis",
        "score_title": "🏆 Lifestyle Health Score",
        "bmi_title": "⚖️ BMI Classification",
        "advice_title": "🤖 AI Clinical Assistant",
        "rec_title": "📋 Clinical Recommendations",
        "download_btn": "📥 Download Clinical Report (PDF)",
        "status": "Current Status"
    },
    "Tamil": {
        "title": "🏥 உயர்தர AI மருத்துவப் பரிசோதனை மையம்",
        "profile": "👤 நோயாளியின் விவரங்கள்",
        "vitals": "🩸 மருத்துவ அளவீடுகள்",
        "run_btn": "🚀 முழுமையான பரிசோதனையைத் தொடங்கு",
        "score_title": "🏆 வாழ்க்கைமுறை ஆரோக்கிய மதிப்பெண்",
        "bmi_title": "⚖️ உடல் எடை குறியீடு (BMI)",
        "advice_title": "🤖 AI மருத்துவ உதவியாளர்",
        "rec_title": "📋 மருத்துவ பரிந்துரைகள்",
        "download_btn": "📥 மருத்துவ அறிக்கையைப் பதிவிறக்கம் செய்",
        "status": "தற்போதைய நிலை"
    }
}

def get_recommendations(bp, glu, bmi, smoking, lang):
    recs = []
    if lang == "English":
        if bp > 140: recs.append("⚠️ High BP: Reduce salt intake and consult a doctor.")
        if glu > 150: recs.append("⚠️ High Glucose: Avoid sugary foods and monitor levels.")
        if bmi > 25: recs.append("🏃 BMI High: Focus on daily 30-min brisk walking.")
        if smoking == "Yes": recs.append("🚭 Smoking: Quit smoking to lower cardiac risk.")
        if not recs: recs.append("✅ Maintain your healthy lifestyle and balanced diet.")
    else:
        if bp > 140: recs.append("⚠️ ரத்த அழுத்தம் அதிகம்: உப்பைக் குறைத்து மருத்துவரை அணுகவும்.")
        if glu > 150: recs.append("⚠️ சர்க்கரை அளவு அதிகம்: இனிப்பு உணவுகளைத் தவிர்க்கவும்.")
        if bmi > 25: recs.append("🏃 எடை அதிகம்: தினமும் 30 நிமிடம் நடைப்பயிற்சி செய்யவும்.")
        if smoking == "Yes": recs.append("🚭 புகைப்பிடித்தலைத் தவிர்க்கவும்: இது இதயத்திற்கு நல்லது.")
        if not recs: recs.append("✅ ஆரோக்கியமான உணவு மற்றும் உடற்பயிற்சியைத் தொடரவும்.")
    return recs

# ==========================================
# 2. ROBUST PDF FUNCTION
# ==========================================
def create_pdf(name, age, gender, result, prob, medical_data, recommendations):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(44, 62, 80)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 20, txt="MEDICAL ASSESSMENT REPORT", ln=True, align='C', fill=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=f"Patient: {name if name else 'N/A'} | Age: {age} | Gender: {gender}", ln=True, border='B')
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=" CLINICAL MEASUREMENTS", ln=True, fill=True)
    pdf.set_font("Arial", size=10)
    for key, value in medical_data.items():
        pdf.cell(95, 8, f" {key}", border=1)
        pdf.cell(95, 8, f" {value}", border=1, ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="DOCTOR'S CLINICAL ADVICE:", ln=True)
    pdf.set_font("Arial", size=10)
    # Note: PDF currently supports English text for recommendations to avoid font issues
    for r in recommendations:
        pdf.multi_cell(0, 8, txt=f"- {r}")
        
    pdf_output = pdf.output(dest='S')
    return pdf_output.encode('latin-1') if isinstance(pdf_output, str) else pdf_output

# ==========================================
# 3. APP UI SETUP
# ==========================================
st.set_page_config(page_title="Health AI Pro", layout="wide")

st.sidebar.title("⚙️ Settings")
sel_lang = st.sidebar.selectbox("Language / மொழி", ["English", "Tamil"])
L = lang_data[sel_lang]

try:
    pipeline = joblib.load('full_pipeline_compressed.sav')
except:
    st.error("⚠️ Model file not found!")

st.title(L["title"])
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.subheader(L["profile"])
    p_name = st.text_input("Full Name", value="", placeholder="Enter Patient Name")
    age = st.number_input("Age", 1, 120, 30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    bmi = st.number_input("BMI (kg/m2)", 10.0, 50.0, 24.5)
    smoking = st.selectbox("Smoking Status", ["No", "Yes"])
    activity = st.slider("Exercise (Hrs/Week)", 0.0, 14.0, 3.5)

with col2:
    st.subheader(L["vitals"])
    bp = st.number_input("Systolic BP (mmHg)", 80, 200, 120)
    glu = st.number_input("Glucose (mg/dL)", 50, 300, 100)
    cho = st.number_input("Cholesterol (mg/dL)", 100, 400, 200)
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)
    diet = st.selectbox("Diet Quality", ["Good", "Average", "Poor"])
    family = st.selectbox("Family History", ["No", "Yes"])

# ==========================================
# 4. ANALYSIS LOGIC
# ==========================================
if st.button(L["run_btn"]):
    features = ['Age', 'Gender', 'BMI', 'Smoking', 'AlcoholIntake', 'PhysicalActivity', 'DietQuality', 'SleepHours', 'BloodPressure', 'Cholesterol', 'Glucose', 'FamilyHistory', 'StressLevel']
    input_df = pd.DataFrame([[age, gender, bmi, smoking, "Low", activity, diet, 7.0, bp, cho, glu, family, stress]], columns=features)
    
    prob = pipeline.predict_proba(input_df)[0][1] * 100
    res_text = "Risk Detected" if prob > 50 else "Healthy Range"

    if prob > 50: st.error(f"### {L['status']}: {res_text} ({prob:.1f}%)")
    else: st.success(f"### {L['status']}: {res_text} ({prob:.1f}%)")

    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("BP", f"{bp}", f"{bp-120}", delta_color="inverse")
    m2.metric("Glucose", f"{glu}", f"{glu-100}", delta_color="inverse")
    m3.metric("BMI", f"{bmi:.1f}", f"{bmi-22.5:.1f}", delta_color="inverse")

    # Feature: Lifestyle Score & Recommendations
    st.markdown("---")
    mid_col1, mid_col2 = st.columns(2)
    with mid_col1:
        st.subheader(L["score_title"])
        score = 100
        if activity < 3: score -= 20
        if smoking == "Yes": score -= 20
        if stress > 7: score -= 15
        st.write(f"### Score: {score}/100")
        st.progress(score / 100)
    
    with mid_col2:
        st.subheader(L["rec_title"])
        patient_recs = get_recommendations(bp, glu, bmi, smoking, sel_lang)
        for r in patient_recs:
            st.info(r)

    # Download Section
    st.markdown("---")
    summary = {"BP": f"{bp} mmHg", "Glucose": f"{glu} mg/dL", "BMI": f"{bmi:.1f}", "Score": f"{score}/100"}
    try:
        # Create PDF with English recommendations for consistency
        pdf_recs = get_recommendations(bp, glu, bmi, smoking, "English")
        pdf_data = create_pdf(p_name, age, gender, res_text, prob, summary, pdf_recs)
        
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            st.download_button(label=L["download_btn"], data=pdf_data, file_name=f"Report_{p_name}.pdf", mime="application/pdf", use_container_width=True)
    except Exception as e:
        st.warning(f"PDF rendering paused: {e}")
