import streamlit as st
import pandas as pd
import joblib
import numpy as np
from fpdf import FPDF

# ==========================================
# 1. LANGUAGE & LOGIC
# ==========================================
lang_data = {
    "English": {"title": "🏥 Health AI Pro", "run_btn": "🚀 Run Analysis", "status": "Status"},
    "Tamil": {"title": "🏥 மருத்துவ நல AI", "run_btn": "🚀 பரிசோதனையைத் தொடங்கு", "status": "நிலை"}
}

def get_recommendations(bp, glu, bmi, smoking, lang):
    recs = []
    # Dashboard text (with Tamil/Emoji) | PDF text (Pure English for Safety)
    if lang == "Tamil":
        if bp > 140: recs.append(("✅ இரத்த அழுத்தம் அதிகம்.", "High BP: Reduce salt."))
        if glu > 150: recs.append(("✅ சர்க்கரை அளவு அதிகம்.", "High Glucose: Avoid sugar."))
    else:
        if bp > 140: recs.append(("✅ BP is High.", "BP is High."))
        if glu > 150: recs.append(("✅ Glucose is High.", "Glucose is High."))
    return recs

def create_pdf(name, age, gender, result, prob, medical_data, recommendations):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt="MEDICAL ASSESSMENT REPORT", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, txt=f"Patient: {name} | Age: {age} | Result: {result}", ln=True)
    for r_pair in recommendations:
        clean_text = r_pair[1].encode('ascii', 'ignore').decode('ascii')
        pdf.cell(0, 10, txt=f"- {clean_text}", ln=True)
    return pdf.output(dest='S').encode('latin-1', 'ignore')

# ==========================================
# 2. UI SETUP
# ==========================================
st.set_page_config(page_title="Health AI Pro", layout="wide")
sel_lang = st.sidebar.selectbox("Language / மொழி", ["English", "Tamil"])
L = lang_data[sel_lang]

# Load Model
pipeline = joblib.load('full_pipeline_compressed.sav')

st.title(L["title"])
col1, col2 = st.columns(2)
with col1:
    p_name = st.text_input("Name", value="Patient")
    age = st.number_input("Age", 1, 120, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bmi = st.number_input("BMI", 10.0, 50.0, 24.0)
    smoking = st.selectbox("Smoking", ["No", "Yes"])
with col2:
    bp = st.number_input("BP", 80, 200, 120)
    glu = st.number_input("Glucose", 50, 300, 100)
    cho = st.number_input("Cholesterol", 100, 400, 200)
    activity = st.slider("Activity", 0, 14, 3)
    stress = st.slider("Stress", 1, 10, 5)

# ==========================================
# 3. CRASH-PROOF EXECUTION
# ==========================================
if st.button(L["run_btn"]):
    # Calculation
    input_df = pd.DataFrame([[age, gender, bmi, smoking, "Low", activity, "Good", 7.0, bp, cho, glu, "No", stress]], 
                            columns=['Age', 'Gender', 'BMI', 'Smoking', 'AlcoholIntake', 'PhysicalActivity', 'DietQuality', 'SleepHours', 'BloodPressure', 'Cholesterol', 'Glucose', 'FamilyHistory', 'StressLevel'])
    
    prob = pipeline.predict_proba(input_df)[0][1] * 100
    res_text = "Risk Detected" if prob > 50 else "Healthy Range"
    
    # Dashboard Display
    st.write("---")
    if prob > 50: st.error(f"{L['status']}: {res_text} ({prob:.1f}%)")
    else: st.success(f"{L['status']}: {res_text} ({prob:.1f}%)")
    
    # Recs
    patient_recs = get_recommendations(bp, glu, bmi, smoking, sel_lang)
    for r in patient_recs: st.write(r[0])

    # --- THE ULTIMATE DOWNLOAD BUTTON FIX ---
    st.write("---")
    st.subheader("📊 Report Center")
    
    try:
        summary = {"BP": f"{bp}", "Glucose": f"{glu}"}
        pdf_data = create_pdf(p_name, age, gender, res_text, prob, summary, patient_recs)
        
        # 💡 CRITICAL: Button label is PURE ENGLISH to avoid API Crash
        st.download_button(
            label="DOWNLOAD PDF REPORT", 
            data=pdf_data,
            file_name=f"Report_{p_name}.pdf",
            mime="application/pdf",
            key="FINAL_FIXED_BTN_001", # Change key if button doesn't show
            use_container_width=True
        )
        st.success("Analysis Complete! Download button is active above.")
    except Exception as e:
        st.error(f"Error: {e}")
