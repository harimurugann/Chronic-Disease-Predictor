import streamlit as st
import pandas as pd
import joblib
import numpy as np
from fpdf import FPDF
import base64

# ==========================================
# 1. ROBUST PDF GENERATION LOGIC
# ==========================================
def create_pdf(name, age, gender, result, prob, medical_data, recommendations, lang):
    try:
        pdf = FPDF()
        pdf.add_page()
        
        # Professional Header
        pdf.set_fill_color(44, 62, 80)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", 'B', 18)
        pdf.cell(0, 15, txt="MEDICAL ASSESSMENT REPORT", ln=True, align='C', fill=True)
        
        # Patient Info
        pdf.set_text_color(0, 0, 0)
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, txt=f"Patient: {name if name else 'N/A'} | Age: {age} | Gender: {gender}", ln=True)
        
        # Clinical Vitals Table
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 10)
        for key, value in medical_data.items():
            pdf.cell(60, 8, f" {key}", border=1)
            pdf.set_font("Arial", size=10)
            pdf.cell(100, 8, f" {value}", border=1, ln=True)
            pdf.set_font("Arial", 'B', 10)

        # Assessment Result
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, txt=f"Diagnosis Result: {result} ({prob:.1f}%)", ln=True)

        # Clinical Advice Section (ASCII Only for PDF Safety)
        if recommendations:
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 12)
            advice_title = "CLINICAL ADVICE & NEXT STEPS:" if lang == "English" else "MARUTHUVA ALOSANAI (ADVICE):"
            pdf.cell(0, 10, txt=advice_title, ln=True)
            
            pdf.set_font("Arial", size=10)
            for r_pair in recommendations:
                # r_pair[1] has the clean English/Phonetic text
                clean_text = r_pair[1].encode('ascii', 'ignore').decode('ascii')
                pdf.multi_cell(0, 8, txt=f"- {clean_text}")
            
        # Binary Output Handling (Fixed AttributeError)
        pdf_bytes = pdf.output()
        if isinstance(pdf_bytes, str):
            return pdf_bytes.encode('latin-1', 'ignore')
        return pdf_bytes
        
    except Exception as e:
        return b""

# ==========================================
# 2. DYNAMIC RECOMMENDATIONS LOGIC
# ==========================================
def get_recommendations(bp, glu, bmi, smoking, lang):
    recs = []
    if lang == "Tamil":
        if bp > 140: recs.append(("✅ இரத்த அழுத்தம் அதிகம்: உப்பைக் குறைக்கவும்.", "High BP: Reduce salt intake."))
        if glu > 150: recs.append(("✅ சர்க்கரை அளவு அதிகம்: இனிப்பைத் தவிர்க்கவும்.", "High Glucose: Avoid sugary foods."))
        if bmi > 25: recs.append(("✅ எடை அதிகம்: நடைப்பயிற்சி செய்யவும்.", "High BMI: Daily walk recommended."))
        if smoking == "Yes": recs.append(("✅ புகைப்பிடித்தலைத் தவிர்க்கவும்.", "Quit smoking for heart health."))
    else:
        if bp > 140: recs.append(("✅ BP is High: Reduce salt intake.", "BP is High: Reduce salt intake."))
        if glu > 150: recs.append(("✅ Glucose is High: Avoid sugar.", "Glucose is High: Avoid sugar."))
        if bmi > 25: recs.append(("✅ Weight Management: Daily exercise.", "Weight Management: Daily exercise."))
    return recs

# ==========================================
# 3. MAIN STREAMLIT INTERFACE
# ==========================================
st.set_page_config(page_title="Health AI Pro", layout="wide")

# Sidebar
st.sidebar.title("⚙️ Control Panel")
sel_lang = st.sidebar.selectbox("Select Language / மொழி", ["English", "Tamil"])

# Load Machine Learning Model
try:
    pipeline = joblib.load('full_pipeline_compressed.sav')
except:
    st.error("Model file 'full_pipeline_compressed.sav' not found!")

st.title("🏥 Professional Health Diagnostic AI")
st.markdown("---")

# Layout Columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Patient Profile")
    p_name = st.text_input("Full Name", placeholder="Enter Patient Name")
    age = st.number_input("Age", 1, 120, 30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    bmi = st.number_input("BMI (Body Mass Index)", 10.0, 50.0, 24.5)
    smoking = st.selectbox("Smoking Status", ["No", "Yes"])
    activity = st.slider("Exercise (Hours/Week)", 0.0, 14.0, 3.5)

with col2:
    st.subheader("🩸 Clinical Vitals")
    bp = st.number_input("Blood Pressure (mmHg)", 80, 200, 120)
    glu = st.number_input("Glucose Level (mg/dL)", 50, 300, 100)
    cho = st.number_input("Cholesterol", 100, 400, 200)
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)
    diet = st.selectbox("Diet Quality", ["Good", "Average", "Poor"])
    family = st.selectbox("Family History", ["No", "Yes"])

# ==========================================
# 4. ANALYSIS & FAIL-SAFE REPORTING
# ==========================================
if st.button("🚀 RUN FULL DIAGNOSTIC ANALYSIS"):
    # Data Preparation
    features = ['Age', 'Gender', 'BMI', 'Smoking', 'AlcoholIntake', 'PhysicalActivity', 'DietQuality', 'SleepHours', 'BloodPressure', 'Cholesterol', 'Glucose', 'FamilyHistory', 'StressLevel']
    input_df = pd.DataFrame([[age, gender, bmi, smoking, "Low", activity, diet, 7.0, bp, cho, glu, family, stress]], columns=features)
    
    # Model Prediction
    prob = pipeline.predict_proba(input_df)[0][1] * 100
    res_text = "Risk Detected" if prob > 50 else "Healthy Range"

    # UI Feedback
    st.markdown("---")
    if prob > 50:
        st.error(f"### AI Diagnosis: {res_text} ({prob:.1f}%)")
    else:
        st.success(f"### AI Diagnosis: {res_text} ({prob:.1f}%)")

    # Recommendations (Dashboard & PDF data)
    patient_recs = get_recommendations(bp, glu, bmi, smoking, sel_lang)
    
    st.subheader("📋 Personalized Recommendations")
    for r in patient_recs:
        st.write(r[0]) # Displays Tamil/English with Emojis on Dashboard

    # --- REPORT CENTER (Base64 Download Fix) ---
    st.markdown("---")
    st.subheader("📊 Secure Report Center")
    
    summary = {"BP": f"{bp} mmHg", "Glucose": f"{glu} mg/dL", "BMI": f"{bmi:.1f}"}
    
    # Pre-generate PDF Bytes
    final_pdf_bytes = create_pdf(p_name, age, gender, res_text, prob, summary, patient_recs, sel_lang)
    
    if final_pdf_bytes and len(final_pdf_bytes) > 0:
        # Generate the Base64 Link (Bypassing Streamlit API Errors)
        b64_data = base64.b64encode(final_pdf_bytes).decode()
        
        # Professional HTML Button Design
        download_btn_html = f'''
        <div style="text-align: center; margin: 30px 0;">
            <a href="data:application/octet-stream;base64,{b64_data}" 
               download="Medical_Report_{p_name if p_name else 'Patient'}.pdf" 
               style="background-color: #007bff; color: white; padding: 18px 40px; text-decoration: none; 
               border-radius: 12px; font-weight: bold; font-size: 20px; display: inline-block; box-shadow: 0px 4px 10px rgba(0,0,0,0.2);">
                📥 DOWNLOAD MEDICAL ASSESSMENT (PDF)
            </a>
        </div>
        '''
        st.markdown(download_btn_html, unsafe_allow_html=True)
        st.success("✅ Assessment complete. Click the button above to secure your report.")
    else:
        st.error("⚠️ System could not generate the PDF. Please refresh and try again.")
