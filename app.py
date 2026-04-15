import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
from fpdf import FPDF

# ==========================================
# 1. PROFESSIONAL PDF FUNCTION (Robust Version)
# ==========================================
def create_pdf(name, age, gender, result, prob, recs, medical_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(44, 62, 80)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 20, txt="HEALTH ASSESSMENT REPORT", ln=True, align='C', fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 10, txt=f"Patient: {name} | Age: {age} | Gender: {gender}", ln=True, border='B')
    pdf.ln(5)
    pdf.cell(0, 10, txt=f"Analysis Result: {result} ({prob:.1f}%)", ln=True)
    pdf.ln(5)

    pdf.cell(0, 10, txt=" CLINICAL VALUES", ln=True, fill=True)
    pdf.set_font("Arial", size=10)
    for k, v in medical_data.items():
        pdf.cell(95, 8, txt=f" {k}", border=1)
        pdf.cell(95, 8, txt=f" {v}", border=1, ln=True)
    
    return pdf.output()

# ==========================================
# 2. MAIN APP SETUP
# ==========================================
st.set_page_config(page_title="Professional AI Health", layout="wide")

try:
    pipeline = joblib.load('full_pipeline_compressed.sav')
except:
    st.error("Error: Model file not found!")

st.title("🏥 Professional Health Prediction & Analytics")
st.markdown("---")

# Language Selection in Sidebar
sel_lang = st.sidebar.selectbox("🌐 Language / மொழி", ["English", "Tamil"])

col1, col2 = st.columns(2)
with col1:
    st.subheader("👤 Profile")
    patient_name = st.text_input("Name", "Palanisamy")
    age = st.number_input("Age", 1, 120, 45)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    bmi = st.number_input("BMI (kg/m2)", 10.0, 50.0, 25.0)
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    activity = st.slider("Activity (Hrs/Week)", 0.0, 10.0, 2.20)

with col2:
    st.subheader("🩸 Vitals")
    bp = st.number_input("Systolic BP (mmHg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 400, 200)
    glucose = st.number_input("Glucose Level (mg/dL)", 50, 300, 100)
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)
    diet = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
    family_hist = st.selectbox("Family History", ["No", "Yes"])

# Prepare Input Data
features = ['Age', 'Gender', 'BMI', 'Smoking', 'AlcoholIntake', 'PhysicalActivity', 'DietQuality', 'SleepHours', 'BloodPressure', 'Cholesterol', 'Glucose', 'FamilyHistory', 'StressLevel']
input_df = pd.DataFrame([[age, gender, bmi, smoking, "Low", activity, diet, 7.0, bp, cholesterol, glucose, family_hist, stress]], columns=features)

clinical_summary = {"BP": f"{bp} mmHg", "Glucose": f"{glucose} mg/dL", "Cholesterol": f"{cholesterol} mg/dL", "BMI": f"{bmi}"}

# ==========================================
# 3. BUTTON TRIGGER & LOGIC
# ==========================================
if st.button("🚀 Generate Full Analysis"):
    # A. Prediction
    prob = pipeline.predict_proba(input_df)[0][1] * 100
    res_text = "Chronic Disease Detected" if prob > 50 else "No Chronic Disease"
    
    st.markdown("---")
    if prob > 50: st.error(f"### Status: {res_text} ({prob:.1f}%)")
    else: st.success(f"### Status: {res_text} ({prob:.1f}%)")

    # B. BMI CLASSIFICATION (Feature 5)
    st.subheader("⚖️ BMI Health Category")
    bmi_cat, bmi_color = ("Normal", "green")
    if bmi < 18.5: bmi_cat, bmi_color = ("Underweight", "blue")
    elif 25 <= bmi < 30: bmi_cat, bmi_color = ("Overweight", "orange")
    elif bmi >= 30: bmi_cat, bmi_color = ("Obese", "red")
    st.markdown(f"Status: :{bmi_color}[**{bmi_cat}**]")

    # C. WHAT-IF SIMULATOR (Fixed ValueError)
    st.markdown("---")
    st.subheader("🏁 Clinical Improvement Simulator")
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        t_bp = st.slider("Target BP", 80, 200, int(bp))
        t_glu = st.slider("Target Glucose", 50, 300, int(glucose))
    with s_col2:
        t_cho = st.slider("Target Cholesterol", 100, 400, int(cholesterol))
        t_bmi = st.slider("Target BMI", 10.0, 50.0, float(bmi))

    # Simulation Logic - No more Scalar error
    sim_df = input_df.copy()
    sim_df.at[0, 'BloodPressure'] = t_bp
    sim_df.at[0, 'Glucose'] = t_glu
    sim_df.at[0, 'Cholesterol'] = t_cho
    sim_df.at[0, 'BMI'] = t_bmi
    
    s_prob = pipeline.predict_proba(sim_df)[0][1] * 100
    st.metric("Simulated Risk Score", f"{s_prob:.1f}%", delta=f"{s_prob-prob:.1f}%", delta_color="inverse")

    # D. ANALYTICS (SHAP)
    st.markdown("---")
    st.subheader("🔬 AI Decision Insights")
    try:
        model = pipeline.named_steps['classifier']
        preprocessor = pipeline.named_steps['preprocessor']
        X_trans = preprocessor.transform(input_df)
        explainer = shap.TreeExplainer(model)
        shap_v = explainer.shap_values(X_trans)
        impact = shap_v[1][0] if isinstance(shap_v, list) else shap_v[0,:,1]
        
        shap_df = pd.DataFrame({'Factor': preprocessor.get_feature_names_out(), 'Impact': np.array(impact).flatten()})
        shap_df = shap_df[shap_df['Impact'] > 0].sort_values(by='Impact', ascending=False).head(5)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x='Impact', y='Factor', data=shap_df, palette='Reds_r', ax=ax)
        st.pyplot(fig)
    except: st.info("Loading AI analysis...")

    # E. PDF DOWNLOAD
    pdf_out = create_pdf(patient_name, age, gender, res_text, prob, [], clinical_summary)
    st.download_button("📥 Download Report", pdf_out, f"Report_{patient_name}.pdf")
