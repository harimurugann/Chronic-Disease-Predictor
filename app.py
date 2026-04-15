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
def create_pdf(name, age, gender, result, prob, recs, medical_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(44, 62, 80)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 20, txt="CHRONIC DISEASE ASSESSMENT REPORT", ln=True, align='C', fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=f"Patient: {name} | Age: {age} | Gender: {gender}", ln=True, border='B')
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=" CLINICAL MEASUREMENTS (RECORDED POINTS)", ln=True, fill=True)
    pdf.set_font("Arial", size=10)
    for key, value in medical_data.items():
        pdf.cell(95, 8, f" {key}", border=1)
        pdf.cell(95, 8, f" {value}", border=1, ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=f"Final Diagnosis: {result} (Risk Score: {prob:.1f}%)", ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=" DOCTOR'S CLINICAL ADVICE", ln=True, fill=True)
    pdf.set_font("Arial", size=10)
    for r in recs:
        pdf.multi_cell(0, 8, txt=r)
    
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 2. MAIN APP SETUP
# ==========================================
st.set_page_config(page_title="Professional AI Health", layout="wide")

try:
    pipeline = joblib.load('full_pipeline_compressed.sav')
except:
    st.error("Error: Model file 'full_pipeline_compressed.sav' not found!")

st.title("🏥 Clinical Health Dashboard & AI Analytics")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.subheader("👤 Patient Profile")
    patient_name = st.text_input("Patient Full Name", "Palanisamy")
    age = st.number_input("Age", 1, 120, 45)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    bmi = st.number_input("BMI (kg/m2)", 10.0, 50.0, 25.0)
    smoking = st.selectbox("Smoking Status", ["No", "Yes"])
    activity = st.slider("Activity (Hrs/Week)", 0.0, 10.0, 2.20)

with col2:
    st.subheader("🩸 Clinical Vitals")
    bp = st.number_input("Systolic BP (mmHg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 400, 200)
    glucose = st.number_input("Glucose Level (mg/dL)", 50, 300, 100)
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)
    diet = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
    family_hist = st.selectbox("Family History", ["No", "Yes"])

# Data Preparation
features = ['Age', 'Gender', 'BMI', 'Smoking', 'AlcoholIntake', 'PhysicalActivity', 'DietQuality', 'SleepHours', 'BloodPressure', 'Cholesterol', 'Glucose', 'FamilyHistory', 'StressLevel']
input_df = pd.DataFrame([[age, gender, bmi, smoking, "Low", activity, diet, 7.0, bp, cholesterol, glucose, family_hist, stress]], columns=features)

clinical_summary = {
    "Blood Pressure": f"{bp} mmHg",
    "Glucose Level": f"{glucose} mg/dL",
    "Total Cholesterol": f"{cholesterol} mg/dL",
    "BMI": f"{bmi:.1f}",
    "Physical Activity": f"{activity} Hrs/Wk",
    "Stress Assessment": f"{stress}/10"
}

# ==========================================
# 3. ANALYSIS & VISUALIZATION
# ==========================================
if st.button("🚀 Run Comprehensive Clinical Analysis"):
    # A. PREDICTION
    prob = pipeline.predict_proba(input_df)[0][1] * 100
    res_text = "Chronic Disease Detected" if prob > 50 else "No Chronic Disease"

    st.markdown("---")
    # Display Result
    if prob > 50:
        st.error(f"### Diagnosis: {res_text} (AI Risk Score: {prob:.1f}%)")
    else:
        st.success(f"### Diagnosis: {res_text} (AI Risk Score: {prob:.1f}%)")

    # B. CLINICAL METRIC CARDS (The New Feature)
    st.subheader("📊 Recorded Clinical Vitals (Points vs Target)")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    with m_col1:
        # Healthy BP Target: 120
        st.metric(label="Systolic BP", value=f"{bp} mmHg", delta=f"{bp-120} from target", delta_color="inverse")
    with m_col2:
        # Healthy Glucose Target: 100
        st.metric(label="Glucose Level", value=f"{glucose} mg/dL", delta=f"{glucose-100} from target", delta_color="inverse")
    with m_col3:
        # Healthy Cholesterol Target: 200
        st.metric(label="Total Cholesterol", value=f"{cholesterol} mg/dL", delta=f"{cholesterol-200} from target", delta_color="inverse")
    with m_col4:
        # Healthy BMI Target: 22
        st.metric(label="Patient BMI", value=f"{bmi:.1f}", delta=f"{bmi-22.0:.1f} from target", delta_color="inverse")

    # C. IMPROVEMENT SIMULATOR
    st.markdown("---")
    st.subheader("🏁 Full Clinical Improvement Simulator")
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        target_bp = st.slider("Target Systolic BP", 80, 200, int(bp), key="sbp")
        target_glucose = st.slider("Target Glucose Level", 50, 300, int(glucose), key="glu")
    with sim_col2:
        target_chol = st.slider("Target Cholesterol", 100, 400, int(cholesterol), key="cho")
        target_bmi = st.slider("Target BMI", 10.0, 50.0, float(bmi), key="bmi_sim")

    # Simulation Calc
    sim_df = input_df.copy()
    sim_df['BloodPressure'] = target_bp
    sim_df['Glucose'] = target_glucose
    sim_df['Cholesterol'] = target_chol
    sim_df['BMI'] = target_bmi
    sim_prob = pipeline.predict_proba(sim_df)[0][1] * 100
    
    st.metric(label="Simulated Risk Score", value=f"{sim_prob:.1f}%", delta=f"{sim_prob-prob:.1f}%", delta_color="inverse")

    # D. DIAGNOSTIC DEEP-DIVE (SHAP)
    st.markdown("---")
    st.subheader("🔬 AI Diagnostic Deep-Dive")
    an_col1, an_col2 = st.columns(2)

    with an_col1:
        st.markdown("**Feature Impact Analysis (SHAP)**")
        try:
            model = pipeline.named_steps['classifier']
            preprocessor = pipeline.named_steps['preprocessor']
            X_trans = preprocessor.transform(input_df)
            f_names = preprocessor.get_feature_names_out()
            explainer = shap.TreeExplainer(model)
            shap_v = explainer.shap_values(X_trans)
            impact_data = shap_v[1][0] if isinstance(shap_v, list) else (shap_v[0,:,1] if len(shap_v.shape)==3 else shap_v[0])
            
            shap_df = pd.DataFrame({'Factor': f_names, 'Impact': np.array(impact_data).flatten()})
            shap_df = shap_df[shap_df['Impact'] > 0].sort_values(by='Impact', ascending=False).head(5)
            
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            sns.barplot(x='Impact', y='Factor', data=shap_df, palette='Reds_r', ax=ax1)
            st.pyplot(fig1)
            plt.close(fig1)
        except: st.info("Analyzing AI risk factors...")

    with an_col2:
        st.markdown("**Recommendations & Advice**")
        recs = []
        if bp > 130: recs.append("-> BP is high. Target: 120 mmHg.")
        if glucose > 140: recs.append("-> Glucose is high. Target: 100 mg/dL.")
        if smoking == "Yes": recs.append("-> Critical: Immediate smoking cessation advised.")
        for r in recs: st.write(r)
        
        pdf_bytes = create_pdf(patient_name, age, gender, res_text, prob, recs, clinical_summary)
        st.download_button("📥 Download Clinical PDF Report", pdf_bytes, f"Report_{patient_name}.pdf")
