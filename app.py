import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
from fpdf import FPDF

# ==========================================
# 1. Professional PDF Function
# ==========================================
def create_pdf(name, age, gender, result, prob, recs, medical_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(44, 62, 80)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 20, txt="CLINICAL HEALTH ASSESSMENT REPORT", ln=True, align='C', fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=f"Patient: {name} | Age: {age} | Gender: {gender}", border='B', ln=True)
    pdf.ln(5)

    pdf.cell(0, 10, txt=" CLINICAL METRICS & DEVIATION", ln=True, fill=True)
    pdf.set_font("Arial", size=10)
    for key, value in medical_data.items():
        pdf.cell(95, 8, f" {key}", border=1)
        pdf.cell(95, 8, f" {value}", border=1, ln=True)
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=" DOCTOR'S RECOMMENDATIONS", ln=True, fill=True)
    pdf.set_font("Arial", size=10)
    for r in recs:
        pdf.multi_cell(0, 8, txt=f" {r}")
    
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 2. Main App Logic
# ==========================================
st.set_page_config(page_title="Pro-Health AI", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load('full_pipeline_compressed.sav')

pipeline = load_model()

st.title("🏥 Professional Clinical Analysis Dashboard")
st.info("AI-powered diagnostic report with clinical benchmarking and point-percentage tracking.")

# Input Section
col1, col2 = st.columns(2)
with col1:
    p_name = st.text_input("Patient Full Name", "User Name")
    age = st.number_input("Age", 1, 120, 40)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    bmi = st.number_input("BMI (kg/m2)", 10.0, 50.0, 26.5)
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    activity = st.slider("Activity (Hrs/Week)", 0.0, 10.0, 3.0)

with col2:
    bp = st.number_input("Systolic BP (mmHg)", 80, 200, 135)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 400, 220)
    glucose = st.number_input("Glucose (mg/dL)", 50, 300, 140)
    stress = st.slider("Stress (1-10)", 1, 10, 6)
    diet = st.selectbox("Diet", ["Poor", "Average", "Good"])
    family_hist = st.selectbox("Family History", ["No", "Yes"])

features = ['Age', 'Gender', 'BMI', 'Smoking', 'AlcoholIntake', 'PhysicalActivity', 'DietQuality', 'SleepHours', 'BloodPressure', 'Cholesterol', 'Glucose', 'FamilyHistory', 'StressLevel']
input_df = pd.DataFrame([[age, gender, bmi, smoking, "Low", activity, diet, 7.0, bp, cholesterol, glucose, family_hist, stress]], columns=features)

if st.button("Generate Professional Analysis"):
    prob = pipeline.predict_proba(input_df)[0][1] * 100
    res_text = "High Risk Level" if prob > 50 else "Safe Health Level"
    
    st.markdown("---")
    # --- 1. KEY METRICS WITH PERCENTAGE DEVIATION ---
    st.subheader("📍 Real-time Clinical Markers")
    m1, m2, m3, m4 = st.columns(4)
    
    # Calculation Logic for % Deviation
    bp_dev = ((bp - 120)/120)*100
    glu_dev = ((glucose - 100)/100)*100
    cho_dev = ((cholesterol - 200)/200)*100
    bmi_dev = ((bmi - 22)/22)*100

    m1.metric("Systolic BP", f"{bp} mmHg", f"{bp_dev:+.1f}% from 120", delta_color="inverse")
    m2.metric("Glucose", f"{glucose} mg/dL", f"{glu_dev:+.1f}% from 100", delta_color="inverse")
    m3.metric("Cholesterol", f"{cholesterol} mg/dL", f"{cho_dev:+.1f}% from 200", delta_color="inverse")
    m4.metric("BMI", f"{bmi:.1f}", f"{bmi_dev:+.1f}% from 22", delta_color="inverse")

    st.markdown("---")
    
    # --- 2. ADVANCED ANALYTICS ROW ---
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**1. AI Logic: Factor Impact (SHAP)**")
        try:
            model = pipeline.named_steps['classifier']
            preprocessor = pipeline.named_steps['preprocessor']
            X_trans = preprocessor.transform(input_df)
            f_names = preprocessor.get_feature_names_out()
            explainer = shap.TreeExplainer(model)
            shap_v = explainer.shap_values(X_trans)
            
            impact = shap_v[1].flatten() if isinstance(shap_v, list) else shap_v[0]
            shap_df = pd.DataFrame({'Factor': f_names, 'Impact': impact}).sort_values(by='Impact', ascending=False).head(5)
            
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            sns.barplot(x='Impact', y='Factor', data=shap_df, palette='Reds_r', ax=ax1)
            plt.title("Why Risk is High?")
            st.pyplot(fig1)
        except:
            st.info("Analyzing risk factors...")

    with col_b:
        st.markdown("**2. Visual Benchmarking (Points)**")
        bench_data = {
            'Metric': ['BP', 'Glucose', 'Chol', 'BMI'],
            'Your Value': [bp, glucose, cholesterol, bmi],
            'Healthy Avg': [120, 100, 200, 22]
        }
        df_bench = pd.DataFrame(bench_data)
        
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        # Plotting both side by side
        x = np.arange(len(df_bench['Metric']))
        width = 0.35
        
        ax2.bar(x - width/2, df_bench['Your Value'], width, label='Your Points', color='#e74c3c')
        ax2.bar(x + width/2, df_bench['Healthy Avg'], width, label='Clinical Target', color='#2ecc71')
        
        # Adding point labels on top of bars
        for i, v in enumerate(df_bench['Your Value']):
            ax2.text(i - width/2, v + 2, str(v), color='black', fontweight='bold', ha='center')

        ax2.set_xticks(x)
        ax2.set_xticklabels(df_bench['Metric'])
        ax2.legend()
        plt.title("Point-to-Point Comparison")
        st.pyplot(fig2)

    # Recommendations Logic
    recs = []
    if bp > 130: recs.append(f"-> BLOOD PRESSURE: Your BP is {bp_dev:.1f}% above healthy limits. Consult for sodium restriction.")
    if glucose > 140: recs.append(f"-> GLUCOSE: Blood sugar is {glu_dev:.1f}% higher than target (100mg/dL). Immediate carb monitoring recommended.")
    if bmi > 25: recs.append(f"-> WEIGHT: BMI ({bmi}) is {bmi_dev:.1f}% above the ideal benchmark (22). Cardio exercises advised.")
    if not recs: recs.append("-> Excellent! All your metrics are within clinical target ranges.")

    # PDF Download
    medical_summary = {
        "BP (Systolic)": f"{bp} mmHg ({bp_dev:+.1f}%)",
        "Glucose": f"{glucose} mg/dL ({glu_dev:+.1f}%)",
        "Cholesterol": f"{cholesterol} mg/dL ({cho_dev:+.1f}%)",
        "Body Mass Index": f"{bmi:.1f} ({bmi_dev:+.1f}%)",
        "Risk Score": f"{prob:.1f}%"
    }
    pdf_b = create_pdf(p_name, age, gender, res_text, prob, recs, medical_summary)
    st.download_button(label="📥 Download Clinical Report", data=pdf_b, file_name=f"Clinical_{p_name}.pdf")
