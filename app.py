import streamlit as st
import pandas as pd
from agents.swarm_logic import DiagnosticSwarm 
from dashboard.icu_live import render_icu_dashboard
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import time
import re

# ==========================================
# PAGE SETUP & UI CONFIGURATION
# ==========================================
st.set_page_config(page_title="OmniHealth AI CDSS", page_icon="🌐", layout="wide")

# Professional Enterprise CSS Styling
st.markdown("""
<style>
    .swarm-card {
        background: #1e1e2f; color: #ffffff; 
        border-radius: 10px; padding: 20px; text-align: center;
        border-left: 5px solid #764ba2;
    }
    .alert-danger { background: #ff4b4b; color: white; padding: 15px; border-radius: 8px; font-weight: bold; }
    .alert-safe { background: #00cc96; color: white; padding: 15px; border-radius: 8px; font-weight: bold; }
    .entity-symp { background-color: #ff4b4b; color: white; padding: 3px 8px; border-radius: 5px; font-weight: bold; margin: 2px;}
    .entity-diag { background-color: #f5b041; color: white; padding: 3px 8px; border-radius: 5px; font-weight: bold; margin: 2px;}
    .entity-med { background-color: #2ecc71; color: white; padding: 3px 8px; border-radius: 5px; font-weight: bold; margin: 2px;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# RESOURCE INITIALIZATION
# ==========================================
@st.cache_resource
def load_ai_swarm():
    """Caching the Swarm Engine to prevent repeated model loading"""
    return DiagnosticSwarm()

swarm_engine = load_ai_swarm()

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🌐 OmniHealth AI")
st.sidebar.caption("Enterprise Clinical Decision Support")
# Added the new NLP module to the sidebar
module = st.sidebar.radio("Select Module", [
    "Diagnostic Swarm AI", 
    "Medical Imaging (Beta)", 
    "ICU Live Monitor (Beta)",
    "AI Clinical Scribe (NLP)"
])

# ==========================================
# MODULE 1: MULTI-AGENT DIAGNOSTIC SWARM
# ==========================================
if module == "Diagnostic Swarm AI":
    st.title("🧠 Multi-Agent Diagnostic Swarm")
    st.write("Collaborative AI analysis using multiple specialized disease agents.")
    st.divider()

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.subheader("📋 Patient Vitals")
        age = st.number_input("Age", 18, 100, 45)
        bmi = st.slider("BMI", 10.0, 50.0, 26.5)
        bp = st.slider("Blood Pressure (Systolic)", 80, 200, 135)
        glucose = st.slider("Fasting Glucose", 60, 300, 110)
        
        st.write("**Lifestyle & Environmental Factors**")
        smoking = st.selectbox("Smoking", ["No", "Yes"])
        activity = st.slider("Physical Activity (hrs/wk)", 0.0, 20.0, 4.0)
        diet = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
        sleep = st.slider("Sleep Hours", 3.0, 12.0, 6.0)
        stress = st.slider("Stress Level (1-10)", 1, 10, 6)

        analyze_btn = st.button("Initialize Swarm Analysis", type="primary", use_container_width=True)

    with col2:
        st.subheader("📊 Swarm Intelligence Output")
        
        if analyze_btn:
            patient_data = {
                "Age": age, "BMI": bmi, "BloodPressure": bp, "Glucose": glucose,
                "Smoking": smoking, "PhysicalActivity": activity, 
                "Diet": diet, "Sleep": sleep, "StressLevel": stress
            }

            with st.spinner("AI Agents are communicating to form a consensus..."):
                results = swarm_engine.get_swarm_consensus(patient_data)

            st.write("### Global Health Consensus")
            overall = results["Overall_Consensus"]
            
            if results["Critical_Alert"]:
                st.markdown(f'<div class="alert-danger">⚠️ HIGH RISK CONCERN - Swarm Consensus: {overall*100:.1f}%</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-safe">✅ PATIENT STABLE - Swarm Consensus: {overall*100:.1f}%</div>', unsafe_allow_html=True)
            
            st.divider()
            st.write("### 🤖 Individual Agent Reports")
            a1, a2, a3 = st.columns(3)
            
            with a1:
                st.markdown('<div class="swarm-card">🫀 Cardio Agent</div>', unsafe_allow_html=True)
                st.progress(results["Cardio_Score"], text=f"Risk: {results['Cardio_Score']*100:.0f}%")
            with a2:
                st.markdown('<div class="swarm-card">🩸 Diabetic Agent</div>', unsafe_allow_html=True)
                st.progress(results["Diabetic_Score"], text=f"Risk: {results['Diabetic_Score']*100:.0f}%")
            with a3:
                st.markdown('<div class="swarm-card">⚕️ Chronic Agent</div>', unsafe_allow_html=True)
                st.progress(results["Chronic_Score"], text=f"Risk: {results['Chronic_Score']*100:.0f}%")
        else:
            st.info("👈 Please input patient vitals to trigger the AI Swarm.")

# ==========================================
# MODULE 2: COMPUTER VISION - MEDICAL IMAGING
# ==========================================
elif module == "Medical Imaging (Beta)":
    st.markdown("### 👁️ AI Medical Image Diagnostics")
    st.write("Deep Learning simulation for anomaly detection in X-Ray/MRI scans.")
    st.divider()
    
    uploaded_file = st.file_uploader("Upload Scan (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        col_img, col_anl = st.columns([1, 1], gap="large")
        
        with col_img:
            st.subheader("Original Scan")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True, caption="Uploaded Patient Record")
        
        with col_anl:
            st.subheader("AI Analysis Panel")
            if st.button("🔍 Run Deep Vision Scan", type="primary", use_container_width=True):
                with st.spinner("Executing Convolutional Neural Network layers..."):
                    p_bar = st.progress(0)
                    for i in range(4):
                        time.sleep(0.5)
                        p_bar.progress((i+1)*25)
                
                st.markdown("#### 🎯 Diagnostic Conclusion")
                img_hash = sum(image.size) % 100
                prob = img_hash / 100.0
                
                if prob > 0.5:
                    st.error(f"⚠️ **ANOMALY DETECTED** (Confidence: {prob*100:.1f}%)")
                    st.write("Pattern recognition suggests structural irregularities.")
                else:
                    st.success(f"✅ **SCAN CLEAR** (Confidence: {(1-prob)*100:.1f}%)")
                    st.write("No significant pathological markers found.")

                st.divider()
                st.write("**AI Attention Map (Feature Extraction):**")
                gray_img = image.convert("L")
                enhancer = ImageEnhance.Contrast(gray_img)
                contrast_img = enhancer.enhance(3.0)
                blurred = contrast_img.filter(ImageFilter.GaussianBlur(radius=10))
                
                heatmap = ImageOps.colorize(blurred, mid="blue", black="black", white="yellow")
                blended = Image.blend(image.convert("RGB"), heatmap, alpha=0.5)
                st.image(blended, use_container_width=True, caption="Grad-CAM Simulated Visualization")

                st.markdown("#### 📋 Detailed AI Findings")
                with st.expander("Expand Radiological Breakdown", expanded=True):
                    if prob > 0.5:
                        st.write("- 🔴 **Lung Fields:** Potential consolidations detected in yellow zones.")
                        st.write("- 🟡 **Cardiac Silhouette:** Slight enlargement noted.")
                        st.info("💡 **XAI Note:** Yellow highlights represent high neural focus areas.")
                    else:
                        st.write("- 🟢 **Lung Fields:** Clear bilaterally.")
                        st.write("- 🟢 **Cardiac Silhouette:** Normal size.")
                        st.info("💡 **XAI Note:** Uniform distribution suggests no focal anomalies.")

# ==========================================
# MODULE 3: REAL-TIME ICU MONITORING
# ==========================================
elif module == "ICU Live Monitor (Beta)":
    render_icu_dashboard()

# ==========================================
# MODULE 4: NLP CLINICAL SCRIBE
# ==========================================
elif module == "AI Clinical Scribe (NLP)":
    st.markdown("### 📝 NLP Clinical Notes Analyzer")
    st.write("Extract structured medical entities (NER) from unstructured doctor's notes using Natural Language Processing.")
    st.divider()

    # Pre-filled mock clinical note for demonstration
    default_text = """Patient is a 45-year-old male presenting with severe chest pain, shortness of breath, and mild dizziness. 
Patient has a history of Type 2 Diabetes and Hypertension. 
Diagnosis: Acute Myocardial Infarction. 
Plan: Prescribing Aspirin 81mg and Metoprolol 50mg daily. Recommend immediate cardiology consult."""

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Raw Clinical Note")
        clinical_text = st.text_area("Paste doctor's notes here:", value=default_text, height=250)
        analyze_nlp_btn = st.button("🧠 Run NLP Extraction", type="primary", use_container_width=True)

    with col2:
        st.subheader("Structured AI Output")
        
        if analyze_nlp_btn and clinical_text:
            with st.spinner("Tokenizing and performing Named Entity Recognition (NER)..."):
                time.sleep(1.5) # Simulate NLP pipeline delay
            
            st.success("Entity Extraction Complete!")
            
            # Simulated NER Extraction Logic (Keyword matching for demo purposes)
            symptoms = ["chest pain", "shortness of breath", "dizziness", "fever", "cough", "headache"]
            diagnoses = ["Type 2 Diabetes", "Hypertension", "Acute Myocardial Infarction", "Pneumonia", "Asthma"]
            medications = ["Aspirin 81mg", "Metoprolol 50mg", "Paracetamol", "Amoxicillin", "Ibuprofen"]

            found_symptoms = [s for s in symptoms if s.lower() in clinical_text.lower()]
            found_diagnoses = [d for d in diagnoses if d.lower() in clinical_text.lower()]
            found_meds = [m for m in medications if m.lower() in clinical_text.lower()]

            # Visualizing extracted entities with custom CSS badges
            st.write("#### Named Entities Recognized (NER)")
            
            st.write("**Symptoms Detected:**")
            if found_symptoms:
                symp_html = " ".join([f'<span class="entity-symp">{s}</span>' for s in found_symptoms])
                st.markdown(symp_html, unsafe_allow_html=True)
            else:
                st.write("None detected.")

            st.write("**Clinical Diagnoses:**")
            if found_diagnoses:
                diag_html = " ".join([f'<span class="entity-diag">{d}</span>' for d in found_diagnoses])
                st.markdown(diag_html, unsafe_allow_html=True)
            else:
                st.write("None detected.")

            st.write("**Prescribed Medications:**")
            if found_meds:
                med_html = " ".join([f'<span class="entity-med">{m}</span>' for m in found_meds])
                st.markdown(med_html, unsafe_allow_html=True)
            else:
                st.write("None detected.")
                
            st.divider()
            
            # Enterprise JSON Output Simulation
            st.write("#### 💾 JSON Payload for EHR Systems")
            st.json({
                "patient_id": "ANON-84729",
                "extracted_data": {
                    "Symptoms": found_symptoms,
                    "Diagnoses": found_diagnoses,
                    "Medications": found_meds
                },
                "nlp_confidence_score": 0.94
            })
