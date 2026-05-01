# app.py
import streamlit as st
import pandas as pd
from agents.swarm_logic import DiagnosticSwarm # Importing our Swarm Engine
from dashboard.icu_live import render_icu_dashboard

# --- PAGE SETUP ---
st.set_page_config(page_title="OmniHealth AI CDSS", page_icon="🌐", layout="wide")

# --- CUSTOM CSS FOR ENTERPRISE LOOK ---
st.markdown("""
<style>
    .swarm-card {
        background: #1e1e2f; color: #ffffff; 
        border-radius: 10px; padding: 20px; text-align: center;
        border-left: 5px solid #764ba2;
    }
    .alert-danger { background: #ff4b4b; color: white; padding: 15px; border-radius: 8px; font-weight: bold; }
    .alert-safe { background: #00cc96; color: white; padding: 15px; border-radius: 8px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZE SWARM ---
@st.cache_resource
def load_ai_swarm():
    return DiagnosticSwarm()

swarm_engine = load_ai_swarm()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🌐 OmniHealth AI")
st.sidebar.caption("Enterprise Clinical Decision Support")
module = st.sidebar.radio("Select Module", ["Diagnostic Swarm AI", "Medical Imaging (Beta)", "ICU Live Monitor (Beta)"])

if module == "Diagnostic Swarm AI":
    st.title("🧠 Multi-Agent Diagnostic Swarm")
    st.write("Real-time collaborative AI analysis for proactive health monitoring.")
    st.divider()

    col1, col2 = st.columns([1, 2], gap="large")

    # --- INPUT SECTION ---
    with col1:
        st.subheader("📋 Patient Vitals")
        age = st.number_input("Age", 18, 100, 45)
        bmi = st.slider("BMI", 10.0, 50.0, 26.5)
        bp = st.slider("Blood Pressure (Systolic)", 80, 200, 135)
        glucose = st.slider("Fasting Glucose", 60, 300, 110)
        
        st.write("**Lifestyle Factors**")
        smoking = st.selectbox("Smoking", ["No", "Yes"])
        activity = st.slider("Physical Activity (hrs/wk)", 0.0, 20.0, 4.0)
        diet = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
        sleep = st.slider("Sleep Hours", 3.0, 12.0, 6.0)
        stress = st.slider("Stress Level (1-10)", 1, 10, 6)

        analyze_btn = st.button("Initialize Swarm Analysis", type="primary", use_container_width=True)

    # --- OUTPUT SECTION ---
    with col2:
        st.subheader("📊 Swarm Intelligence Output")
        
        if analyze_btn:
            # 1. Package data for the swarm
            patient_data = {
                "Age": age, "BMI": bmi, "BloodPressure": bp, "Glucose": glucose,
                "Smoking": smoking, "PhysicalActivity": activity, 
                "Diet": diet, "Sleep": sleep, "StressLevel": stress
            }

            # 2. Trigger the multi-agent engine
            with st.spinner("Agents are analyzing patient data..."):
                results = swarm_engine.get_swarm_consensus(patient_data)

            # 3. Display Overall Consensus
            st.write("### Global Health Consensus")
            overall = results["Overall_Consensus"]
            
            if results["Critical_Alert"]:
                st.markdown(f'<div class="alert-danger">⚠️ HIGH RISK CONCERN DETECTED - Swarm Consensus: {overall*100:.1f}%</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-safe">✅ PATIENT STABLE - Swarm Consensus: {overall*100:.1f}%</div>', unsafe_allow_html=True)
            
            st.divider()

            # 4. Display Individual Agent Reports
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
            st.info("👈 Enter patient vitals and click 'Initialize Swarm Analysis' to see the multi-agent collaboration in real-time.")

elif module == "Medical Imaging (Beta)":
    st.title("👁️ Computer Vision Diagnostics")
    st.warning("Module under development. Will feature X-Ray and MRI anomaly detection soon.")

elif module == "ICU Live Monitor (Beta)":
    st.title("🫀 Live Vitals Dashboard")
    st.warning("Module under development. Will feature real-time IoT simulated data streaming.")
