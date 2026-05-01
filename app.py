import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

# ─── PAGE CONFIGURATION ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pro-Health AI Predictor",
    page_icon="🏥",
    layout="wide"
)

# ─── PREMIUM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; color: #764ba2; }
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 3rem; font-weight: 800; text-align: center; margin-bottom: 0px;
    }
    .card {
        border-radius: 15px; padding: 20px;
        background-color: #f8f9fa; border: 1px solid #e9ecef;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border-radius: 10px; height: 3em; width: 100%;
        font-weight: bold; border: none; transition: 0.3s;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.2); }
</style>
""", unsafe_allow_html=True)

# ─── DATA & MODEL LOADING ───────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    # Model Loading
    paths = ["artifacts/chronic_disease_best_pipeline.pkl.gz", "artifacts/chronic_disease_pipeline.pkl.gz"]
    model = None
    for p in paths:
        if os.path.exists(p):
            model = joblib.load(p)
            break
    
    # Dataset Loading (For Real-time Visuals)
    # Replace with your actual dataset path
    data_path = "data/chronic_disease_dataset.csv" 
    df = pd.read_csv(data_path) if os.path.exists(data_path) else pd.DataFrame()
    
    return model, df

model, df = load_resources()

# ─── SIDEBAR NAVIGATION ──────────────────────────────────────────────────────
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=100)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Analysis & Prediction", "Live Insights", "Model Performance"])

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🏥 Chronic Disease AI</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Real-time Clinical Decision Support System</p>", unsafe_allow_html=True)
st.divider()

if page == "Analysis & Prediction":
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.subheader("📋 Patient Vitals")
        with st.container():
            age = st.number_input("Age", 1, 120, 45)
            gender = st.selectbox("Gender", ["Male", "Female"])
            bmi = st.slider("Body Mass Index (BMI)", 10.0, 50.0, 24.5)
            bp = st.slider("Systolic Blood Pressure", 80, 200, 120)
            glucose = st.slider("Glucose Level", 50, 300, 100)
            
            with st.expander("Lifestyle Factors"):
                smoking = st.selectbox("Smoking Status", ["No", "Yes"])
                alcohol = st.selectbox("Alcohol Intake", ["Low", "Moderate", "High"])
                activity = st.slider("Physical Activity (hrs/wk)", 0, 21, 5)
                stress = st.select_slider("Stress Level", options=list(range(1, 11)))
            
            predict_btn = st.button("RUN AI DIAGNOSIS")

    with col2:
        if predict_btn:
            if model is not None:
                # Prepare Input for Model
                input_df = pd.DataFrame([{
                    "Age": age, "Gender": gender, "BMI": bmi, "Smoking": smoking,
                    "AlcoholIntake": alcohol, "PhysicalActivity": activity,
                    "BloodPressure": bp, "Glucose": glucose, "StressLevel": stress,
                    # Add other missing features used during training here
                }])

                # Prediction Logic
                prob = model.predict_proba(input_df)[0][1]
                
                # Real-time Result UI
                st.subheader("🎯 Diagnosis Result")
                risk_color = "#f5576c" if prob > 0.5 else "#2ecc71"
                
                st.markdown(f"""
                    <div style="background-color:{risk_color}; padding:30px; border-radius:15px; text-align:center; color:white;">
                        <h1 style='color:white;'>{prob*100:.1f}%</h1>
                        <p style='font-size:1.5rem;'>{"HIGH RISK DETECTED" if prob > 0.5 else "LOW RISK / HEALTHY"}</p>
                    </div>
                """, unsafe_allow_html=True)

                # Summary Statistics
                st.write("---")
                m1, m2, m3 = st.columns(3)
                m1.metric("Patient BMI", bmi, delta="- Normal" if 18.5 <= bmi <= 25 else "Attention Required", delta_color="inverse")
                m2.metric("BP Status", bp, delta="Elevated" if bp > 130 else "Normal", delta_color="inverse")
                m3.metric("Glucose", glucose, delta="High" if glucose > 140 else "Stable", delta_color="inverse")
            else:
                st.error("Model file not found in artifacts folder!")
        else:
            st.info("Please input patient data and click 'Run AI Diagnosis' to see real-time results.")

elif page == "Live Insights":
    st.subheader("📈 Real-time Population Analysis")
    if not df.empty:
        col_a, col_b = st.columns(2)
        with col_a:
            fig1 = px.box(df, x="HasChronicDisease", y="Glucose", color="HasChronicDisease", title="Glucose Impact on Disease")
            st.plotly_chart(fig1, use_container_width=True)
        with col_b:
            fig2 = px.scatter(df, x="Age", y="BloodPressure", color="HasChronicDisease", title="Age vs BP Trends")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Dataset not found for live analysis.")

elif page == "Model Performance":
    st.subheader("🧪 Validation Metrics")
    # Static plots from artifacts
    plot_map = {
        "Confusion Matrix": "artifacts/06_confusion_matrix.png",
        "Feature Importance": "artifacts/09_feature_importance.png",
        "ROC Curve": "artifacts/07_roc_curve.png"
    }
    
    selected_plot = st.selectbox("Select Metric", list(plot_map.keys()))
    if os.path.exists(plot_map[selected_plot]):
        st.image(plot_map[selected_plot], use_container_width=True)
    else:
        st.error(f"Plot {selected_plot} not found in artifacts.")

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.caption("⚕️ For Research Purposes Only")
