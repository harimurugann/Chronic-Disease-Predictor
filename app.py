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
    paths = [
        "artifacts/chronic_disease_best_pipeline.pkl.gz",
        "artifacts/chronic_disease_pipeline.pkl.gz",
        "artifacts/chronic_disease_gbm_model.sav"
    ]
    model = None
    model_path = "None"
    for p in paths:
        if os.path.exists(p):
            model = joblib.load(p)
            model_path = p
            break
    
    # Dataset for Live Insights
    data_path = "data/chronic_disease_dataset.csv" 
    df = pd.read_csv(data_path) if os.path.exists(data_path) else pd.DataFrame()
    return model, df, model_path

model, df, model_path = load_resources()

# ─── SIDEBAR NAVIGATION ──────────────────────────────────────────────────────
st.sidebar.title("🏥 Health Panel")
page = st.sidebar.radio("Navigation", ["Diagnosis", "Live Insights", "Metrics"])

if page == "Diagnosis":
    st.subheader("🔬 Patient Risk Assessment")
    
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.write("### Input Vitals")
        age = st.number_input("Age", 18, 100, 45)
        gender = st.selectbox("Gender", ["Male", "Female"])
        bmi = st.slider("BMI", 10.0, 50.0, 25.0)
        smoking = st.selectbox("Smoking", ["No", "Yes"])
        alcohol = st.selectbox("Alcohol Intake", ["Low", "Moderate", "High"])
        activity = st.slider("Physical Activity (hrs/wk)", 0.0, 20.0, 5.0)
        diet = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
        sleep = st.slider("Sleep Hours", 3.0, 12.0, 7.0)
        bp = st.slider("Blood Pressure", 80, 200, 120)
        cholesterol = st.slider("Cholesterol", 100, 350, 180)
        glucose = st.slider("Glucose", 60, 300, 100)
        family_hist = st.selectbox("Family History", ["No", "Yes"])
        stress = st.slider("Stress Level", 1, 10, 5)
        
        predict_btn = st.button("RUN ANALYSIS")

    with col2:
        if predict_btn:
            if model:
                # 🛠️ EXACT MATCH WITH TRAINING COLUMNS (Check spelling carefully)
                input_df = pd.DataFrame([{
                    "Age": age,
                    "Gender": gender,
                    "BMI": bmi,
                    "Smoking": smoking,
                    "AlcoholIntake": alcohol,
                    "PhysicalActivity": activity,
                    "DietQuality": diet,
                    "SleepHours": sleep,
                    "BloodPressure": bp,
                    "Cholesterol": cholesterol,
                    "Glucose": glucose,
                    "FamilyHistory": family_hist,
                    "StressLevel": stress
                }])

                try:
                    prob = model.predict_proba(input_df)[0][1]
                    res_color = "#f5576c" if prob > 0.5 else "#2ecc71"
                    
                    st.markdown(f"""
                        <div style="background-color:{res_color}; padding:25px; border-radius:15px; text-align:center; color:white;">
                            <h2 style='color:white;'>Risk Probability: {prob*100:.1f}%</h2>
                            <p style='font-size:1.2rem;'>{"HIGH RISK - MEDICAL ATTENTION NEEDED" if prob > 0.5 else "LOW RISK - STATUS HEALTHY"}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.divider()
                    st.write("#### 📊 Key Indicators")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("BMI", f"{bmi:.1f}", "Warning" if bmi > 25 else "Normal", delta_color="inverse")
                    m2.metric("BP", bp, "High" if bp > 130 else "Optimal", delta_color="inverse")
                    m3.metric("Glucose", glucose, "High" if glucose > 140 else "Stable", delta_color="inverse")
                
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
                    st.info("💡 Hint: Your model expects specific column names. Check your training dataset columns.")
            else:
                st.error("Model artifacts not found! Check 'artifacts' folder.")

elif page == "Live Insights":
    st.subheader("📈 Population Data Trends")
    if not df.empty:
        tab1, tab2 = st.tabs(["Glucose vs Risk", "Age vs BP"])
        with tab1:
            fig = px.histogram(df, x="Glucose", color="HasChronicDisease", barmode="overlay", title="Glucose Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            fig2 = px.scatter(df, x="Age", y="BloodPressure", color="HasChronicDisease", trendline="ols")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Please place 'chronic_disease_dataset.csv' in the 'data' folder for live visuals.")

elif page == "Metrics":
    st.subheader("🎯 Model Validation")
    img_path = "artifacts/09_feature_importance.png"
    if os.path.exists(img_path):
        st.image(img_path, caption="Feature Importance (Global)")
    else:
        st.error("Metric images not found in artifacts.")

st.sidebar.divider()
st.sidebar.info(f"Model: {model_path.split('/')[-1]}")
