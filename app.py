import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(page_title="Chronic Disease Expert", layout="wide")

# Load the saved compressed pipeline
pipeline = joblib.load('full_pipeline_compressed.sav')

st.title("🏥 Chronic Disease Prediction & Health Advisor")
st.markdown("---")

# Layout using columns for a better look
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Patient Information")
    age = st.number_input("Age", 1, 120, 30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    alcohol = st.selectbox("Alcohol Intake", ["Low", "Moderate", "High"])
    activity = st.slider("Physical Activity (Hours/Week)", 0.0, 10.0, 3.0)
    
with col2:
    st.header("Medical Metrics")
    diet = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
    sleep = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
    bp = st.number_input("Blood Pressure (Systolic)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol", 100, 400, 200)
    glucose = st.number_input("Glucose Level", 50, 300, 100)
    family_hist = st.selectbox("Family History", ["No", "Yes"])
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)

# Create input dataframe
features = ['Age', 'Gender', 'BMI', 'Smoking', 'AlcoholIntake', 'PhysicalActivity', 'DietQuality', 'SleepHours', 'BloodPressure', 'Cholesterol', 'Glucose', 'FamilyHistory', 'StressLevel']
input_data = pd.DataFrame([[age, gender, bmi, smoking, alcohol, activity, diet, sleep, bp, cholesterol, glucose, family_hist, stress]], columns=features)

st.markdown("---")

if st.button("Analyze Health Status"):
    # 1. Prediction & Risk Probability
    prediction = pipeline.predict(input_data)[0]
    prob = pipeline.predict_proba(input_data)[0][1] * 100

    if prediction == 1:
        st.error(f"### Result: Chronic Disease Detected (Risk: {prob:.1f}%)")
    else:
        st.success(f"### Result: No Chronic Disease (Risk: {prob:.1f}%)")

    # 2. FEATURE IMPORTANCE CHART
    st.subheader("📊 Key Factors Influencing Your Result")
    try:
        rf_model = pipeline.named_steps['classifier']
        preprocessor = pipeline.named_steps['preprocessor']
        ohe_feature_names = preprocessor.get_feature_names_out()
        importances = rf_model.feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': ohe_feature_names, 
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', ax=ax)
        plt.title("Top Factors in Prediction", fontweight='bold')
        st.pyplot(fig)
    except:
        st.info("Feature importance data is loading...")

    # 3. PERSONALIZED RECOMMENDATIONS
    st.subheader("💡 Health Recommendations")
    recs = []
    if bmi > 25: recs.append("- **Weight:** Your BMI is high. Focus on a balanced diet and cardio.")
    if glucose > 140: recs.append("- **Glucose:** High sugar levels. Reduce sweets and refined carbs.")
    if bp > 130: recs.append("- **Blood Pressure:** High BP. Minimize salt and practice meditation.")
    if activity < 3: recs.append("- **Activity:** Try to walk at least 30 minutes daily.")
    
    if not recs:
        st.write("You are doing great! Keep up the healthy lifestyle.")
    else:
        for r in recs:
            st.write(r)
