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
    # 1. Prediction & Probability
    prediction = pipeline.predict(input_data)[0]
    # get_params() or accessing named steps to get probability if supported
    prob = pipeline.predict_proba(input_data)[0][1] * 100

    # Display Result
    if prediction == 1:
        st.error(f"### Result: Chronic Disease Detected (Risk: {prob:.1f}%)")
    else:
        st.success(f"### Result: No Chronic Disease (Risk: {prob:.1f}%)")

    # 2. FEATURE IMPORTANCE (New Feature)
    st.subheader("📊 Key Factors Influencing Your Result")
    # Getting feature importances from the RandomForest inside the pipeline
    rf_model = pipeline.named_steps['classifier']
    importances = rf_model.feature_importances_
    
    # Simple Bar Chart for Visual Appeal
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', ax=ax)
    plt.title("Importance of Each Health Metric", fontweight='bold')
    st.pyplot(fig)

    # 3. PERSONALIZED RECOMMENDATIONS (New Feature)
    st.subheader("💡 Health Recommendations for You")
    
    recs = []
    if bmi > 25: recs.append("- **Weight Management:** Your BMI is slightly high. Consider a balanced diet and regular exercise.")
    if glucose > 140: recs.append("- **Blood Sugar Control:** Your glucose levels are elevated. Reduce sugar intake and consult a doctor.")
    if bp > 130: recs.append("- **BP Monitoring:** Your blood pressure is above normal. Reduce salt intake and practice relaxation.")
    if smoking == "Yes": recs.append("- **Quit Smoking:** Smoking significantly increases chronic disease risk. Seek support to quit.")
    if stress > 7: recs.append("- **Stress Relief:** High stress detected. Try meditation or yoga for mental well-being.")
    if activity < 2: recs.append("- **Stay Active:** Increase your physical activity to at least 150 minutes per week.")
    
    if not recs:
        st.write("Great job! Maintain your current healthy lifestyle.")
    else:
        for r in recs:
            st.write(r)
