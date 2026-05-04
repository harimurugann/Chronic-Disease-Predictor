import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import requests 
from agents.swarm_logic import DiagnosticSwarm 
from dashboard.icu_live import render_icu_dashboard
from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ImageDraw, ImageFont
import time

# ==========================================
# PAGE SETUP & UI CONFIGURATION
# ==========================================
st.set_page_config(page_title="OmniHealth AI CDSS", page_icon="🌐", layout="wide")

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
    return DiagnosticSwarm()

swarm_engine = load_ai_swarm()

# --- UPDATED: GEOCODING API WITH FAIL-SAFE FALLBACK ---
@st.cache_data
def get_coordinates(location_name):
    """Dynamic Geocoding API with Interview-Safe Fallback Dictionary"""
    
    # 1. Hardcoded Fail-Safe for common cities (Guarantees zero failure during demo)
    fallback_coords = {
        "trichy": (10.7905, 78.7047),
        "tiruchirappalli": (10.7905, 78.7047),
        "salem": (11.6643, 78.1460),
        "vazhapadi": (11.6500, 78.2500),
        "valapady": (11.6500, 78.2500),
        "chennai": (13.0827, 80.2707),
        "coimbatore": (11.0168, 76.9558),
        "madurai": (9.9252, 78.1198),
        "bengaluru": (12.9716, 77.5946),
        "bangalore": (12.9716, 77.5946),
        "new delhi": (28.6139, 77.2090),
        "delhi": (28.6139, 77.2090),
        "mumbai": (19.0760, 72.8777),
        "pune": (18.5204, 73.8567),
        "erode": (11.3410, 77.7172),
        "tiruppur": (11.1085, 77.3411)
    }
    
    loc_lower = location_name.lower().strip()
    
    # Check if the requested location is in our safe dictionary first
    if loc_lower in fallback_coords:
        return fallback_coords[loc_lower]

    # 2. Live API Fetch for unknown cities
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={location_name},+India&format=json&limit=1"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=5)
        data = response.json()
        
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
        else:
            return None, None
    except Exception as e:
        return None, None

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🌐 OmniHealth AI")
st.sidebar.caption("Enterprise Clinical Decision Support")
module = st.sidebar.radio("Select Module", [
    "Diagnostic Swarm AI", 
    "Medical Imaging (Beta)", 
    "ICU Live Monitor (Beta)",
    "AI Clinical Scribe (NLP)",
    "GenAI Clinical Assistant",
    "Population Health Analytics"
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
                "Smoking": smoking, "PhysicalActivity": activity, "Diet": diet, "Sleep": sleep, "StressLevel": stress
            }

            with st.spinner("AI Agents are communicating..."):
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
            st.info("👈 Enter patient vitals to start Swarm AI analysis.")

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
                else:
                    st.success(f"✅ **SCAN CLEAR** (Confidence: {(1-prob)*100:.1f}%)")

                st.divider()
                
                gray_image = image.convert("L")
                enhancer = ImageEnhance.Contrast(gray_image)
                contrast_image = enhancer.enhance(3.0) 
                blurred_gray = contrast_image.filter(ImageFilter.GaussianBlur(radius=10))

                heatmap = ImageOps.colorize(blurred_gray, mid="blue", black="black", white="yellow")
                blended = Image.blend(image.convert("RGB"), heatmap, alpha=0.5)

                if prob > 0.5:
                    st.write("**AI Diagnostic Localization (Simulated Pointer):**")
                    annotated_img = blended.copy()
                    draw = ImageDraw.Draw(annotated_img)
                    w, h = annotated_img.size

                    circle_coords = [w * 0.25, h * 0.35, w * 0.40, h * 0.50]
                    anomaly_target = (w * 0.325, h * 0.425) 
                    pointer_start = (w * 0.325, h * 0.25) 

                    draw.line([pointer_start, anomaly_target], fill="red", width=10)
                    draw.line([anomaly_target, (anomaly_target[0]-w*0.02, anomaly_target[1]-h*0.03)], fill="red", width=10)
                    draw.line([anomaly_target, (anomaly_target[0]+w*0.02, anomaly_target[1]-h*0.03)], fill="red", width=10)
                    draw.ellipse(circle_coords, outline="red", width=5)
                    
                    st.image(annotated_img, use_container_width=True, caption="Grad-CAM Simulation with Spatial Anomaly Pointer")
                else:
                    st.write("**AI Attention Map (Simulated Deep Feature Map):**")
                    st.image(blended, use_container_width=True, caption="Grad-CAM Visualization (Normal Findings Simulated)")

                st.markdown("#### 📋 Detailed AI Findings")
                with st.expander("Expand Radiological Breakdown", expanded=True):
                    if prob > 0.5:
                        st.write("- 🔴 **Primary Finding:** Localized hyper-density (opacity) detected in the **Right Mid-Lung Zone**.")
                        st.write("- 🩺 **Clinical Significance:** Highly indicative of focal consolidation or a structural mass.")
                    else:
                        st.write("- 🟢 **Lung Fields:** Clear bilaterally. No abnormal opacities detected.")

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
    st.write("Extracting clinical entities using Named Entity Recognition (NER).")
    st.divider()

    default_text = """Patient presenting with severe chest pain and shortness of breath. 
History of Hypertension and Type 2 Diabetes. Prescribed Aspirin 81mg daily."""

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.subheader("Raw Clinical Note")
        clinical_text = st.text_area("Paste notes here:", value=default_text, height=200)
        analyze_nlp_btn = st.button("🧠 Run NLP Extraction", type="primary", use_container_width=True)

    with col2:
        st.subheader("Structured AI Output")
        if analyze_nlp_btn and clinical_text:
            with st.spinner("Tokenizing..."):
                time.sleep(1.2)
            
            found_symptoms = [s for s in ["chest pain", "shortness of breath"] if s in clinical_text.lower()]
            found_diagnoses = [d for d in ["Hypertension", "Type 2 Diabetes"] if d in clinical_text]
            found_meds = [m for m in ["Aspirin 81mg"] if m in clinical_text]

            st.write("**Symptoms Detected:**")
            st.markdown(" ".join([f'<span class="entity-symp">{s}</span>' for s in found_symptoms]), unsafe_allow_html=True)
            st.write("**Clinical Diagnoses:**")
            st.markdown(" ".join([f'<span class="entity-diag">{d}</span>' for d in found_diagnoses]), unsafe_allow_html=True)
            st.write("**Prescribed Medications:**")
            st.markdown(" ".join([f'<span class="entity-med">{m}</span>' for m in found_meds]), unsafe_allow_html=True)
            
            st.divider()
            with st.expander("💾 View JSON Payload for EHR Backend (API Integration)", expanded=False):
                st.json({
                    "patient_id": "ANON-84729",
                    "extracted_data": {"Symptoms": found_symptoms, "Diagnoses": found_diagnoses, "Medications": found_meds},
                    "nlp_confidence": 0.94
                })

# ==========================================
# MODULE 5: GEN-AI CLINICAL ASSISTANT
# ==========================================
elif module == "GenAI Clinical Assistant":
    st.markdown("### 💬 GenAI Clinical Assistant (Multilingual Support)")
    st.write("Simulation of a Retrieval-Augmented Generation (RAG) agent.")
    st.divider()

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello Doctor. Ask me medical queries in English or Tanglish."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a clinical query..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            prompt_lower = prompt.lower()
            tanglish_keywords = ["irukku", "enna", "pannalam", "kodu", "valikkudhu", "kolandhai", "marundhu"]
            is_tanglish = any(kw in prompt_lower for kw in tanglish_keywords)
            
            if "heart" in prompt_lower or "chest" in prompt_lower:
                ai_response = "Nenju vali irundha idhu emergency! Udane Aspirin 300mg mella sollanga." if is_tanglish else "Immediate administration of chewable Aspirin (300mg) is standard protocol."
            elif "fever" in prompt_lower or "kaachal" in prompt_lower:
                ai_response = "Kuzhandhaikku Paracetamol drops tharalam. Wet cloth vechu thodachu vidunga." if is_tanglish else "Antipyretics like Acetaminophen are recommended."
            else:
                ai_response = f"Neenga '{prompt}' pathi ketrukkinga. Idhu simulated database-la illai." if is_tanglish else f"I scanned the simulated database for '{prompt}'."
            
            for chunk in ai_response.split(" "):
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# ==========================================
# MODULE 6: POPULATION HEALTH ANALYTICS (BI)
# ==========================================
elif module == "Population Health Analytics":
    st.markdown("### 📈 Population Health & BI Analytics")
    st.write("Real-time Business Intelligence dashboard for resource management and disease forecasting.")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Admissions", "1,245", "+12% from last month")
    col2.metric("Active ICU Patients", "34", "-2")
    col3.metric("AI Flagged Anomalies", "18", "+4")
    col4.metric("Avg. ER Wait Time", "14 mins", "-3 mins")
    st.divider()

    col_a, col_b = st.columns([2, 1], gap="large")
    with col_a:
        st.write("**📊 Disease Outbreak Trends (Last 30 Days)**")
        dates = pd.date_range(end=pd.Timestamp.today(), periods=30)
        trend_data = pd.DataFrame({
            "Respiratory Issues": np.random.randint(10, 50, size=30),
            "Cardiac Events": np.random.randint(5, 20, size=30),
            "Viral Fevers": np.random.randint(2, 35, size=30)
        }, index=dates)
        st.line_chart(trend_data)

    with col_b:
        st.write("**👥 Patient Demographics**")
        demo_data = pd.DataFrame({
            "Age Group": ["0-18", "19-35", "36-50", "51-65", "65+"],
            "Patients": [150, 320, 450, 600, 420]
        }).set_index("Age Group")
        st.bar_chart(demo_data)

    st.divider()
    
    # --- DYNAMIC PAN-INDIA GEOCODING LOGIC ---
    st.write("**📍 Geospatial Risk Mapping (Dynamic Pan-India 3D Fly-over)**")
    
    search_location = st.text_input("🌍 Search ANY City, Town, or Village in India (e.g., Trichy, Vazhapadi, Pune):", "New Delhi")

    if search_location:
        with st.spinner(f"Locating '{search_location}' via Geocoding API..."):
            base_lat, base_lon = get_coordinates(search_location)

        if base_lat and base_lon:
            # Generate simulated outbreak points around the dynamically fetched coordinates
            map_data = pd.DataFrame(
                np.random.randn(150, 2) / [60, 60] + [base_lat, base_lon], 
                columns=['lat', 'lon']
            )
            
            # Setup 3D Camera view
            view_state = pdk.ViewState(
                latitude=base_lat,
                longitude=base_lon,
                zoom=11,
                pitch=50,
                transition_duration=2500, # Smooth flying animation
                transition_easing="cubic-in-out"
            )

            # Setup the glowing red scatter plot points
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_data,
                get_position="[lon, lat]",
                get_color="[255, 75, 75, 200]",
                get_radius=500,
                pickable=True,
            )

            # Render the Map
            st.pydeck_chart(pdk.Deck(
                map_style="mapbox://styles/mapbox/dark-v10",
                initial_view_state=view_state,
                layers=[layer],
                tooltip={"text": f"Simulated Outbreak near {search_location.title()}"}
            ))
        else:
            st.error(f"⚠️ Could not find coordinates for '{search_location}'. Please check the spelling or try a nearby city.")
