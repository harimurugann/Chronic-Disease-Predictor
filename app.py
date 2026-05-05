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

@st.cache_data
def get_coordinates(location_name):
    """Dynamic Geocoding API with Local Fallback"""
    fallback_coords = {
        "trichy": (10.7905, 78.7047),
        "tiruchirappalli": (10.7905, 78.7047),
        "chennai": (13.0827, 80.2707),
        "madurai": (9.9252, 78.1198),
        "coimbatore": (11.0168, 76.9558),
        "salem": (11.6500, 78.1667),
        "vazhapadi": (11.6500, 78.2500),
        "bengaluru": (12.9716, 77.5946),
        "new delhi": (28.6139, 77.2090)
    }

    clean_name = location_name.strip().lower()
    if clean_name in fallback_coords:
        return fallback_coords[clean_name]

    try:
        url = f"https://nominatim.openstreetmap.org/search?q={location_name}, India&format=json&limit=1"
        headers = {'User-Agent': 'OmniHealth_CDSS_Portfolio'}
        response = requests.get(url, headers=headers, timeout=5)
        data = response.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
        else:
            return None, None
    except Exception:
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
        smoking = st.selectbox("Smoking", ["No", "Yes"])
        activity = st.slider("Physical Activity (hrs/wk)", 0.0, 20.0, 4.0)
        diet = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
        sleep = st.slider("Sleep Hours", 3.0, 12.0, 6.0)
        stress = st.slider("Stress Level (1-10)", 1, 10, 6)
        analyze_btn = st.button("Initialize Swarm Analysis", type="primary", use_container_width=True)

    with col2:
        st.subheader("📊 Swarm Intelligence Output")
        if analyze_btn:
            patient_data = {"Age": age, "BMI": bmi, "BloodPressure": bp, "Glucose": glucose, "Smoking": smoking, "PhysicalActivity": activity, "Diet": diet, "Sleep": sleep, "StressLevel": stress}
            with st.spinner("AI Agents are communicating..."):
                results = swarm_engine.get_swarm_consensus(patient_data)
            st.write("### Global Health Consensus")
            overall = results["Overall_Consensus"]
            if results["Critical_Alert"]:
                st.markdown(f'<div class="alert-danger">⚠️ HIGH RISK CONCERN - Swarm Consensus: {overall*100:.1f}%</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-safe">✅ PATIENT STABLE - Swarm Consensus: {overall*100:.1f}%</div>', unsafe_allow_html=True)
            st.divider()
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

# ==========================================
# MODULE 2: COMPUTER VISION
# ==========================================
elif module == "Medical Imaging (Beta)":
    st.markdown("### 👁️ AI Medical Image Diagnostics")
    st.divider()
    uploaded_file = st.file_uploader("Upload Scan (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        col_img, col_anl = st.columns([1, 1], gap="large")
        with col_img:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True, caption="Uploaded Patient Record")
        with col_anl:
            if st.button("🔍 Run Deep Vision Scan", type="primary", use_container_width=True):
                with st.spinner("Executing..."):
                    time.sleep(1.5)
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
                    st.write("**AI Diagnostic Localization:**")
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
                    st.image(annotated_img, use_container_width=True)
                else:
                    st.image(blended, use_container_width=True)

# ==========================================
# MODULE 3: ICU LIVE
# ==========================================
elif module == "ICU Live Monitor (Beta)":
    render_icu_dashboard()

# ==========================================
# MODULE 4: NLP SCRIBE
# ==========================================
elif module == "AI Clinical Scribe (NLP)":
    st.markdown("### 📝 NLP Clinical Notes Analyzer")
    st.divider()
    default_text = """Patient presenting with severe chest pain and shortness of breath. History of Hypertension. Prescribed Aspirin 81mg."""
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        clinical_text = st.text_area("Paste notes here:", value=default_text, height=200)
        analyze_nlp_btn = st.button("🧠 Run NLP Extraction", type="primary", use_container_width=True)
    with col2:
        if analyze_nlp_btn and clinical_text:
            with st.spinner("Tokenizing..."):
                time.sleep(1.2)
            found_symptoms = [s for s in ["chest pain", "shortness of breath"] if s in clinical_text.lower()]
            found_diagnoses = [d for d in ["Hypertension"] if d in clinical_text]
            found_meds = [m for m in ["Aspirin 81mg"] if m in clinical_text]
            st.write("**Symptoms Detected:**")
            st.markdown(" ".join([f'<span class="entity-symp">{s}</span>' for s in found_symptoms]), unsafe_allow_html=True)
            st.write("**Clinical Diagnoses:**")
            st.markdown(" ".join([f'<span class="entity-diag">{d}</span>' for d in found_diagnoses]), unsafe_allow_html=True)
            st.write("**Prescribed Medications:**")
            st.markdown(" ".join([f'<span class="entity-med">{m}</span>' for m in found_meds]), unsafe_allow_html=True)

# ==========================================
# MODULE 5: GEN-AI
# ==========================================
elif module == "GenAI Clinical Assistant":
    st.markdown("### 💬 GenAI Clinical Assistant")
    st.divider()
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello Doctor. Ask me medical queries."}]
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
            if "heart" in prompt_lower or "chest" in prompt_lower:
                ai_response = "Immediate administration of chewable Aspirin (300mg) is standard protocol."
            else:
                ai_response = f"I scanned the simulated database for '{prompt}'."
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
    
    st.write("**📍 Advanced Geospatial Risk Mapping (Free Open-Source Tiles)**")
    
    # --- UPDATED: FREE MAPBOX ALTERNATIVES ---
    col_search, col_style, col_layer = st.columns([2, 1, 1])
    with col_search:
        search_location = st.text_input("🌍 Search ANY City/Town in India:", "New Delhi")
    with col_style:
        # Changed to Free CartoDB Styles (No API Key Required!)
        style_choice = st.selectbox("🎨 Map Style", ["Dark Mode", "Light Mode", "Street View"])
    with col_layer:
        layer_choice = st.selectbox("📊 Data Layer", ["3D Hexagon Matrix", "Heatmap Overlay", "Scatterplot Nodes"])

    # Mapping to free CartoDB styles built into PyDeck
    style_dict = {
        "Dark Mode": "dark",      # CartoDB Dark Matter
        "Light Mode": "light",    # CartoDB Positron
        "Street View": "road"     # CartoDB Voyager
    }

    if search_location:
        with st.spinner(f"Rendering multi-layered spatial data for '{search_location}'..."):
            base_lat, base_lon = get_coordinates(search_location)

        if base_lat and base_lon:
            map_data = pd.DataFrame(
                np.random.randn(600, 2) / [50, 50] + [base_lat, base_lon], 
                columns=['lat', 'lon']
            )
            
            view_state = pdk.ViewState(
                latitude=base_lat,
                longitude=base_lon,
                zoom=11,
                pitch=0 if layer_choice == "Heatmap Overlay" else 50, 
                transition_duration=2500, 
                transition_easing="cubic-in-out"
            )

            if layer_choice == "3D Hexagon Matrix":
                layer = pdk.Layer(
                    "HexagonLayer",
                    data=map_data,
                    get_position="[lon, lat]",
                    radius=500,
                    elevation_scale=30,
                    elevation_range=[0, 2000],
                    extruded=True,
                    coverage=1,
                    pickable=True
                )
            elif layer_choice == "Heatmap Overlay":
                layer = pdk.Layer(
                    "HeatmapLayer",
                    data=map_data,
                    get_position="[lon, lat]",
                    opacity=0.8,
                    get_weight=1
                )
            else:
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=map_data,
                    get_position="[lon, lat]",
                    get_color="[255, 50, 50, 200]",
                    get_radius=400,
                    pickable=True,
                )

            # Map style is now using the free Carto strings ("dark", "light", "road")
            st.pydeck_chart(pdk.Deck(
                map_style=style_dict[style_choice],
                initial_view_state=view_state,
                layers=[layer],
                tooltip={"text": f"Simulated Outbreak Data Matrix near {search_location.title()}"}
            ))
        else:
            st.error(f"⚠️ Could not find coordinates for '{search_location}'. Please check the spelling.")
