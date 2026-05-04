import streamlit as st
import pandas as pd
from agents.swarm_logic import DiagnosticSwarm 
from dashboard.icu_live import render_icu_dashboard
# Import ImageDraw and ImageFont for drawing pointers and labels on image
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
    "GenAI Clinical Assistant"
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
                
                # --- Step 2: Render Stylized AI Attention Map ---
                # Convert to grayscale, enhance contrast, and blur to create a heatmap base
                gray_image = image.convert("L")
                enhancer = ImageEnhance.Contrast(gray_image)
                contrast_image = enhancer.enhance(3.0) 
                blurred_gray = contrast_image.filter(ImageFilter.GaussianBlur(radius=10))

                # Colorize the heatmap (blue to yellow scale) and blend with original image
                heatmap = ImageOps.colorize(blurred_gray, mid="blue", black="black", white="yellow")
                blended = Image.blend(image.convert("RGB"), heatmap, alpha=0.5)

                # --- NEW LOGIC: Drawing Arrow and Label if Anomaly Detected (Simulation) ---
                if prob > 0.5:
                    st.write("**AI Diagnostic Localization (Simulated Pointer):**")
                    
                    # Create copy of blended image to draw on
                    annotated_img = blended.copy()
                    draw = ImageDraw.Draw(annotated_img)
                    w, h = annotated_img.size

                    # --- Define simulated anomaly coordinates (Upper Lung Area, Viewer's left) ---
                    # These are arbitrary coordinates for demonstration.
                    # Bounding box coordinates for a circle [x0, y0, x1, y1]
                    circle_coords = [w * 0.25, h * 0.35, w * 0.40, h * 0.50]
                    anomaly_target = (w * 0.325, h * 0.425) # Center of circle for arrow to point to
                    pointer_start = (w * 0.325, h * 0.25) # Starting point of arrow (above)

                    # 1. Draw Red Pointer Arrow
                    # Draw main line
                    draw.line([pointer_start, anomaly_target], fill="red", width=10)
                    # Draw arrowhead (simple V shape)
                    draw.line([anomaly_target, (anomaly_target[0]-w*0.02, anomaly_target[1]-h*0.03)], fill="red", width=10)
                    draw.line([anomaly_target, (anomaly_target[0]+w*0.02, anomaly_target[1]-h*0.03)], fill="red", width=10)

                    # 2. Draw Red Circle around anomaly area
                    draw.ellipse(circle_coords, outline="red", width=5)
                    
                    # 3. Add Text Label ("Anomaly Detected")
                    # Try to load default font. NOTE: Text size control is limited without custom .ttf files.
                    try:
                        font = ImageFont.load_default()
                        # Default font is tiny, drawing text larger via Pillow is complex across platforms.
                        # We just draw the pointer for better visibility and explain in findings expander.
                        # draw.text((w*0.1, h*0.1), "POTENTIAL OPACITY", fill="red", font=font)
                    except Exception:
                        pass # Skip font drawing if it fails
                    
                    # Render final annotated image
                    st.image(annotated_img, use_container_width=True, caption="Grad-CAM Simulation with Simulated Anomaly Pointer")
                
                else:
                    st.write("**AI Attention Map (Simulated Deep Feature Map):**")
                    # Render standard blended heatmap for safe scan
                    st.image(blended, use_container_width=True, caption="Grad-CAM Visualization (Normal Findings Simulated)")


                st.markdown("#### 📋 Detailed AI Findings")
                with st.expander("Expand Radiological Breakdown", expanded=True):
                    if prob > 0.5:
                        st.write("- 🔴 **Finding:** Potential opacity/consolidation detected in highlighted zone.")
                        st.info("💡 **Explainable AI Note:** The simulated Grad-CAM heatmap shows concentrated high activation (yellow) in this region, triggered anomaly alert. The AI has drawn a red arrow pointing to this area.")
                    else:
                        st.write("- 🟢 **Lung Fields:** Clear bilaterally.")
                        st.info("💡 **Explainable AI Note:** Uniform heatmap distribution suggests normal anatomy based on current training data.")

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
# MODULE 5: GEN-AI CLINICAL ASSISTANT (LLM/RAG)
# ==========================================
elif module == "GenAI Clinical Assistant":
    st.markdown("### 💬 GenAI Clinical Assistant")
    st.write("Simulation of a Retrieval-Augmented Generation (RAG) agent querying medical guidelines.")
    st.warning("⚠️ **Disclaimer:** Simulation for demonstration. Not for actual medical use.")
    st.divider()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello Doctor. I am your AI Clinical Assistant. How can I help you today?"}
        ]

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
            if "hypertension" in prompt_lower or "bp" in prompt_lower:
                ai_response = "Typically involves Thiazide diuretics, Calcium channel blockers (CCBs), or ACE inhibitors. Lifestyle modifications are highly recommended. \n\n*Source Simulation: AHA Guidelines.*"
            elif "diabetes" in prompt_lower or "sugar" in prompt_lower:
                ai_response = "Metformin is generally the first-line treatment. Continual A1C monitoring every 3-6 months is advised. \n\n*Source Simulation: ADA Standards.*"
            else:
                ai_response = f"I scanned the simulated database for '{prompt}'. A fully deployed RAG architecture would fetch specific peer-reviewed content here."
            
            for chunk in ai_response.split(" "):
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
