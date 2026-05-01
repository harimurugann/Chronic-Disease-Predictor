# dashboard/icu_live.py
import streamlit as st
import numpy as np
import time
import plotly.graph_objs as go

def render_icu_dashboard():
    st.markdown("### 🫀 ICU Live Vitals Stream (IoT Simulation)")
    st.write("Simulating real-time edge AI data streaming from patient monitors.")
    
    start_btn = st.button("▶️ Start Live Stream", type="primary", use_container_width=True)
    
    if start_btn:
        # Create empty placeholders for live updating
        col1, col2, col3 = st.columns(3)
        hr_metric = col1.empty()
        bp_metric = col2.empty()
        o2_metric = col3.empty()
        
        chart_placeholder = st.empty()
        
        # Initialize historical data for the chart
        hr_data = list(np.random.normal(80, 2, 30))
        o2_data = list(np.random.normal(98, 0.5, 30))
        
        st.toast("Connected to ICU Edge Device...", icon="📡")
        
        # Simulate real-time streaming for 30 seconds
        for i in range(30):
            # Generate fluctuating live data
            current_hr = int(np.random.normal(80, 6))
            current_sys = int(np.random.normal(120, 5))
            current_dia = int(np.random.normal(80, 3))
            current_o2 = min(100, int(np.random.normal(98, 1)))
            
            # Update chart history
            hr_data.append(current_hr)
            hr_data.pop(0)
            o2_data.append(current_o2)
            o2_data.pop(0)
            
            # Update Metrics UI
            hr_metric.metric("Heart Rate (BPM)", current_hr, delta=current_hr - 80, delta_color="inverse")
            bp_metric.metric("Blood Pressure (mmHg)", f"{current_sys}/{current_dia}")
            o2_metric.metric("Oxygen Level (SpO2 %)", current_o2, delta=current_o2 - 98)
            
            # Create Live Plotly Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=hr_data, mode='lines', name='Heart Rate', line=dict(color='#f5576c', width=3)))
            fig.add_trace(go.Scatter(y=o2_data, mode='lines', name='SpO2', line=dict(color='#2ecc71', width=3)))
            
            fig.update_layout(
                title="Continuous Vitals Timeline",
                template="plotly_dark",
                height=350,
                margin=dict(l=0, r=0, t=40, b=0),
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(range=[60, 110])
            )
            
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            
            # Pause for 1 second to simulate live streaming
            time.sleep(1)
            
        st.success("Simulation complete. The edge device connection has been safely closed.")