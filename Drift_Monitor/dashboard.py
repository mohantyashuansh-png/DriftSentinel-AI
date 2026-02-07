import streamlit as st
import requests
import time

# TITLE
st.set_page_config(page_title="Sentinel PPE Monitor", layout="wide")
st.title("🛡️ Sentinel: Industrial Safety AI")

# SIDEBAR: CONTROLS (The "Inputs")
st.sidebar.header("⚙️ Supervisor Controls")

# 1. THE SLIDER (Organic Drift)
quality = st.sidebar.slider("Environment Visibility (Fog Level)", 0.0, 1.0, 1.0)
if st.sidebar.button("Update Environment"):
    requests.post(f"http://127.0.0.1:8000/set-environment?quality={quality}")
    st.sidebar.success(f"Environment set to {quality*100}%")

# 2. CALIBRATION BUTTON
if st.sidebar.button("🛠️ CALIBRATE BASELINE"):
    resp = requests.post("http://127.0.0.1:8000/calibrate")
    st.sidebar.success("System Re-Baselined!")

# MAIN DASHBOARD (The "Outputs")
st.subheader("Live Safety Status")

# Fetch Data from your API
try:
    response = requests.get("http://127.0.0.1:8000/status").json()
    
    # METRICS ROW
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Risk Budget (Fuel)", f"{response['risk_budget']}%", delta_color="normal")
    
    with col2:
        # Color logic for Risk Level
        risk = response['risk_level']
        color = "red" if risk == "CRITICAL" else "green"
        st.markdown(f"**Risk Level:** :{color}[{risk}]")
        
    with col3:
        st.metric("Model Confidence", "High" if response['risk_level'] == "Low" else "Low")

    # ACTION BANNER
    if response['risk_level'] != "Low":
        st.error(f"🚨 ACTION REQUIRED: {response['action_required']}")
    else:
        st.success(f"✅ SYSTEM STATUS: {response['action_required']}")

    # RAW DATA (For "Black Box" proof)
    with st.expander("Show Live Black Box Logs"):
        logs = requests.get("http://127.0.0.1:8000/logs").json()
        st.write(logs)

except Exception as e:
    st.error("⚠️ Backend is offline. Run 'uvicorn main:app --reload' first.")