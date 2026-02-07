import streamlit as st
import cv2
import numpy as np
import time
from ultralytics import YOLO
from sentinel_core import DriftMonitor
import tempfile
import os
from datetime import datetime
from collections import deque
import pandas as pd
import altair as alt

# --- 1. PAGE CONFIG (Must be first) ---
st.set_page_config(page_title="Sentinel AI | Enterprise", layout="wide", page_icon="🛡️")

# --- 2. SESSION STATE INITIALIZATION ---
if 'baseline_loss' not in st.session_state:
    st.session_state['baseline_loss'] = 0.0
if 'is_calibrated' not in st.session_state:
    st.session_state['is_calibrated'] = False
if 'logs' not in st.session_state:
    st.session_state['logs'] = []
if 'force_recalibrate' not in st.session_state:
    st.session_state['force_recalibrate'] = False

# --- 3. CUSTOM CSS ---
st.markdown("""
<style>
    .stMetric { background-color: #0E1117; border: 1px solid #262730; }
</style>
""", unsafe_allow_html=True)

def add_log(message, type="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    icon = "ℹ️" if type == "INFO" else "🚨" if type == "ALERT" else "✅"
    st.session_state['logs'].insert(0, f"{timestamp} {icon} {message}")

# --- 4. SIDEBAR CONTROLS ---
st.sidebar.title("🛡️ Sentinel Command")
input_source = st.sidebar.radio("Video Feed Source", ["Upload Video", "Webcam"])

st.sidebar.markdown("---")
def calibrate():
    st.session_state['force_recalibrate'] = True

st.sidebar.button("🎯 Calibrate Baseline", on_click=calibrate)

drift_threshold = st.sidebar.slider("Anomaly Threshold", 0.5, 10.0, 3.0)
strict_mode = st.sidebar.checkbox("Strict Safety Mode", value=True)

st.sidebar.markdown("---")
st.sidebar.write("**Live Event Log**")
log_container = st.sidebar.container(height=200)
for log in st.session_state['logs']:
    log_container.text(log)

# --- 5. LOAD MODELS ---
@st.cache_resource
def load_models():
    path_to_yolo = 'models/best.pt' if os.path.exists('models/best.pt') else 'yolov8n.pt'
    yolo = YOLO(path_to_yolo) 
    
    if not os.path.exists("models/sentinel_model.pth"):
        st.error("FATAL: 'sentinel_model.pth' not found.")
        st.stop()
        
    monitor = DriftMonitor("models/sentinel_model.pth")
    return yolo, monitor

try:
    yolo_model, drift_monitor = load_models()
except Exception as e:
    st.error(f"FATAL: {e}")
    st.stop()

# --- 6. UPLOADER ---
temp_file_path = None
if input_source == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload CCTV Footage", type=["mp4", "avi"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        temp_file_path = tfile.name

# --- 7. MAIN LAYOUT ---
st.markdown("## 🏭 Industrial AI Reliability Monitor")
m1, m2, m3, m4 = st.columns(4)
with m1: status_metric = st.empty()
with m2: score_metric = st.empty()
with m3: obj_metric = st.empty()
with m4: action_metric = st.empty()

col_video, col_graph = st.columns([1.8, 1.2])
with col_video: video_placeholder = st.empty()
with col_graph: 
    st.write("**Real-time Drift Signature**")
    chart_placeholder = st.empty() 

# --- 8. RUN LOGIC ---
run_system = st.toggle("🚀 Activate Sentinel System", value=False)

if run_system:
    if input_source == "Webcam": cap = cv2.VideoCapture(0)
    elif temp_file_path: cap = cv2.VideoCapture(temp_file_path)
    else: st.warning("Waiting for video source..."); st.stop()

    smoothing_buffer = deque(maxlen=30) 
    graph_data = deque(maxlen=50) 

    while cap.isOpened() and run_system:
        ret, frame = cap.read()
        if not ret: break

        # A. CORE LOGIC
        raw_loss = drift_monitor.get_drift_score(frame)

        # B. CALIBRATION HANDLING
        if st.session_state.get('force_recalibrate', False):
            st.session_state['baseline_loss'] = raw_loss
            st.session_state['force_recalibrate'] = False 
            st.session_state['is_calibrated'] = True
            smoothing_buffer.clear()
            add_log(f"Baseline calibrated to {raw_loss:.2f}", "SUCCESS")
            st.toast("System Calibrated", icon="🎯")

        instant_drift = max(0.0, raw_loss - st.session_state['baseline_loss'])
        smoothing_buffer.append(instant_drift)
        smoothed_drift = sum(smoothing_buffer) / len(smoothing_buffer) if smoothing_buffer else instant_drift

        # C. FUNCTIONAL CHECK (YOLO)
        results = yolo_model(frame, verbose=False)
        annotated_frame = results[0].plot()
        
        worker_count = 0
        for box in results[0].boxes:
            if int(box.cls[0]) == 0: 
                worker_count += 1
        
        penalty = 0.0
        if strict_mode and st.session_state['is_calibrated'] and worker_count == 0:
            penalty = 10.0 

        final_score = smoothed_drift + penalty
        graph_data.append(final_score)

        # D. DECISION
        is_alarm = final_score > drift_threshold

        if is_alarm:
            status_metric.metric("System Status", "CRITICAL", delta="FAILURE DETECTED", delta_color="inverse")
            score_metric.metric("Drift Score", f"{final_score:.2f}", delta=f"+{final_score-drift_threshold:.1f} High", delta_color="inverse")
            action_metric.metric("Auto-Response", "RE-ROUTING", "Triggering Backup")
            
            cv2.rectangle(annotated_frame, (0,0), (annotated_frame.shape[1], annotated_frame.shape[0]), (0,0,255), 30)
            cv2.putText(annotated_frame, "DATA CORRUPTED", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 4)
             
            if not st.session_state['logs'] or "CRITICAL" not in st.session_state['logs'][0]:
                add_log(f"Drift Spike: {final_score:.2f}", "ALERT")
        else:
            status_metric.metric("System Status", "NOMINAL", delta="Optimal")
            score_metric.metric("Drift Score", f"{final_score:.2f}", delta="Stable", delta_color="normal")
            action_metric.metric("Auto-Response", "IDLE", "Monitoring...")

        obj_metric.metric("Workers Detected", f"{worker_count}")

        # E. RENDER VIDEO
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # F. RENDER CHART
        df = pd.DataFrame({"Frame": range(len(graph_data)), "Drift Score": list(graph_data)})
        
        base = alt.Chart(df).encode(
            x=alt.X('Frame', axis=None),
            y=alt.Y('Drift Score', scale=alt.Scale(domain=[0, 15])) 
        )
        area = base.mark_area(
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='#00FFAA', offset=0),
                       alt.GradientStop(color='rgba(0, 255, 170, 0.1)', offset=1)],
                x1=1, x2=1, y1=1, y2=0
            ), opacity=0.5
        )
        line = base.mark_line(color='#00FFAA', strokeWidth=3)
        chart_placeholder.altair_chart((area + line).properties(height=200), use_container_width=True)

    # --- THE SILENCER (Final Fix for PermissionError) ---
    cap.release()
    
    # Wait for Windows to release file handle
    time.sleep(0.2)
    
    if temp_file_path:
        try:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        except PermissionError:
            pass # IGNORE the error if Windows locks the file
        except Exception:
            pass # IGNORE any other cleanup errors