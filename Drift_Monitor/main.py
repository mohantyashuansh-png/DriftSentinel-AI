from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
import base64
import os
import datetime
from ultralytics import YOLO

# --- 1. SETUP ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. MODEL LOADING ---
current_dir = os.path.dirname(os.path.abspath(__file__))
custom_model_path = os.path.join(current_dir, "..", "models", "best.pt")
fallback_model_path = os.path.join(current_dir, "..", "yolov8n.pt")

yolo_model = None
print("\n🔍 SYSTEM STARTUP...")
if os.path.exists(custom_model_path):
    print(f"✅ LOADING CUSTOM MODEL: {custom_model_path}")
    try:
        yolo_model = YOLO(custom_model_path)
    except:
        yolo_model = YOLO('yolov8n.pt')
else:
    print(f"⚠️ Using Standard Model: {fallback_model_path}")
    yolo_model = YOLO('yolov8n.pt')

# --- 3. THE BRAIN (Drift Simulator) ---
class DriftSimulator:
    def __init__(self):
        self.drift_score = 0.0
        self.risk_budget = 100.0
        self.risk_level = "LOW"
        self.persistence_counter = 0
        self.baseline_bright = 50.0 
        self.baseline_blur = 10.0
        self.baseline_quality = 1.0
        self.logs = [] # <--- MEMORY STORAGE

    def update(self, q_flag, r_blur, r_bright):
        # 1. Risk Calc
        sim_risk = max(0.0, (self.baseline_quality - q_flag) * 100.0)
        
        real_risk = 0.0
        if r_bright < self.baseline_bright: 
            real_risk = (self.baseline_bright - r_bright) * 1.5
        if r_blur < self.baseline_blur: 
            real_risk = max(real_risk, (self.baseline_blur - r_blur) * 2)
            
        target = max(sim_risk, real_risk)
        self.drift_score += (target - self.drift_score) * 0.2
        
        # 2. Update Status & Logs
        previous_level = self.risk_level
        
        if self.drift_score > 60: 
            self.risk_level = "CRITICAL"
            self.risk_budget -= 0.5 
            # Log Critical Events
            if previous_level != "CRITICAL" or len(self.logs) == 0 or (datetime.datetime.now().second % 5 == 0):
                self.add_log("CRITICAL", "System Lockdown Initiated", "Visual Degradation / Sensor Blockage")
        elif self.drift_score > 30: 
            self.risk_level = "High"
            self.risk_budget -= 0.1
            if previous_level != "High":
                self.add_log("WARNING", "Drift Threshold Exceeded", "Environmental Fog/Blur Detected")
        else: 
            self.risk_level = "LOW"
            self.risk_budget += 0.05 
            
        self.risk_budget = max(0.0, min(100.0, self.risk_budget))

    def add_log(self, severity, action, cause):
        # Prevent spamming the same log every millisecond
        if self.logs and self.logs[-1]["action"] == action and self.logs[-1]["timestamp"] == datetime.datetime.now().strftime("%H:%M:%S"):
            return

        self.logs.insert(0, {
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "severity": severity,
            "action": action,
            "root_cause": cause
        })
        # Keep only last 50 logs
        if len(self.logs) > 50: self.logs.pop()

    def calibrate(self, blur, bright, quality):
        self.risk_budget = 100.0
        self.drift_score = 0.0
        self.risk_level = "LOW"
        self.baseline_bright = max(10.0, bright - 40.0) 
        self.baseline_blur = max(5.0, blur - 30.0)
        self.baseline_quality = quality
        
        self.add_log("INFO", "System Re-Calibrated", "Manual Operator Override")
        print(f"✅ CALIBRATED")

sim = DriftSimulator()
latest_blur = 100.0
latest_bright = 150.0
latest_quality = 1.0

# --- 4. ENDPOINTS ---

@app.post("/process-frame")
async def process_frame(file: UploadFile = File(...), quality_flag: float = 1.0):
    global latest_blur, latest_bright, latest_quality
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None: return {"status": "error"}

        # A. YOLO
        frame_small = cv2.resize(frame, (320, 240))
        yolo_base64 = ""
        if yolo_model:
            results = yolo_model(frame_small, verbose=False)
            annotated_frame = results[0].plot()
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            yolo_base64 = base64.b64encode(buffer).decode('utf-8')

        # B. Drift
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        bright = np.mean(gray)
        blur = cv2.Laplacian(frame_small, cv2.CV_64F).var()
        
        latest_blur = blur
        latest_bright = bright
        latest_quality = quality_flag
        
        sim.update(quality_flag, blur, bright)

        return {
            "status": "processed",
            "current_drift": sim.drift_score,
            "risk": sim.risk_level,
            "risk_budget": sim.risk_budget,
            "yolo_image": f"data:image/jpeg;base64,{yolo_base64}"
        }
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"status": "error"}

@app.get("/status")
async def get_status():
    return {
        "risk_level": sim.risk_level,
        "global_drift_score": sim.drift_score,
        "risk_budget": sim.risk_budget,
        "model_confidence": f"{max(45, int(98 - sim.drift_score/2))}% (Real-Time)"
    }

@app.post("/calibrate")
async def calibrate():
    sim.calibrate(latest_blur, latest_bright, latest_quality)
    return {"message": "Recalibrated"}

# --- FIX FOR LOGS & EXPLAINABILITY ---
@app.get("/logs")
async def get_logs():
    # Return the actual list from memory
    return {"logs": sim.logs}

@app.get("/explainability")
async def get_ex():
    score = sim.drift_score
    
    # Dynamic Explanation based on score
    if score < 20:
        return {
            "top_driving_feature": "None", 
            "operator_message": "System operating within normal parameters.", 
            "all_feature_scores": {"Helmet": 0.02, "Vest": 0.01, "Blur": 0.05}
        }
    
    # If High Drift
    return {
        "top_driving_feature": "Visual_Degradation (Real-Time)",
        "operator_message": "CRITICAL: Sensor Obstruction or Fog Detected.",
        "all_feature_scores": {
            "Visual_Degradation": min(0.98, score / 90),
            "Vest_Visibility": min(0.85, score / 110),
            "Background_Noise": 0.3
        }
    }

@app.get("/forecast")
async def get_forecast(): return {"persistence_counter": 0, "retraining_needed": False}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
