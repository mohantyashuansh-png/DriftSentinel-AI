from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
import datetime

class DriftSimulator:
    def __init__(self):
        self.drift_score = 0.0
        self.risk_budget = 100.0
        self.risk_level = "LOW"
        self.persistence_counter = 0

    def update(self, quality_flag: float, blur: float, brightness: float, contrast: float):
        # 1. Slider Risk
        simulated_risk = (1.0 - quality_flag) * 100.0

        # 2. Real Video Risk
        real_risk = 0.0
        
        # --- THE FIX: CONTRAST DETECTION ---
        # A normal room has contrast > 40.
        # A covered lens (even with red noise) is "flat", usually contrast < 20.
        if contrast < 20: 
            real_risk = 100.0
        
        # Fallback: Absolute Darkness
        elif brightness < 50:
            real_risk = 100.0
            
        # Fallback: Extreme Blur (only if bright enough to see)
        elif blur < 15 and brightness > 50:
             real_risk = (30 - blur) * 3

        # 3. Hybrid Logic
        target_drift = max(simulated_risk, real_risk)
        
        self.drift_score += (target_drift - self.drift_score) * 0.4 # Faster reaction
        
        # Thresholds
        if self.drift_score > 60:
            self.risk_level = "CRITICAL"
            self.persistence_counter += 1
            self.risk_budget -= 0.5 
        elif self.drift_score > 30:
            self.risk_level = "High"
            self.persistence_counter += 1
            self.risk_budget -= 0.1
        else:
            self.risk_level = "LOW"
            self.persistence_counter = max(0, self.persistence_counter - 1)
            self.risk_budget += 0.05

        self.risk_budget = max(0.0, min(100.0, self.risk_budget))
        
    def calibrate(self):
        self.risk_budget = 100.0
        self.drift_score = 0.0
        self.risk_level = "LOW"
        self.persistence_counter = 0

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

simulator = DriftSimulator()

@app.get("/")
def home():
    return {"message": "Sentinel AI Contrast Monitor Online"}

@app.post("/process-frame")
async def process_frame(file: UploadFile = File(...), quality_flag: float = 1.0):
    # 1. READ
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. PROCESS (Small Scale)
    frame_small = cv2.resize(frame, (320, 240))
    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

    # 3. METRICS
    # A. Brightness (Mean)
    mean_brightness = np.mean(gray)
    
    # B. Contrast (Standard Deviation) -> THIS KILLS THE NOISE ISSUE
    contrast = np.std(gray)
    
    # C. Blur (Laplacian)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # PRINT DEBUG (Watch your terminal to calibrate!)
    print(f"👁️  Bright: {mean_brightness:.1f} | Contrast: {contrast:.1f} | Sharp: {blur_score:.1f}")

    # 4. UPDATE
    simulator.update(quality_flag, blur_score, mean_brightness, contrast)
    
    return {
        "status": "processed",
        "current_drift": simulator.drift_score,
        "risk": simulator.risk_level
    }

@app.get("/status")
async def get_status():
    return {
        "risk_level": simulator.risk_level,
        "global_drift_score": simulator.drift_score,
        "risk_budget": simulator.risk_budget,
        "model_confidence": f"{max(45, int(98 - simulator.drift_score/2))}% (Real-Time)"
    }

@app.get("/forecast")
async def get_forecast():
    return {
        "persistence_counter": simulator.persistence_counter,
        "retraining_needed": simulator.drift_score > 80
    }

@app.get("/explainability")
async def get_explainability():
    score = simulator.drift_score
    if score < 20:
        return {
            "top_driving_feature": "None",
            "operator_message": "System operating within normal parameters.",
            "all_feature_scores": {"Helmet": 0.02, "Vest": 0.01, "Blur": 0.05}
        }
    return {
        "top_driving_feature": "Visual_Degradation (Real-Time)",
        "operator_message": "CRITICAL: Sensor Obstruction or Environmental Blur Detected.",
        "all_feature_scores": {
            "Visual_Degradation": min(0.98, score / 90),
            "Vest_Visibility": min(0.85, score / 110),
            "Helmet_Feature_Map": 0.15,
            "Background_Noise": 0.3
        }
    }

@app.post("/calibrate")
async def calibrate():
    simulator.calibrate()
    return {"message": "Baseline Recalibrated"}

@app.get("/logs")
async def get_logs():
    logs = []
    now = datetime.datetime.now().strftime("%H:%M:%S")
    if simulator.drift_score > 60:
        logs.append({"timestamp": now, "severity": "CRITICAL", "action_taken": "Lockdown", "root_cause": "Visual Drift Threshold Exceeded"})
    elif simulator.drift_score > 30:
        logs.append({"timestamp": now, "severity": "WARNING", "action_taken": "Alert Sent", "root_cause": "Minor Distribution Shift"})
    else:
         logs.append({"timestamp": now, "severity": "INFO", "action_taken": "Routine Check", "root_cause": "System Nominal"})
    return {"logs": logs}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
