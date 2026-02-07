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
        
        # --- ADAPTIVE BASELINES ---
        # Instead of hard numbers (110, 40), we compare against these.
        # When you calibrate, we update these to match the current environment.
        self.baseline_brightness_threshold = 110.0 
        self.baseline_blur_threshold = 40.0

    def update(self, quality_flag: float, real_blur_score: float, brightness: float):
        # 1. Slider Risk (Simulation)
        simulated_risk = (1.0 - quality_flag) * 100.0

        # 2. Real Video Risk (Reality) - Now ADAPTIVE
        real_risk = 0.0
        
        # LOGIC: Only trigger if it is WORSE than the calibrated baseline
        # If baseline is 20 (Dark), and current is 20, Risk = 0.
        
        if brightness < self.baseline_brightness_threshold: 
            # Risk scales based on how much darker it is than the baseline
            diff = self.baseline_brightness_threshold - brightness
            real_risk = diff * 1.5 
        
        if real_blur_score < self.baseline_blur_threshold:
            diff = self.baseline_blur_threshold - real_blur_score
            real_risk = max(real_risk, diff * 2) 
            
        # 3. Take the Worst Case
        target_drift = max(simulated_risk, real_risk)
        
        # Smoothing
        self.drift_score += (target_drift - self.drift_score) * 0.5
        
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
        
    def calibrate(self, current_blur, current_bright):
        """
        The Magic Fix: Learn the NEW Normal.
        """
        self.risk_budget = 100.0
        self.drift_score = 0.0
        self.risk_level = "LOW"
        self.persistence_counter = 0
        
        # If it's dark/blurry right now, accept it as the new baseline!
        # We lower the thresholds so the current state is considered "Safe".
        # We subtract a small buffer (e.g. -10) so it doesn't trigger immediately.
        self.baseline_brightness_threshold = max(10, current_bright - 10)
        self.baseline_blur_threshold = max(10, current_blur - 10)
        
        print(f"✅ CALIBRATED! New Baselines -> Brightness: {self.baseline_brightness_threshold:.1f} | Blur: {self.baseline_blur_threshold:.1f}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

simulator = DriftSimulator()

# Global variables to store latest frame stats for calibration
latest_blur = 100.0
latest_bright = 150.0

@app.get("/")
def home():
    return {"message": "Sentinel AI Adaptive Monitor Online"}

@app.post("/process-frame")
async def process_frame(file: UploadFile = File(...), quality_flag: float = 1.0):
    global latest_blur, latest_bright
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None: return {"status": "error"}

        frame_small = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        mean_brightness = np.mean(gray)
        gray_clean = cv2.GaussianBlur(gray, (5, 5), 0)
        blur_score = cv2.Laplacian(gray_clean, cv2.CV_64F).var()

        # Update globals so calibrate() can use them
        latest_blur = blur_score
        latest_bright = mean_brightness

        # DEBUG PRINT
        print(f"👀 CAM: B={mean_brightness:.0f} (Base {simulator.baseline_brightness_threshold:.0f}) | S={blur_score:.0f} (Base {simulator.baseline_blur_threshold:.0f}) | Risk={simulator.drift_score:.0f}")

        simulator.update(quality_flag, blur_score, mean_brightness)
        
        return {
            "status": "processed",
            "current_drift": simulator.drift_score,
            "risk": simulator.risk_level
        }
    except Exception as e:
        print(f"Error: {e}")
        return {"status": "error"}

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
    # Use the LATEST known values to set the new baseline
    simulator.calibrate(latest_blur, latest_bright)
    return {"message": "Baseline Recalibrated"}

@app.get("/logs")
async def get_logs():
    logs = []
    now = datetime.datetime.now().strftime("%H:%M:%S")
    
    if simulator.drift_score > 60:
        logs.append({
            "timestamp": now,
            "severity": "CRITICAL",
            "action_taken": "Lockdown Initiated",
            "root_cause": "Visual Drift Threshold Exceeded"
        })
    elif simulator.drift_score > 30:
        logs.append({
            "timestamp": now,
            "severity": "WARNING",
            "action_taken": "Alert Sent to Supervisor",
            "root_cause": "Minor Distribution Shift"
        })
    else:
         logs.append({
            "timestamp": now,
            "severity": "INFO",
            "action_taken": "Routine Check",
            "root_cause": "System Nominal"
        })
    return {"logs": logs}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
