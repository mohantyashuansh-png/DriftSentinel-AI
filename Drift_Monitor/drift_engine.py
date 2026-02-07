import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, entropy
import shap
from sklearn.ensemble import RandomForestRegressor

class DriftEngine:
    def __init__(self, reference_data: pd.DataFrame):
        self.reference_data = reference_data
        self.numeric_features = reference_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # --- INNOVATION 1: RISK BUDGETING (The "Leaky Bucket") ---
        # "System maintains a fixed risk budget. Small drifts consume it."
        # If budget hits 0, SAFE MODE (Lockdown) is mandatory.
        self.risk_budget = 100.0  # Starting Fuel
        self.max_budget = 100.0
        self.refill_rate = 5.0    # Credits regained per cycle
        
        # --- INNOVATION 4: ATTRIBUTION TIMELINE ---
        # Stores the history of which feature caused drift over time.
        self.history = []

        # --- PPE DOMAIN WEIGHTS ---
        # "Formalize the risk score" -> Safety Hierarchy.
        self.feature_weights = {
            "Helmet_Conf": 0.40,  # CRITICAL: Falling objects = Fatality.
            "Harness_Conf": 0.40, # CRITICAL: Fall from height = Fatality.
            "Vest_Conf": 0.20,    # HIGH: Visibility important, but less immediate risk.
        }
        self.default_weight = 0.1

        # --- ROBUSTNESS UPGRADE: SMOOTHING (EMA) ---
        # Prevents the alarm from flickering on/off.
        self.ema_score = 0.0
        self.alpha = 0.7  # 0.7 = Fast reaction, 0.1 = Slow

        # Standard Thresholds
        self.thresholds = {
            col: self.reference_data[col].std() * 3 for col in self.numeric_features
        }

        # Initialize SHAP Logic
        self._init_shap_explainer()

    def _init_shap_explainer(self):
        try:
            # Predict Helmet Confidence based on other factors
            target_col = 'Helmet_Conf' if 'Helmet_Conf' in self.reference_data.columns else self.numeric_features[-1]
            feature_cols = [c for c in self.numeric_features if c != target_col]
            X = self.reference_data[feature_cols].fillna(0)
            y = self.reference_data[target_col].fillna(0)
            self.model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            self.model.fit(X, y)
            self.explainer = shap.TreeExplainer(self.model)
            self.feature_cols = feature_cols
            print("✅ PPE SHAP Explainer Ready")
        except Exception as e:
            print(f"⚠️ SHAP Init Failed: {e}")

    def check_data_drift(self, current_data: pd.DataFrame):
        drift_report = {}
        weighted_drift_sum = 0
        total_weight = 0

        for col in self.numeric_features:
            if col not in current_data.columns: continue
            
            # 1. Calculate KS Statistic (Distance)
            stat, p_value = ks_2samp(self.reference_data[col], current_data[col])
            is_drifted = p_value < 0.05
            
            # 2. Apply PPE Domain Weights
            weight = self.feature_weights.get(col, self.default_weight)
            risk_contribution = stat * weight if is_drifted else 0
            
            weighted_drift_sum += risk_contribution
            total_weight += weight

            drift_report[col] = {
                "drift_detected": is_drifted,
                "p_value": float(p_value),
                "distance": float(stat),
                "severity": "High" if stat > 0.3 else ("Medium" if stat > 0.1 else "Low")
            }

        # 3. Calculate Weighted Score
        raw_score = (weighted_drift_sum / total_weight) * 100 if total_weight > 0 else 0
        final_instant_score = min(100, raw_score * 2.0)

        # 4. Apply Smoothing
        self.ema_score = (self.alpha * final_instant_score) + ((1 - self.alpha) * self.ema_score)

        # 5. Manage Risk Budget
        cost = self.ema_score / 10.0
        self.risk_budget -= cost
        self.risk_budget += self.refill_rate
        if self.risk_budget > self.max_budget: self.risk_budget = self.max_budget
        if self.risk_budget < 0: self.risk_budget = 0.0

        return drift_report, round(self.ema_score, 2), round(self.risk_budget, 1)

    # --- INNOVATION 3: CONFIDENCE COLLAPSE (ENTROPY) ---
    def check_confidence_entropy(self, current_data):
        """
        If model confidence in Helmets drops to 0.5 (Guessing), Entropy Spikes.
        This detects 'Model Confusion' even if drift is subtle.
        """
        avg_conf = current_data.get('Helmet_Conf', pd.Series([1])).mean()
        # Simulate confidence based on the data
        simulated_confidence = max(0.1, min(0.99, avg_conf))
        
        # Entropy Formula
        p = simulated_confidence
        entr = entropy([p, 1-p], base=2)
        
        status = "High Confidence"
        if entr > 0.8: status = "Model Confused (Unsafe)"
        
        return {"entropy": round(entr, 3), "status": status}

    # --- FEATURE 2: PREDICTION DRIFT (PSI) ---
    def check_prediction_drift(self, ref_preds, curr_preds, buckets=10):
        def calculate_psi(expected, actual, buckets):
            breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
            breakpoints = np.percentile(expected, breakpoints)
            
            expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
            actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
            
            expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
            actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
            
            psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
            return psi_value

        try:
            psi = calculate_psi(ref_preds, curr_preds, buckets)
            status = "Stable"
            if psi > 0.1: status = "Warning"
            if psi > 0.2: status = "Critical"
            return {"psi": float(psi), "status": status}
        except Exception as e:
            return {"psi": 0.0, "status": "Error: " + str(e)}

    # --- FEATURE 3: SUBGROUP DRIFT ---
    def check_subgroup_drift(self, current_data, group_col):
        if group_col not in current_data.columns: return {}
        unique_groups = current_data[group_col].unique()
        subgroup_report = {}

        for group in unique_groups:
            sub_data = current_data[current_data[group_col] == group]
            if len(sub_data) < 5: continue 

            report, score, _ = self.check_data_drift(sub_data)
            if score > 40:
                subgroup_report[str(group)] = {"risk_score": score, "details": report}
        return subgroup_report

    # --- FEATURE 5: EXPLAINABILITY (SHAP) + INNOVATION 4 (TIMELINE) ---
    def check_feature_importance(self, current_data: pd.DataFrame):
        curr_X = current_data[self.feature_cols].fillna(0)
        sample = curr_X.sample(min(100, len(curr_X)), random_state=42)
        shap_values = self.explainer.shap_values(sample)
        importance_scores = np.abs(shap_values).mean(axis=0)
        importance_dict = dict(zip(self.feature_cols, importance_scores))
        
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        top_feat = sorted_importance[0][0]
        
        # Update History
        self.history.append({
            "time_step": len(self.history) + 1,
            "feature": top_feat,
            "impact_score": round(sorted_importance[0][1], 3)
        })
        if len(self.history) > 10: self.history.pop(0)

        return {
            "top_feature": top_feat,
            "importance_ranking": [x[0] for x in sorted_importance],
            "scores": importance_dict,
            "history": self.history
        }
    # --- INNOVATION: DYNAMIC RE-BASELINING (CALIBRATION) ---
    def update_baseline(self, new_reference_data: pd.DataFrame):
        """
        Real Calibration: Accepts the CURRENT data as the new "Normal".
        Recalculates all statistical thresholds based on this new reality.
        """
        print("🔄 RE-BASELINING SYSTEM...")
        
        # 1. Update the Reference Data
        self.reference_data = new_reference_data
        
        # 2. Recalculate Standard Deviation Thresholds
        # (e.g., If new environment is noisier, expand the safe thresholds)
        self.thresholds = {
            col: self.reference_data[col].std() * 3 for col in self.numeric_features
        }
        
        # 3. Refill the Risk Budget (The Leaky Bucket)
        self.risk_budget = self.max_budget
        self.ema_score = 0.0 # Reset smoothing history
        
        # 4. Re-Initialize SHAP (Because the baseline distribution changed)
        # We need to retrain the shadow model to understand the new "Normal" relationships
        self._init_shap_explainer()
        
        print("✅ SYSTEM CALIBRATED. New Baseline Established.")

    # --- INNOVATION 1: DRIFT SIGNATURE ---
    def get_drift_fingerprint(self, drift_report):
        fingerprint = []
        for col in self.numeric_features:
            if drift_report.get(col, {}).get("drift_detected", False):
                fingerprint.append(1)
            else:
                fingerprint.append(0)
        return fingerprint