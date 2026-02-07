from drift_engine import DriftEngine
from data_simulator import get_reference_data, get_drifted_data

# 1. Load Training Data
ref_data = get_reference_data()
engine = DriftEngine(ref_data)

# 2. Simulate High Drift
print("--- TESTING HIGH DRIFT ---")
curr_data = get_drifted_data(severity='high')

report, score = engine.check_data_drift(curr_data)
print(f"Global Risk Score: {score:.2f}/100")
print(f"Age Drifted? {report['Age']['drift_detected']}")
print(f"Income Drifted? {report['Income']['drift_detected']}")

# 3. Test Fingerprint
fp = engine.get_drift_fingerprint(report)
print(f"Drift Signature: {fp}")