import pandas as pd
import numpy as np

def get_reference_data(n=1000):
    """Generates perfect baseline data."""
    np.random.seed(42)
    data = pd.DataFrame({
        'Helmet_Conf': np.random.normal(0.92, 0.05, n),
        'Vest_Conf': np.random.normal(0.88, 0.06, n),
        'Harness_Conf': np.random.normal(0.90, 0.04, n),
    })
    data = data.clip(0.0, 1.0)
    data['Camera_Zone'] = np.random.choice(['Zone_Entry', 'Zone_Mining'], n)
    return data

def get_drifted_data(n=1000, quality=1.0):
    """
    Generates data based on Environmental Quality (0.0 = Bad, 1.0 = Perfect).
    The math reacts to the slider.
    """
    np.random.seed(99)
    
    # Base Confidence (Perfect Scenario)
    base_helmet = 0.92
    base_vest = 0.88
    base_harness = 0.90
    
    # DEGRADATION MATH:
    # As quality drops, the Mean Confidence drops AND Variance increases (Model gets confused)
    # If quality is 0.5 (Foggy), Helmet Conf drops to ~0.55
    
    degradation_factor = quality  # Direct mapping for simplicity
    
    data = pd.DataFrame({
        'Helmet_Conf': np.random.normal(base_helmet * degradation_factor, 0.05 + (1-quality)*0.2, n),
        'Vest_Conf': np.random.normal(base_vest * degradation_factor, 0.06 + (1-quality)*0.2, n),
        'Harness_Conf': np.random.normal(base_harness * degradation_factor, 0.04 + (1-quality)*0.2, n),
    })
    
    data = data.clip(0.0, 1.0)
    data['Camera_Zone'] = np.random.choice(['Zone_Entry', 'Zone_Mining'], n)
    
    return data