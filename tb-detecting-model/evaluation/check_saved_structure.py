import pickle
import os

MODEL_DIR = './saved_models'
ENSEMBLE_PATH = os.path.join(MODEL_DIR, 'tb_detector_v1_20260201_135222_ensemble.pkl')
FEATURES_PATH = os.path.join(MODEL_DIR, 'tb_detector_v1_20260201_135222_features.pkl')

print("Checking ensemble.pkl structure...")
if os.path.exists(ENSEMBLE_PATH):
    with open(ENSEMBLE_PATH, 'rb') as f:
        ensemble_data = pickle.load(f)
    print(f"ensemble.pkl keys: {ensemble_data.keys()}")

print("\nChecking features.pkl structure...")
if os.path.exists(FEATURES_PATH):
    with open(FEATURES_PATH, 'rb') as f:
        features_data = pickle.load(f)
    print(f"features.pkl keys: {features_data.keys() if isinstance(features_data, dict) else type(features_data)}")
    if isinstance(features_data, dict):
        for key in features_data.keys():
            try:
                print(f"  {key}: {type(features_data[key])}, shape={getattr(features_data[key], 'shape', 'N/A')}")
            except:
                print(f"  {key}: {type(features_data[key])}")
