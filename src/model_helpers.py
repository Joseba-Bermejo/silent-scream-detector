import os
import pandas as pd
from heapq import nlargest
from .slack_notifications import send_slack_alert_summary

import os
project_root = os.path.dirname(os.path.abspath(__file__))  # Points to /src

# One level up to reach project root
project_root = os.path.abspath(os.path.join(project_root, ".."))

def warn_if_high_confidence_slack(model, X_scaled, filenames, threshold=0.80):
    probs = model.predict_proba(X_scaled)[:, 1]
    preds = model.predict(X_scaled)

    distress_cases = [
        (conf, filenames[i])
        for i, (pred, conf) in enumerate(zip(preds, probs))
        if pred == 1 and conf > threshold
    ]

    for i, (conf, fname) in enumerate(distress_cases):
        print(f"[WARNING] Prediction {i}: DISTRESS in {fname} with confidence {conf:.2f}")

    if distress_cases:
        top_3 = nlargest(3, distress_cases, key=lambda x: x[0])
        send_slack_alert_summary(len(distress_cases), top_3)
    else:
        print("No high-confidence distress cases found.")


import numpy as np
import joblib
from src.extract_features import extract_features_from_single_image

# Build full paths to model files
model_path = os.path.join(project_root, "models", "rf_model.pkl")
scaler_path = os.path.join(project_root, "models", "scaler.pkl")

# Load model and scaler
rf_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

feature_order = [
    "eye_openness",
    "mouth_openness",
    "brow_furrow",
    "shoulder_tilt",
    "hand_face_dist",
    "head_tilt_angle",
    "wrist_distance",
    "torso_lean"
]