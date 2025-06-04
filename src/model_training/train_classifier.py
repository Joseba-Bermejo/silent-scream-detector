# src/model_training/train_classifier.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

def train_model(feature_csv="data/features.csv", model_path="models/distress_rf.pkl"):
    # Load features
    df = pd.read_csv(feature_csv)

    # TODO: Manually label a few rows first!
    if "label" not in df.columns:
        print("[ERROR] Please add a 'label' column with values: 1 = distress, 0 = neutral")
        return

    # Drop rows with missing/incomplete features
    df = df.dropna()
    df = df[df["eye_openness"] != -1]

    # Prepare data
    X = df[["eye_openness", "mouth_openness", "shoulder_tilt", "hand_face_dist"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    preds = clf.predict(X_test)
    print("\n[RESULTS] Model Performance:\n")
    print(classification_report(y_test, preds))

    # Save
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"[INFO] Model saved to {model_path}")