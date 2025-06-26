import os
import cv2
import av
import numpy as np
import tempfile
import streamlit as st
import pandas as pd
import mediapipe as mp
import math
import joblib
import matplotlib.pyplot as plt
from slack_sdk import WebClient
from src.preprocessing import preprocess_image
from src.model_helpers import warn_if_high_confidence_slack
from src.slack_notifications import send_slack_alert_summary
from dotenv import load_dotenv
load_dotenv()
import io
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from PIL import Image


# === UI Enhancements ===
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1614853317094-072a63362e31");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    .main, .block-container {
        background-color: rgba(0, 0, 0, 0.0);
        padding: 2rem;
        border-radius: 10px;
    }

    section[data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.6);
        font-size: 1.1rem !important;
    }

    html, body, [class*="css"] {
        color: white;
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
    }

    h1, h2, h3, h4 {
        font-size: 2rem !important;
        font-weight: bold;
    }

    .markdown-text-container, .stMarkdown {
        font-size: 1.2rem !important;
    }

    label, .stButton>button, .stSlider, .stRadio, .stToggle {
        font-size: 1.1rem !important;
    }

    #MainMenu, footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# === Sidebar Layout ===
st.sidebar.title("Configuration")
option = st.sidebar.radio("Choose input type:", ["Upload Video", "Upload Image", "Webcam"], key="input_type")
slack_enabled = st.sidebar.toggle("Enable Slack Notifications", value=False)

# === Welcome Header ===
st.title("Silent Scream Detector")
st.markdown("""
Welcome to the **Distress Detection Interface**.

This tool analyzes **video or image input** to identify signs of **psychological distress** based on body and facial cues.

---

### What you can do:
- Upload video clips or pictures to identify potential signs of distress
- Access detailed feature-level explanations when distress is detected
- Activate alerts to automatically notify designated safety contacts
""")

# === Load Model and Scaler ===
project_root = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(project_root, "models", "rf_model.pkl")
scaler_path = os.path.join(project_root, "models", "scaler.pkl")
rf_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# === Initialize MediaPipe Models ===
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
face_model = mp_face.FaceMesh(static_image_mode=True)
pose_model = mp_pose.Pose(static_image_mode=True)

# === Record Video From Webcam Function ===

def record_video_from_webcam():
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = 20
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_path = os.path.join(tempfile.gettempdir(), "webcam_capture.mp4")
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

    recording = True
    stop_btn = st.button("Stop Recording")

    st.warning("Recording... Click **Stop Recording** when ready.")
    while recording:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        stframe.image(frame, channels="BGR")
        if stop_btn:
            recording = False

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return temp_path

# === Define Feature Extraction ===
def extract_features_from_single_image(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_results = face_model.process(rgb)
    pose_results = pose_model.process(rgb)
    features = {}

    # FACE
    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0].landmark
        left_eye = [face_landmarks[i] for i in [159, 145]]
        features["eye_openness"] = abs(left_eye[0].y - left_eye[1].y)
        features["mouth_openness"] = abs(face_landmarks[13].y - face_landmarks[14].y)
        dx = face_landmarks[133].x - face_landmarks[362].x
        dy = face_landmarks[133].y - face_landmarks[362].y
        features["brow_furrow"] = math.sqrt(dx**2 + dy**2)
    else:
        features.update({k: -1 for k in ["eye_openness", "mouth_openness", "brow_furrow"]})

    # POSE
    if pose_results.pose_landmarks:
        lm = pose_results.pose_landmarks.landmark
        left_shoulder, right_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER], lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_wrist, right_wrist, nose = lm[mp_pose.PoseLandmark.LEFT_WRIST], lm[mp_pose.PoseLandmark.RIGHT_WRIST], lm[mp_pose.PoseLandmark.NOSE]
        left_hip, right_hip = lm[mp_pose.PoseLandmark.LEFT_HIP], lm[mp_pose.PoseLandmark.RIGHT_HIP]

        shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
        hand_face_dist = np.sqrt((left_wrist.x - nose.x)**2 + (left_wrist.y - nose.y)**2)
        shoulder_mid = ((left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2)
        dx, dy = nose.x - shoulder_mid[0], nose.y - shoulder_mid[1]
        head_tilt = math.degrees(math.atan2(dy, dx))
        wrist_distance = math.sqrt((left_wrist.x - right_wrist.x)**2 + (left_wrist.y - right_wrist.y)**2)
        torso_lean = shoulder_mid[1] - (left_hip.y + right_hip.y) / 2

        features.update({
            "shoulder_tilt": shoulder_diff,
            "hand_face_dist": hand_face_dist,
            "head_tilt_angle": head_tilt,
            "wrist_distance": wrist_distance,
            "torso_lean": torso_lean,
        })
    else:
        features.update({k: -1 for k in ["shoulder_tilt", "hand_face_dist", "head_tilt_angle", "wrist_distance", "torso_lean"]})

    return features

# === Prediction Logic ===
feature_order = [
    "eye_openness", "mouth_openness", "brow_furrow", "shoulder_tilt",
    "hand_face_dist", "head_tilt_angle", "wrist_distance", "torso_lean"
]

def predict_frame(image):
    features = extract_features_from_single_image(image)
    values = [features.get(f, -1) for f in feature_order]
    X = np.array(values).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prob = rf_model.predict_proba(X_scaled)[0, 1]
    return prob

if option == "Upload Video":
    video_file = st.session_state.get("uploaded_video", None)
    if video_file is None:
        video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        frame_interval = 10
        frame_count = 0
        filenames, all_feature_vectors = [], []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                features = extract_features_from_single_image(frame)
                values = [features.get(f, -1) for f in feature_order]
                all_feature_vectors.append(values)
                filenames.append((frame.copy(), frame_count))
            frame_count += 1
        cap.release()

        if all_feature_vectors:
            X = np.array(all_feature_vectors)
            X_scaled = scaler.transform(X)
            probs = rf_model.predict_proba(X_scaled)[:, 1]
            preds = rf_model.predict(X_scaled)

            threshold = 0.77
            distress_indices = [i for i, (p, pr) in enumerate(zip(preds, probs)) if p == 1 and pr > threshold]
            distress_detected = len(distress_indices) > 0

            if distress_detected:
                mean_conf = np.mean([probs[i] for i in distress_indices])
                st.warning(f"🚨 Distress detected in {len(distress_indices)} frames.")
                st.info(f"Average distress confidence: {mean_conf:.2f}")

                top_3 = sorted(
                    [(float(probs[i]), filenames[i]) for i in distress_indices],
                    key=lambda x: x[0],
                    reverse=True
                )[:3]

                slack_frames = []
                for conf, (frame, idx) in top_3:
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Top frame (confidence: {conf:.2f})")
                    temp_path = os.path.join(tempfile.gettempdir(), f"top_{idx}.jpg")
                    cv2.imwrite(temp_path, frame)
                    slack_frames.append((conf, temp_path))

                # === SUMMARY FEATURE CONTRIBUTIONS ACROSS DISTRESS FRAMES ===
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
                importances = rf_model.feature_importances_
                contributions_list = []

                for i in distress_indices:
                    features = extract_features_from_single_image(filenames[i][0])
                    values = [features.get(f, -1) for f in feature_order]
                    X = np.array(values).reshape(1, -1)
                    X_scaled = scaler.transform(X)
                    contribution = X_scaled[0] * importances
                    contributions_list.append(contribution)

                if contributions_list:
                    contributions_array = np.array(contributions_list)
                    mean_contribution = np.abs(contributions_array).mean(axis=0)

                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(7, 5))
                    ax.barh(feature_order, mean_contribution)
                    ax.set_xlabel("Mean Contribution")
                    ax.set_title("Mean Feature Contributions Across Distressed Frames")
                    st.pyplot(fig)

                if slack_enabled:
                    send_slack_alert_summary(len(distress_indices), top_frames=slack_frames)
            
            else:
                st.success("✅ No distress detected.")

elif option == "Upload Image":
    image_file = st.file_uploader("Upload an image", type=["jpg", "png"])
    if image_file:
        img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        prob = predict_frame(img)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Confidence: {prob:.2f}")

        if prob > 0.77:
            st.warning("🚨 Distress detected!")

            # === Save frame
            temp_path = os.path.join(tempfile.gettempdir(), "uploaded_image.jpg")
            cv2.imwrite(temp_path, img)

            # === Feature Contributions
            importances = rf_model.feature_importances_
            features = extract_features_from_single_image(img)
            values = [features.get(f, -1) for f in feature_order]
            X = np.array(values).reshape(1, -1)
            X_scaled = scaler.transform(X)
            contribution = X_scaled[0] * importances
            mean_contribution = np.abs(contribution)

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.barh(feature_order, mean_contribution)
            ax.set_xlabel("Mean Contribution")
            ax.set_title("Feature Contributions for Uploaded Image")
            st.pyplot(fig)

            # === Slack Notification
            if slack_enabled:
                send_slack_alert_summary(1, top_frames=[(prob, temp_path)])
        else:
            st.success("✅ No distress detected.")

elif option == "Webcam":
    if st.button("Start Webcam Recording"):
        video_path = record_video_from_webcam()
        if video_path:
            st.video(video_path)
            st.info("Processing recorded video...")
            uploaded_file = open(video_path, "rb")
            st.session_state.uploaded_video = uploaded_file
            # Switch to Upload Video tab to reuse your processing logic
            st.session_state.input_type = "Upload Video"
            st.experimental_rerun()
    else:
        st.info("Click **Start Webcam Recording** to begin.")