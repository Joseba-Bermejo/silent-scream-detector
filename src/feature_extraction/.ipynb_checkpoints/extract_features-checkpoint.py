# src/feature_extraction/extract_features.py

import cv2
import mediapipe as mp
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

def extract_features_from_frames(frame_dir="data/processed_frames", output_csv="data/features.csv"):
    face_model = mp_face.FaceMesh(static_image_mode=True)
    pose_model = mp_pose.Pose(static_image_mode=True)

    feature_list = []

    for file in tqdm(sorted(os.listdir(frame_dir))):
        path = os.path.join(frame_dir, file)
        image = cv2.imread(path)
        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_results = face_model.process(rgb)
        pose_results = pose_model.process(rgb)

        features = {"filename": file}

        # FACE FEATURES
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0].landmark

            # Eye openness (left)
            left_eye = [face_landmarks[i] for i in [159, 145]]  # vertical: upper/lower eyelid
            eye_openness = abs(left_eye[0].y - left_eye[1].y)
            features["eye_openness"] = eye_openness

            # Mouth openness
            top_lip = face_landmarks[13].y
            bottom_lip = face_landmarks[14].y
            features["mouth_openness"] = abs(top_lip - bottom_lip)
        else:
            features["eye_openness"] = -1
            features["mouth_openness"] = -1

        # POSE FEATURES
        if pose_results.pose_landmarks:
            pose_landmarks = pose_results.pose_landmarks.landmark

            # Shoulder angle (slouch indicator)
            left_shoulder = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
            features["shoulder_tilt"] = shoulder_diff

            # Hand near face? (self-protection or stress)
            left_wrist = pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            nose = pose_landmarks[mp_pose.PoseLandmark.NOSE]
            hand_face_dist = np.sqrt((left_wrist.x - nose.x)**2 + (left_wrist.y - nose.y)**2)
            features["hand_face_dist"] = hand_face_dist
        else:
            features["shoulder_tilt"] = -1
            features["hand_face_dist"] = -1

        feature_list.append(features)

    df = pd.DataFrame(feature_list)
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Saved features to {output_csv}")