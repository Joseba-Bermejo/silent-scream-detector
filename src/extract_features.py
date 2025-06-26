import cv2
import mediapipe as mp
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import math

mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

def extract_features_from_frames(frame_dir="data/processed_frames", output_csv="data/features.csv", classification=None):
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
            left_eye = [face_landmarks[i] for i in [159, 145]]
            eye_openness = abs(left_eye[0].y - left_eye[1].y)
            features["eye_openness"] = eye_openness

            # Mouth openness
            top_lip = face_landmarks[13].y
            bottom_lip = face_landmarks[14].y
            features["mouth_openness"] = abs(top_lip - bottom_lip)

            # Brow furrow
            left_eye_inner = face_landmarks[133]
            right_eye_inner = face_landmarks[362]
            brow_dx = left_eye_inner.x - right_eye_inner.x
            brow_dy = left_eye_inner.y - right_eye_inner.y
            brow_furrow = math.sqrt(brow_dx**2 + brow_dy**2)
            features["brow_furrow"] = brow_furrow
        else:
            features["eye_openness"] = -1
            features["mouth_openness"] = -1
            features["brow_furrow"] = -1

        # POSE FEATURES
        if pose_results.pose_landmarks:
            pose_landmarks = pose_results.pose_landmarks.landmark

            left_shoulder = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
            features["shoulder_tilt"] = shoulder_diff

            left_wrist = pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            nose = pose_landmarks[mp_pose.PoseLandmark.NOSE]
            hand_face_dist = np.sqrt((left_wrist.x - nose.x)**2 + (left_wrist.y - nose.y)**2)
            features["hand_face_dist"] = hand_face_dist

            shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
            shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
            dx = nose.x - shoulder_mid_x
            dy = nose.y - shoulder_mid_y
            head_tilt_angle = math.degrees(math.atan2(dy, dx))
            features["head_tilt_angle"] = head_tilt_angle

            right_wrist = pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            wrist_dx = left_wrist.x - right_wrist.x
            wrist_dy = left_wrist.y - right_wrist.y
            wrist_distance = math.sqrt(wrist_dx**2 + wrist_dy**2)
            features["wrist_distance"] = wrist_distance

            left_hip = pose_landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            hip_mid_y = (left_hip.y + right_hip.y) / 2
            torso_lean = shoulder_mid_y - hip_mid_y
            features["torso_lean"] = torso_lean
        else:
            features["shoulder_tilt"] = -1
            features["hand_face_dist"] = -1
            features["head_tilt_angle"] = -1
            features["wrist_distance"] = -1
            features["torso_lean"] = -1

        # Add classification label if provided
        if classification is not None:
            features["label"] = classification

        feature_list.append(features)

    df = pd.DataFrame(feature_list)
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Saved features to {output_csv}")

import cv2
import mediapipe as mp
import numpy as np
import math

mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

face_model = mp_face.FaceMesh(static_image_mode=True)
pose_model = mp_pose.Pose(static_image_mode=True)

def extract_features_from_single_image(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_results = face_model.process(rgb)
    pose_results = pose_model.process(rgb)

    features = {
        "eye_openness": -1,
        "mouth_openness": -1,
        "brow_furrow": -1,
        "shoulder_tilt": -1,
        "hand_face_dist": -1,
        "head_tilt_angle": -1,
        "wrist_distance": -1,
        "torso_lean": -1
    }

    # FACE FEATURES
    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0].landmark

        # Eye openness (left)
        left_eye = [face_landmarks[i] for i in [159, 145]]
        features["eye_openness"] = abs(left_eye[0].y - left_eye[1].y)

        # Mouth openness
        top_lip = face_landmarks[13].y
        bottom_lip = face_landmarks[14].y
        features["mouth_openness"] = abs(top_lip - bottom_lip)

        # Brow furrow
        left_eye_inner = face_landmarks[133]
        right_eye_inner = face_landmarks[362]
        brow_dx = left_eye_inner.x - right_eye_inner.x
        brow_dy = left_eye_inner.y - right_eye_inner.y
        features["brow_furrow"] = math.sqrt(brow_dx**2 + brow_dy**2)

    # POSE FEATURES
    if pose_results.pose_landmarks:
        pose_landmarks = pose_results.pose_landmarks.landmark

        left_shoulder = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        features["shoulder_tilt"] = abs(left_shoulder.y - right_shoulder.y)

        left_wrist = pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        nose = pose_landmarks[mp_pose.PoseLandmark.NOSE]

        features["hand_face_dist"] = np.sqrt((left_wrist.x - nose.x)**2 + (left_wrist.y - nose.y)**2)

        shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
        dx = nose.x - shoulder_mid_x
        dy = nose.y - shoulder_mid_y
        features["head_tilt_angle"] = math.degrees(math.atan2(dy, dx))

        wrist_dx = left_wrist.x - right_wrist.x
        wrist_dy = left_wrist.y - right_wrist.y
        features["wrist_distance"] = math.sqrt(wrist_dx**2 + wrist_dy**2)

        left_hip = pose_landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        hip_mid_y = (left_hip.y + right_hip.y) / 2
        features["torso_lean"] = shoulder_mid_y - hip_mid_y

    return features