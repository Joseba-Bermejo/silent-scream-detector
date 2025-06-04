# src/data_preprocessing/extract_frames.py

import cv2
import os

def extract_frames(source="webcam", video_path=None, output_dir="data/processed_frames", frame_interval=10):
    """
    Extract frames from webcam or video file and save them.

    Parameters:
        source (str): "webcam" or "video"
        video_path (str): path to video file (if source="video")
        output_dir (str): folder to save extracted frames
        frame_interval (int): save every Nth frame
    """
    os.makedirs(output_dir, exist_ok=True)

    if source == "webcam":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)

    frame_count = 0
    saved_count = 0

    print("[INFO] Press 'q' to stop webcam capture.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = f"{saved_count:05d}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            saved_count += 1

        cv2.imshow("Frame Capture", frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Saved {saved_count} frames to {output_dir}")