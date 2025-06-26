import cv2
import os
import shutil

def extract_frames(source="webcam", video_path=None, output_dir="data/processed_frames", frame_interval=10):
    """
    Extract frames from webcam, video file, or a single image.

    Parameters:
        source (str): "webcam", "video", or "image"
        video_path (str): path to video/image file (if source is not webcam)
        output_dir (str): folder to save extracted frames
        frame_interval (int): save every Nth frame (ignored for image)
    """
    os.makedirs(output_dir, exist_ok=True)

    if source == "webcam":
        cap = cv2.VideoCapture(0)
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

    elif source == "video":
        if not video_path or not os.path.exists(video_path):
            raise ValueError(f"[ERROR] Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        saved_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                filename = f"{saved_count:05d}.jpg"
                cv2.imwrite(os.path.join(output_dir, filename), frame)
                saved_count += 1

            frame_count += 1

        cap.release()
        print(f"[INFO] Extracted {saved_count} frames from video to {output_dir}")

    elif source == "image":
        if not video_path or not os.path.exists(video_path):
            raise ValueError(f"[ERROR] Image file not found: {video_path}")

        filename = os.path.basename(video_path)
        dest_path = os.path.join(output_dir, filename)
        shutil.copy(video_path, dest_path)
        print(f"[INFO] Copied image to {dest_path}")

    else:
        raise ValueError("Invalid source type. Use 'webcam', 'video', or 'image'.")