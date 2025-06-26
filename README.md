# Silent Scream Detector

A lightweight computer vision pipeline that detects **non-verbal signs of distress** (e.g., silent screams, tense posture) in webcam or video footage using landmark-based features and machine learning. Includes Slack-based alerting for real-time demo scenarios.

---

## Objective

To build a system that flags **psychological distress** using visual cues like facial expressions and body posture — without relying on sound. This is particularly useful in **silent or constrained environments** where speech is not an option.

---

## Dataset

- Custom dataset built from **webcam recordings**.
- Each clip labeled as:
  - `0` → Calm
  - `1` → Distress
- **MediaPipe landmarks** extracted from each frame.
- Engineered **8 numerical features** from those landmarks.

---

## Tools & Libraries

| Task                     | Libraries Used                      |
|--------------------------|-------------------------------------|
| Frame Extraction         | OpenCV                              |
| Landmark Detection       | MediaPipe (FaceMesh + Pose)         |
| Feature Engineering      | NumPy, math                         |
| ML Model Training        | scikit-learn (Random Forest)        |
| Data Handling            | pandas                              |
| Visualization            | matplotlib, seaborn                 |
| Prototyping              | Jupyter Notebooks                   |
| Alerts / Demo Interface  | Slack API, Streamlit                |

---

## Features Extracted

| Feature Name       | Description                              |
|--------------------|------------------------------------------|
| Eye Openness       | Squinting or wide-open eyes              |
| Mouth Openness     | Screaming, gasping, or mouth tension     |
| Brow Furrow        | Tension between eyebrows                 |
| Shoulder Tilt      | Postural asymmetry or slouching          |
| Hand-to-Face Dist. | Defensive or soothing gestures           |
| Head Tilt Angle    | Emotional leaning or imbalance           |
| Wrist Distance     | Unusual hand movements or clasping       |
| Torso Lean         | Curling or hunching forward              |

---

## Pipeline Overview

1. **Video Upload or Webcam Capture** (`OpenCV`)
2. **Frame Extraction** every *n* frames
3. **Landmark Detection** using `MediaPipe`
4. **Feature Engineering** → 8D vector per frame
5. **Classifier**: Random Forest
6. **Top-3 Distress Frames** flagged with probability
7. **Slack Alerts** sent 

---

## Results

| Metric        | Value |
|---------------|-------|
| Accuracy      | 0.88  |
| Recall (1s)   | 0.91  |
| Precision (1s)| 0.84  |

> ⚠️ Prioritized **Recall** to minimize missed distress cases (false negatives).

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/silent-scream-detector.git
cd silent-scream-detector

# 2. Set up environment
conda env create -f environment.yml
conda activate silent-scream

# 3. Configure Slack (optional)
cp .env.example .env
# Fill in SLACK_BOT_TOKEN and SLACK_CHANNEL_ID

# 4. Run the demo
streamlit run app.py
