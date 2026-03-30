# 🎥 Video Analyst — Student 4

**Modality: CCTV / Surveillance Footage**

This module processes video footage to detect abnormal activities, extract key events frame by frame, and produce a timestamped event log from surveillance clips.

---

## 🎯 Responsibilities

- Extract frames from video clips at regular intervals
- Apply motion detection or anomaly detection to identify events of interest
- Detect and classify objects or activities: running, fighting, vehicle movement, fire
- Output a structured CSV with timestamped event detections

---

## 📤 Output Schema

| Clip_ID | Timestamp | Frame_ID | Event_Detected | Persons_Count | Confidence |
|---------|-----------|----------|----------------|---------------|------------|
| CAVIAR_03 | 00:00:12 | FRM_036 | Person collapsing | 1 person | 0.88 |

**CSV header (exact order):** `Clip_ID,Timestamp,Frame_ID,Event_Detected,Persons_Count,Confidence`

---

## 🛠️ Tools & Libraries

| Library | Purpose | Install |
|---------|---------|---------|
| `OpenCV` | Frame extraction and motion detection | `pip install opencv-python` |
| `ultralytics` (YOLOv8) | Real-time object detection on extracted frames | `pip install ultralytics` |
| `PyTorch` / `TensorFlow` | Anomaly detection model implementation | `pip install torch` |
| `imageio` / `moviepy` | Video loading and manipulation | `pip install moviepy imageio` |

---

## 📦 Dataset

**CAVIAR CCTV Dataset** — simulated indoor surveillance footage of people walking, fighting, and collapsing. Lightweight and ideal for a student prototype.

- **Link:** [homepages.inf.ed.ac.uk/rbf/CAVIARDATA1](https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/)
- **Access:** Open the link → browse the scenario folders → download `.mpg` video clips directly. No account or signup needed.
- **Recommended clips:** Start with `Browse` or `OneStopEnter` folders for basic motion, then try `Fight` or `Collapse` scenarios for anomaly detection.

> **Tip:** Download only 3–5 short clips to start. Extract frames with OpenCV and run YOLOv8 on each frame. This is more than enough for a working prototype.

---

## 🚀 Running the Module

```bash
pip install -r requirements.txt
python video_analyzer.py
```

Output will be saved to `video_output.csv`.
