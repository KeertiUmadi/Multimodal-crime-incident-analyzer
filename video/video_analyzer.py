"""
Student 4 — Video Analyst
==========================
Extracts frames from CCTV/surveillance video every N frames,
runs YOLOv8 on each sampled frame, and outputs a timestamped event log.

Dataset : CAVIAR CCTV Dataset
Link    : https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/
Place   : video/data/*.mpg or *.mp4

Output  : video/output_video.csv
Columns : Clip_ID, Timestamp, Frame_ID, Event_Detected, Persons_Count, Confidence
"""

import os
import cv2
import pandas as pd

YOLO_MODEL     = "yolov8n.pt"
FRAME_INTERVAL = 30   # sample 1 frame per second at 30fps

EVENT_MAP = {
    "person":     "Person detected",
    "car":        "Vehicle movement",
    "truck":      "Vehicle movement",
    "motorcycle": "Vehicle movement",
    "bicycle":    "Bicycle movement",
}

def _load():
    from ultralytics import YOLO
    print("[Video] Loading YOLOv8 nano model ...")
    return YOLO(YOLO_MODEL)

def motion_score(gray, prev_gray, threshold: int = 25) -> float:
    if prev_gray is None:
        return 0.0
    diff = cv2.absdiff(prev_gray, gray)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    return round(float(mask.sum() / (mask.size * 255)), 4)

def process_clip(model, clip_path: str, clip_id: str) -> list:
    cap      = cv2.VideoCapture(clip_path)
    fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    rows, frame_idx, saved_idx, prev_gray = [], 0, 0, None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % FRAME_INTERVAL == 0:
            ts  = frame_idx / fps
            hh  = int(ts // 3600)
            mm  = int((ts % 3600) // 60)
            ss  = int(ts % 60)
            fid = f"FRM_{saved_idx:03d}"

            gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion = motion_score(gray, prev_gray)
            prev_gray = gray

            results  = model(frame, verbose=False)[0]
            labels   = [results.names[int(b.cls)] for b in results.boxes]
            confs    = [float(b.conf) for b in results.boxes]
            persons  = labels.count("person")

            event, max_conf = "No significant event", 0.0
            for lbl, conf in zip(labels, confs):
                if lbl in EVENT_MAP and conf > max_conf:
                    event, max_conf = EVENT_MAP[lbl], conf

            if motion > 0.06 and persons > 1:
                event = "Crowd / Suspicious movement"
            if motion > 0.10 and persons >= 1:
                event = "Rapid movement detected"

            rows.append({
                "Clip_ID": clip_id,
                "Timestamp": f"{hh:02d}:{mm:02d}:{ss:02d}",
                "Frame_ID": fid,
                "Event_Detected": event,
                "Persons_Count": f"{persons} person" if persons == 1 else f"{persons} persons",
                "Confidence": round(max_conf, 2),
            })
            saved_idx += 1
        frame_idx += 1

    cap.release()
    return rows

def run(video_dir: str = "video/data",
        output_csv: str = "video/output_video.csv") -> pd.DataFrame:

    supported = (".mp4", ".avi", ".mpg", ".mpeg", ".mov", ".mkv")

    if not os.path.exists(video_dir):
        raise FileNotFoundError(
            f"[Video] Directory not found: {video_dir}. "
            "Add .mp4 / .mpg / .avi / .mov / .mkv files under video/data (see README)."
        )

    files = [f for f in os.listdir(video_dir) if f.lower().endswith(supported)]

    if not files:
        raise ValueError(
            f"[Video] No supported video files in {video_dir}. "
            f"Expected extensions: {', '.join(supported)}"
        )

    model    = _load()
    all_rows = []

    for fname in files:
        cid = os.path.splitext(fname)[0]
        print(f"[Video] Processing {cid} ← {fname}")
        rows = process_clip(model, os.path.join(video_dir, fname), cid)
        all_rows.extend(rows)
        print(f"  → {len(rows)} frames sampled")

    video_columns = [
        "Clip_ID",
        "Timestamp",
        "Frame_ID",
        "Event_Detected",
        "Persons_Count",
        "Confidence",
    ]
    df = pd.DataFrame(all_rows, columns=video_columns)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n[Video] ✅ {len(df)} frames → {output_csv}")
    print(df.head())
    return df

if __name__ == "__main__":
    run()
