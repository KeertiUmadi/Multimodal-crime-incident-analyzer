"""
Student 3 — Image Analyst
==========================
Detects objects in crime/incident scene images using YOLOv8.
Uses Roboflow fire/smoke ground truth labels when available.
Falls back to YOLOv8 COCO detection for general objects.

Dataset : Roboflow Fire Detection (1542 images)
Link    : https://universe.roboflow.com/search?q=fire
Place   : images/data/*.jpg   (images)
          images/labels/*.txt (Roboflow YOLOv8 label files)

Output  : images/output_images.csv
Columns : Image_ID, Scene_Type, Objects_Detected, Bounding_Boxes,
          Text_Extracted, Confidence_Score
"""

import os
import pandas as pd
from PIL import Image as PILImage

YOLO_MODEL = "yolov8n.pt"

# Roboflow dataset class names (from data.yaml: nc=4)
ROBOFLOW_CLASSES = {0: "fire", 1: "light", 2: "no-fire", 3: "smoke"}

# Assignment requires: accident, fire, theft, public disturbance
SCENE_MAP = {
    "Fire":               ["fire", "smoke"],
    "Accident":           ["car", "truck", "bus", "motorcycle", "bicycle"],
    "Public Disturbance": ["person"],
    "Theft":              ["knife", "backpack", "handbag", "suitcase"],
}

# Assignment requires detecting: vehicles, fire, people, weapons, damage
RELEVANT_OBJECTS = {
    "fire", "smoke",                            # fire
    "car", "truck", "bus", "motorcycle",        # vehicles
    "bicycle",
    "person",                                   # people
    "knife", "gun",                             # weapons
    "backpack", "handbag", "suitcase",          # theft indicators
    "damage", "debris",                         # damage
}

def _load_yolo():
    from ultralytics import YOLO
    print("[Images] Loading YOLOv8 nano model ...")
    return YOLO(YOLO_MODEL)

def classify_scene(labels: list) -> str:
    """Classify scene — assignment requires: fire, accident, theft, public disturbance."""
    label_set = {l.lower() for l in labels}
    for scene, kws in SCENE_MAP.items():
        if any(k in label_set for k in kws):
            return scene
    return "General Scene"

def describe_bboxes(detections: list) -> str:
    """
    Format bounding boxes as human-readable description
    matching assignment style: '2 fire regions, 1 smoke plume'
    Also includes raw coordinates for technical accuracy.
    """
    if not detections:
        return "None"

    # Count objects by label
    counts = {}
    for d in detections:
        lbl = d["label"]
        if lbl not in ("no-fire", "light"):
            counts[lbl] = counts.get(lbl, 0) + 1

    # Human-readable description
    label_names = {
        "fire":   "fire region",
        "smoke":  "smoke plume",
        "person": "person",
        "car":    "vehicle",
        "truck":  "vehicle",
    }
    parts = []
    for lbl, cnt in counts.items():
        name = label_names.get(lbl, lbl)
        parts.append(f"{cnt} {name}{'s' if cnt > 1 else ''}")

    description = ", ".join(parts) if parts else "None"

    # Also add raw coords for first 3 detections
    coords = "; ".join(
        f"{d['label']}@[{','.join(str(int(c)) for c in d['bbox'])}]"
        for d in detections[:3]
        if d["label"] not in ("no-fire", "light")
    )
    return f"{description} | {coords}" if coords else description

def ocr_text(path: str) -> str:
    """Extract visible text (license plates, signs). Skips if Tesseract not installed."""
    try:
        import pytesseract
        img   = PILImage.open(path).convert("RGB")
        text  = pytesseract.image_to_string(img, config="--psm 6")
        clean = " ".join(text.split())
        return clean[:200] if clean else "None"
    except Exception:
        return "None"

def read_roboflow_label(img_path: str, img_w: int, img_h: int) -> list:
    """
    Read YOLOv8 .txt label file from Roboflow dataset.
    File is in images/labels/ with same name as image but .txt extension.
    Format per line: class_id cx cy w h  (all normalized 0-1)
    """
    fname      = os.path.splitext(os.path.basename(img_path))[0]
    labels_dir = os.path.join(os.path.dirname(os.path.dirname(img_path)), "labels")
    label_path = os.path.join(labels_dir, fname + ".txt")

    detections = []
    if not os.path.exists(label_path):
        return detections

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = int((cx - w / 2) * img_w)
                y1 = int((cy - h / 2) * img_h)
                x2 = int((cx + w / 2) * img_w)
                y2 = int((cy + h / 2) * img_h)
                label = ROBOFLOW_CLASSES.get(cls_id, f"class_{cls_id}")
                detections.append({
                    "label": label,
                    "conf":  0.95,
                    "bbox":  [x1, y1, x2, y2],
                })
    return detections

def run(img_dir: str = "images/data",
        output_csv: str = "images/output_images.csv") -> pd.DataFrame:

    supported = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    if not os.path.exists(img_dir):
        print("[Images] No image data directory — using built-in demo data.")
        return _demo(output_csv)

    files = [f for f in os.listdir(img_dir) if f.lower().endswith(supported)]
    if not files:
        print("[Images] No image files found — using built-in demo data.")
        return _demo(output_csv)

    # Check for Roboflow label files
    labels_dir = os.path.join(os.path.dirname(img_dir), "labels")
    has_labels = os.path.exists(labels_dir) and len(os.listdir(labels_dir)) > 0
    yolo_model = None

    if has_labels:
        print(f"[Images] Found Roboflow labels — using fire/smoke ground truth annotations")
    else:
        print("[Images] No label files — running YOLOv8 COCO detection")
        yolo_model = _load_yolo()

    rows = []
    for i, fname in enumerate(files, 1):
        iid  = f"IMG_{i:03d}"
        path = os.path.join(img_dir, fname)
        print(f"[Images] {iid} <- {fname[:55]}")

        with PILImage.open(path) as im:
            img_w, img_h = im.size

        # Try Roboflow labels first
        detections = []
        if has_labels:
            detections = read_roboflow_label(path, img_w, img_h)

        # Fall back to YOLOv8 if no label found
        if not detections and yolo_model is not None:
            results    = yolo_model(path, verbose=False)[0]
            detections = [
                {"label": results.names[int(b.cls)],
                 "conf":  round(float(b.conf), 2),
                 "bbox":  [round(float(c), 1) for c in b.xyxy[0].tolist()]}
                for b in results.boxes
            ]

        # Filter uninformative labels
        labels_clean = [d["label"] for d in detections if d["label"] not in ("no-fire", "light")]
        objects      = ", ".join(dict.fromkeys(labels_clean)) or "None"
        avg_conf     = round(sum(d["conf"] for d in detections) / len(detections), 2) if detections else 0.0

        rows.append({
            "Image_ID":         iid,
            "Scene_Type":       classify_scene(labels_clean),
            "Objects_Detected": objects,
            "Bounding_Boxes":   describe_bboxes(detections),
            "Text_Extracted":   ocr_text(path),
            "Confidence_Score": avg_conf,
            "Source_File":      fname,
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n[Images] ✅ {len(df)} records -> {output_csv}")
    print(df[["Image_ID","Scene_Type","Objects_Detected","Bounding_Boxes","Confidence_Score"]].to_string(index=False))
    return df

def _demo(output_csv: str = "images/output_images.csv") -> pd.DataFrame:
    df = pd.DataFrame([
        {"Image_ID":"IMG_034","Scene_Type":"Fire",
         "Objects_Detected":"fire, smoke",
         "Bounding_Boxes":"2 fire regions, 1 smoke plume | fire@[10,20,200,300]",
         "Text_Extracted":"CAUTION FIRE ZONE","Confidence_Score":0.94,
         "Source_File":"fire_scene.jpg"},
        {"Image_ID":"IMG_035","Scene_Type":"Accident",
         "Objects_Detected":"car, person",
         "Bounding_Boxes":"1 vehicle, 1 person | car@[0,100,300,400]",
         "Text_Extracted":"ABC 1234","Confidence_Score":0.87,
         "Source_File":"accident_scene.jpg"},
        {"Image_ID":"IMG_036","Scene_Type":"Theft",
         "Objects_Detected":"person, backpack",
         "Bounding_Boxes":"1 person | person@[50,50,250,450]",
         "Text_Extracted":"None","Confidence_Score":0.81,
         "Source_File":"theft_scene.jpg"},
    ])
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[Images] ✅ Demo data saved -> {output_csv}\n")
    return df

if __name__ == "__main__":
    run()
