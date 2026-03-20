"""
Student 3 — Image Analyst
==========================
Detects objects in crime/incident scene images using YOLOv8,
classifies scene type, extracts visible text via OCR.

Dataset : Roboflow Fire Detection
Link    : https://universe.roboflow.com/search?q=fire
Place   : images/data/*.jpg or *.png

Output  : images/output_images.csv
Columns : Image_ID, Source_File, Scene_Type, Objects_Detected,
          Bounding_Boxes, Text_Extracted, Confidence_Score
"""

import os
import pandas as pd

YOLO_MODEL = "yolov8n.pt"   # auto-downloads on first run (~6 MB)

SCENE_MAP = {
    "Fire Scene":    ["fire", "smoke"],
    "Accident":      ["car", "truck", "bus", "motorcycle", "bicycle"],
    "Crowd/Fight":   ["person"],
    "Theft/Robbery": ["knife", "backpack", "handbag", "suitcase"],
}

def _load():
    from ultralytics import YOLO
    print("[Images] Loading YOLOv8 nano model ...")
    return YOLO(YOLO_MODEL)

def classify_scene(labels: list) -> str:
    label_set = {l.lower() for l in labels}
    for scene, kws in SCENE_MAP.items():
        if any(k in label_set for k in kws):
            return scene
    return "General Scene"

def ocr_text(path: str) -> str:
    import pytesseract
    from PIL import Image
    img   = Image.open(path).convert("RGB")
    text  = pytesseract.image_to_string(img, config="--psm 6")
    clean = " ".join(text.split())
    return clean[:200] if clean else "None"

def run(img_dir: str = "images/data",
        output_csv: str = "images/output_images.csv") -> pd.DataFrame:

    supported = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = [f for f in os.listdir(img_dir) if f.lower().endswith(supported)]

    if not files:
        print("[Images] No image files found — using built-in demo data.")
        return _demo(output_csv)

    model = _load()
    rows  = []
    for i, fname in enumerate(files, 1):
        iid  = f"IMG_{i:03d}"
        path = os.path.join(img_dir, fname)
        print(f"[Images] {iid} ← {fname}")

        results    = model(path, verbose=False)[0]
        detections = [
            {"label": results.names[int(b.cls)],
             "conf":  round(float(b.conf), 2),
             "bbox":  [round(float(c), 1) for c in b.xyxy[0].tolist()]}
            for b in results.boxes
        ]
        labels   = [d["label"] for d in detections]
        objects  = ", ".join(dict.fromkeys(labels)) or "None"
        avg_conf = round(sum(d["conf"] for d in detections) / len(detections), 2) if detections else 0.0
        bboxes   = "; ".join(f"{d['label']}@{[int(c) for c in d['bbox']]}" for d in detections) or "None"

        rows.append({
            "Image_ID":         iid,
            "Source_File":      fname,
            "Scene_Type":       classify_scene(labels),
            "Objects_Detected": objects,
            "Bounding_Boxes":   bboxes,
            "Text_Extracted":   ocr_text(path),
            "Confidence_Score": avg_conf,
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[Images] ✅ {len(df)} records → {output_csv}\n")
    return df

def _demo(output_csv: str = "images/output_images.csv") -> pd.DataFrame:
    df = pd.DataFrame([
        {"Image_ID": "IMG_034", "Source_File": "fire_scene.jpg",
         "Scene_Type": "Fire Scene", "Objects_Detected": "fire, smoke",
         "Bounding_Boxes": "fire@[10,20,200,300]; smoke@[250,50,500,400]",
         "Text_Extracted": "CAUTION FIRE ZONE", "Confidence_Score": 0.94},
        {"Image_ID": "IMG_035", "Source_File": "accident_scene.jpg",
         "Scene_Type": "Accident", "Objects_Detected": "car, person",
         "Bounding_Boxes": "car@[0,100,300,400]; person@[320,150,400,450]",
         "Text_Extracted": "ABC 1234", "Confidence_Score": 0.87},
        {"Image_ID": "IMG_036", "Source_File": "theft_scene.jpg",
         "Scene_Type": "Theft/Robbery", "Objects_Detected": "person, backpack",
         "Bounding_Boxes": "person@[50,50,250,450]; backpack@[90,200,190,380]",
         "Text_Extracted": "None", "Confidence_Score": 0.81},
    ])
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[Images] ✅ Demo data saved → {output_csv}\n")
    return df

if __name__ == "__main__":
    run()
