import os
import pandas as pd
from PIL import Image as PILImage

YOLO_MODEL = "yolov8n.pt"

# Roboflow dataset class names
ROBOFLOW_CLASSES = {0: "fire", 1: "light", 2: "no-fire", 3: "smoke"}

# ✅ FIXED scene names (match expected output)
SCENE_MAP = {
    "Fire Scene":           ["fire", "smoke"],
    "Accident":             ["car", "truck", "bus", "motorcycle", "bicycle"],
    "Public Disturbance":   ["person"],
    "Theft":                ["knife", "backpack", "handbag", "suitcase"],
}

def _load_yolo():
    from ultralytics import YOLO
    print("[Images] Loading YOLOv8 model ...")
    return YOLO(YOLO_MODEL)

def classify_scene(labels):
    label_set = {l.lower() for l in labels}
    for scene, kws in SCENE_MAP.items():
        if any(k in label_set for k in kws):
            return scene
    return "General Scene"

def describe_bboxes(detections):
    if not detections:
        return "None"

    counts = {}
    for d in detections:
        lbl = d["label"]
        if lbl not in ("no-fire", "light"):
            counts[lbl] = counts.get(lbl, 0) + 1

    label_names = {
        "fire": "fire region",
        "smoke": "smoke plume",
        "person": "person",
        "car": "vehicle",
        "truck": "vehicle",
    }

    parts = []
    for lbl, cnt in counts.items():
        name = label_names.get(lbl, lbl)
        parts.append(f"{cnt} {name}{'s' if cnt > 1 else ''}")

    return ", ".join(parts) if parts else "None"

def read_roboflow_label(img_path, img_w, img_h):
    fname = os.path.splitext(os.path.basename(img_path))[0]
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
                cx, cy, w, h = map(float, parts[1:])

                x1 = int((cx - w / 2) * img_w)
                y1 = int((cy - h / 2) * img_h)
                x2 = int((cx + w / 2) * img_w)
                y2 = int((cy + h / 2) * img_h)

                label = ROBOFLOW_CLASSES.get(cls_id, f"class_{cls_id}")

                detections.append({
                    "label": label,
                    "conf": 0.95,
                    "bbox": [x1, y1, x2, y2],
                })

    return detections

def run(img_dir="images/data", output_csv="images/output_images.csv"):

    supported = (".jpg", ".jpeg", ".png")

    files = [f for f in os.listdir(img_dir) if f.lower().endswith(supported)]

    labels_dir = os.path.join(os.path.dirname(img_dir), "labels")
    has_labels = os.path.exists(labels_dir) and len(os.listdir(labels_dir)) > 0

    yolo_model = None
    if not has_labels:
        yolo_model = _load_yolo()

    rows = []

    for i, fname in enumerate(files, 1):
        iid = f"IMG_{i:03d}"
        path = os.path.join(img_dir, fname)
        print(f"[Images] Processing {iid} / IMG_{len(files):03d} <- {fname[:50]}")

        with PILImage.open(path) as im:
            img_w, img_h = im.size

        detections = []

        # Use Roboflow labels if available
        if has_labels:
            detections = read_roboflow_label(path, img_w, img_h)

        # Else YOLO detection
        if not detections and yolo_model:
            results = yolo_model(path, verbose=False)[0]
            detections = [
                {
                    "label": results.names[int(b.cls)],
                    "conf": round(float(b.conf), 2),
                    "bbox": [round(float(c), 1) for c in b.xyxy[0].tolist()],
                }
                for b in results.boxes
            ]

        labels_clean = [d["label"] for d in detections if d["label"] not in ("no-fire", "light")]
        objects = ", ".join(dict.fromkeys(labels_clean)) if labels_clean else "None"

        avg_conf = round(sum(d["conf"] for d in detections) / len(detections), 2) if detections else 0.0

        rows.append(
            {
                "Image_ID": iid,
                "Scene_Type": classify_scene(labels_clean),
                "Objects_Detected": objects,
                "Bounding_Boxes": describe_bboxes(detections),
                "Confidence_Score": avg_conf,
            }
        )

    output_columns = [
        "Image_ID",
        "Scene_Type",
        "Objects_Detected",
        "Bounding_Boxes",
        "Confidence_Score",
    ]
    df = pd.DataFrame(rows)[output_columns]
    df.to_csv(output_csv, index=False)

    print(f"\n✅ Output saved to {output_csv}")
    print(df.head())

    return df


if __name__ == "__main__":
    run()