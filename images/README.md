# 🖼️ Image Analyst — Student 3

**Modality: Crime Scene / Accident Scene Photographs**

This module analyzes images from incident scenes, detecting objects of interest and extracting visual evidence into structured data using computer vision and OCR.

---

## 🎯 Responsibilities

- Run object detection to identify relevant items: vehicles, fire, people, weapons, damage
- Classify scene type: accident, fire, theft, public disturbance
- Detect objects and scene type (OCR text is not written to the modality CSV schema)
- Output a structured CSV with detection results and confidence scores

---

## 📤 Output Schema

| Image_ID | Scene_Type | Objects_Detected | Bounding_Boxes | Confidence_Score |
|----------|------------|------------------|----------------|----------------|
| IMG_034 | Fire Scene | fire, smoke | 2 fire regions, 1 smoke plume | 0.94 |

**CSV header (exact order):** `Image_ID,Scene_Type,Objects_Detected,Bounding_Boxes,Confidence_Score`

---

## 🛠️ Tools & Libraries

| Library | Purpose | Install |
|---------|---------|---------|
| `ultralytics` (YOLOv8) | State-of-the-art object detection | `pip install ultralytics` |
| `OpenCV` | Image preprocessing and frame handling | `pip install opencv-python` |
| `pytesseract` | OCR for text visible in images | `pip install pytesseract` |
| `torchvision` / HuggingFace | Pre-trained image classification models | `pip install torchvision transformers` |

---

## 📦 Dataset

**Roboflow Fire Detection** — pre-labeled fire and smoke images ready for YOLOv8. Choose a dataset with 1000+ images from the search results.

- **Link:** [universe.roboflow.com — fire detection datasets](https://universe.roboflow.com/search?q=fire)
- **Access:** Open the link → pick a dataset with 1000+ images and a trained model badge → click **Download Dataset** → choose **YOLOv8 format**. Free with a Roboflow account.

> **Tip:** Roboflow provides a starter Colab notebook with every dataset. Use it to run YOLOv8 inference immediately without writing setup code from scratch.

---

## 🚀 Running the Module

```bash
pip install -r requirements.txt
python image_analyzer.py
```

Output will be saved to `images/output_images.csv`.
