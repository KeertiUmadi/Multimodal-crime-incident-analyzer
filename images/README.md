# Student 3 — Image Analyst 🖼️

## Task
Detect objects in crime scene images, classify scene type, extract visible text via OCR.

## Tools
- `YOLOv8 (ultralytics)` — real-time object detection (80 COCO classes)
- `OpenCV` — image preprocessing
- `pytesseract` — OCR for license plates, street signs

## Dataset
**Roboflow Fire Detection** — [Link](https://universe.roboflow.com/search?q=fire)
1. Open link → pick a dataset with 1000+ images
2. Click **Download Dataset** → choose **YOLOv8 format**
3. Free with a Roboflow account
4. Place images in `images/data/`

## Run
```bash
pip install -r images/requirements.txt
python images/image_analyzer.py
```
> No files in `images/data/`? Demo data runs automatically.

## Output: `images/output_images.csv`
| Image_ID | Scene_Type | Objects_Detected | Bounding_Boxes | Text_Extracted | Confidence_Score |
|----------|------------|------------------|----------------|----------------|------------------|
| IMG_034 | Fire Scene | fire, smoke | fire@[10,20,200,300] | CAUTION FIRE | 0.94 |
