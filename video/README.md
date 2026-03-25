# Student 4 — Video Analyst 🎥

## Task
Extract frames from CCTV footage, detect motion anomalies, classify activities per frame using YOLOv8.

## Tools
- `OpenCV` — frame extraction + motion detection
- `YOLOv8` — object detection on sampled frames
- `moviepy / imageio` — video loading utilities

## Dataset
**CAVIAR CCTV Dataset** — [Edinburgh University](https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/)
1. Open link → browse scenario folders
2. Download `.mpg` clips directly (no account needed)
3. Start with: `Browse`, `OneStopEnter`, `Fight`, or `Collapse` folders
4. Place clips in `video/data/`

## Run
```bash
pip install -r video/requirements.txt
python video/video_analyzer.py
```
> No files in `video/data/`? Demo data runs automatically.

## Output: `video/output_video.csv`
| Clip_ID | Timestamp | Frame_ID | Motion_Score | Event_Detected | Persons_Count | Confidence |
|---------|-----------|----------|--------------|----------------|---------------|------------|
| CAVIAR_03 | 00:00:12 | FRM_036 | 0.08 | Person collapsing | 1 | 0.88 |
