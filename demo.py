"""
demo.py — Deliverable 5: Demonstration
========================================
Shows step-by-step how raw unstructured data from each modality
is converted into the final structured incident report.

Run: python demo.py

Designed for live presentation — prints each transformation
clearly so the audience can follow the raw → structured flow.
"""

import os
import time
import pandas as pd

DIVIDER = "=" * 65

def pause():
    time.sleep(0.3)

def section(title):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)

def show_transform(raw_label, raw_value, structured: dict):
    print(f"\n  📥 RAW INPUT ({raw_label}):")
    print(f"     {raw_value[:130]}")
    print(f"\n  📤 STRUCTURED OUTPUT:")
    for k, v in structured.items():
        print(f"     {k:<24} : {v}")

def load_csv(path, label):
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"\n  ✅ {label} — {len(df)} records in {path}")
        return df
    else:
        print(f"\n  ⚠️  {path} not found — run the module first")
        return None

# ─────────────────────────────────────────────────────────────────────────────

def demo_audio():
    section("STUDENT 1 — AUDIO ANALYST  🎙️")
    print("  Modality : Emergency audio calls (.wav / .mp3)")
    print("  Dataset  : 911 Calls (Kaggle)")
    print("  Tools    : Whisper STT → spaCy NER → DistilBERT Sentiment")
    pause()

    show_transform(
        raw_label="911 Audio Call — transcribed by Whisper",
        raw_value='"There is a fire! People are trapped on the second floor of Downtown Avenue! Please hurry!"',
        structured={
            "Call_ID":          "C001",
            "Transcript":       "There is a fire! People are trapped on the second floor of Downtown Avenue!",
            "Extracted_Event":  "Fire",
            "Location":         "Downtown Avenue  ← spaCy GPE/LOC entity",
            "Sentiment":        "Distressed       ← DistilBERT: NEGATIVE (0.97)",
            "Urgency_Score":    "0.91             ← 13/14 urgency keywords matched",
        }
    )
    pause()
    df = load_csv("audio/output_audio.csv", "Audio output")
    if df is not None:
        print(df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────

def demo_pdf():
    section("STUDENT 2 — DOCUMENT ANALYST  📄")
    print("  Modality : Police PDF reports (.pdf)")
    print("  Dataset  : Arkansas PD 1033 Proposals (MuckRock FOIA)")
    print("  Tools    : pdfplumber → spaCy NER → pytesseract OCR (fallback)")
    pause()

    show_transform(
        raw_label="Raw PDF text extracted by pdfplumber",
        raw_value="Arkansas Police Dept. 1033 Training Plan Proposal, April 10 2015. Officer Johnson requests tactical equipment.",
        structured={
            "Report_ID":        "RPT_001",
            "Department":       "Arkansas PD        ← spaCy ORG entity",
            "Incident_Type":    "Administrative     ← keyword: training/proposal",
            "Doc_Type":         "Training Proposal",
            "Date":             "2015-04-10         ← regex date extraction",
            "Location":         "Little Rock, AR    ← spaCy GPE entity",
            "Officer":          "Officer Johnson    ← spaCy PERSON entity",
            "Key_Detail":       "Equipment request: tactical gear listed",
        }
    )
    pause()
    df = load_csv("pdf/output_pdf.csv", "PDF output")
    if df is not None:
        print(df[["Report_ID", "Department", "Incident_Type", "Date", "Location", "Officer"]].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────

def demo_images():
    section("STUDENT 3 — IMAGE ANALYST  🖼️")
    print("  Modality : Crime scene photographs (.jpg / .png)")
    print("  Dataset  : Roboflow Fire Detection Dataset")
    print("  Tools    : YOLOv8 object detection → scene classifier → pytesseract OCR")
    pause()

    show_transform(
        raw_label="Image file: fire_scene.jpg  [1920×1080 JPEG]",
        raw_value="[Raw pixel data — image of burning building with visible signage]",
        structured={
            "Image_ID":          "IMG_034",
            "Scene_Type":        "Fire Scene         ← fire/smoke labels → scene map",
            "Objects_Detected":  "fire, smoke        ← YOLOv8 detections",
            "Bounding_Boxes":    "fire@[10,20,200,300]; smoke@[250,50,500,400]",
            "Text_Extracted":    "CAUTION FIRE ZONE  ← pytesseract OCR",
            "Confidence_Score":  "0.94",
        }
    )
    pause()
    df = load_csv("images/output_images.csv", "Images output")
    if df is not None:
        print(df[["Image_ID", "Scene_Type", "Objects_Detected", "Text_Extracted", "Confidence_Score"]].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────

def demo_video():
    section("STUDENT 4 — VIDEO ANALYST  🎥")
    print("  Modality : CCTV / surveillance footage (.mp4 / .mpg)")
    print("  Dataset  : CAVIAR CCTV Dataset (Edinburgh University)")
    print("  Tools    : OpenCV frame extraction → motion scoring → YOLOv8 per frame")
    pause()

    show_transform(
        raw_label="Video clip: CAVIAR_03.mpg  [25fps, 384×288, grayscale]",
        raw_value="[Raw video stream — sampled every 30 frames = 1 frame per second]",
        structured={
            "Clip_ID":          "CAVIAR_03",
            "Timestamp":        "00:00:12          ← frame_index / fps",
            "Frame_ID":         "FRM_036",
            "Motion_Score":     "0.08              ← 8% pixel diff vs previous frame",
            "Event_Detected":   "Person collapsing ← YOLOv8 label + motion rule",
            "Persons_Count":    "1                 ← YOLOv8 person class count",
            "Confidence":       "0.88",
        }
    )
    pause()
    df = load_csv("video/output_video.csv", "Video output")
    if df is not None:
        print(df[["Clip_ID", "Timestamp", "Frame_ID", "Event_Detected", "Persons_Count", "Confidence"]].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────

def demo_text():
    section("STUDENT 5 — TEXT ANALYST  📝")
    print("  Modality : Social media posts / news articles (.csv)")
    print("  Dataset  : CrimeReport (Kaggle) / sample_data/sample_text.csv")
    print("  Tools    : NLTK clean → spaCy NER → DistilBERT sentiment → BART zero-shot")
    pause()

    show_transform(
        raw_label="Raw Twitter post",
        raw_value='"Just witnessed a robbery on Oak Street, Chicago! Someone grabbed a bag and ran. Police are on the way."',
        structured={
            "Text_ID":           "TXT_112",
            "Source":            "Twitter",
            "Cleaned_Text":      "witnessed robbery oak street chicago grabbed bag ran police way",
            "Sentiment":         "Negative          ← DistilBERT NEGATIVE (0.97)",
            "Location_Entity":   "Oak Street, Chicago  ← spaCy GPE/LOC NER",
            "Crime_Type":        "Robbery           ← BART zero-shot (top label)",
            "Topic":             "Theft / Robbery",
            "Severity_Label":    "High              ← SEVERITY_MAP rule",
        }
    )
    pause()
    df = load_csv("text/output_text.csv", "Text output")
    if df is not None:
        print(df[["Text_ID", "Source", "Sentiment", "Location_Entity", "Crime_Type", "Severity_Label"]].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────

def demo_integration():
    section("INTEGRATION — UNIFIED INCIDENT REPORT  🔗")
    print("  All 5 module CSVs merged → single structured dataset")
    print("  Severity scored from combined signals across all modalities\n")
    pause()

    steps = [
        ("Step 1", "Define common Incident_ID (INC_001, INC_002 ...) across all 5 CSVs"),
        ("Step 2", "Merge all 5 DataFrames row-by-row on Incident_ID"),
        ("Step 3", "Handle missing values → fill with 'N/A'"),
        ("Step 4", "Compute severity: Audio urgency + sentiment + Text label + Image/Video confidence"),
        ("Step 5", "Save integration/integrated_incidents.csv → launch dashboard"),
    ]
    for label, desc in steps:
        print(f"  ✅ {label}: {desc}")
        pause()

    print("\n  📊 Final Output Schema (matches assignment requirement):")
    print(f"  {'Incident_ID':<12} {'Source':<28} {'Event':<10} {'Location':<18} {'Time':<10} Severity")
    print(f"  {'-'*12} {'-'*28} {'-'*10} {'-'*18} {'-'*10} --------")
    print(f"  {'INC_001':<12} {'Audio+PDF+Image+Video+Text':<28} {'Fire':<10} {'Downtown Ave':<18} {'00:00:12':<10} High")
    print(f"  {'INC_002':<12} {'Audio+PDF+Image+Video+Text':<28} {'Accident':<10} {'Main Street':<18} {'00:00:24':<10} Medium")
    print(f"  {'INC_003':<12} {'Audio+PDF+Image+Video+Text':<28} {'Theft':<10} {'Oak Avenue':<18} {'00:00:08':<10} High")

    pause()
    df = load_csv("integration/integrated_incidents.csv", "Integrated output")
    if df is not None:
        print(df[["Incident_ID", "Source", "Event", "Location", "Time", "Severity"]].to_string(index=False))

    print(f"\n  🖥️  Launch dashboard:")
    print(f"      streamlit run integration/dashboard.py")
    print(f"      Open http://localhost:8501 in your browser\n")

# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*65}")
    print("  🚨 MULTIMODAL CRIME / INCIDENT ANALYZER — DEMONSTRATION")
    print("     Deliverable 5 | AI for Engineers Group Assignment")
    print(f"{'='*65}")
    print("\n  This demo shows how each modality converts raw unstructured")
    print("  data into structured fields, then merges into one report.\n")

    demo_audio()
    demo_pdf()
    demo_images()
    demo_video()
    demo_text()
    demo_integration()

    section("DEMONSTRATION COMPLETE ✅")
    print("  All 5 modalities shown: Audio → PDF → Images → Video → Text")
    print("  Final output: integration/integrated_incidents.csv")
    print("  Dashboard:    streamlit run integration/dashboard.py\n")

if __name__ == "__main__":
    main()
