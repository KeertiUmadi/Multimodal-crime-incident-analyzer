"""
Integration — Full Team
========================
Merges all 5 modality CSV outputs into one unified incident dataset.

Assignment Steps:
  1. Define common Incident_ID across all 5 CSVs
  2. Merge DataFrames on Incident_ID
  3. Handle missing values with N/A
  4. Generate severity classification (Low / Medium / High)
  5. Save integrated_incidents.csv (ready for dashboard)

Output  : integration/integrated_incidents.csv
Schema  : Incident_ID | Source | Event | Location | Time | Severity | + all modality fields
"""

import os
import pandas as pd

OUTPUTS = {
    "audio":  "audio/output_audio.csv",
    "pdf":    "pdf/output_pdf.csv",
    "images": "images/output_images.csv",
    "video":  "video/output_video.csv",
    "text":   "text/output_text.csv",
}
INTEGRATED_CSV = "integration/integrated_incidents.csv"
os.makedirs("integration", exist_ok=True)

# ── Severity scoring ─────────────────────────────────────────────────────────
SEV_WEIGHT = {"High": 3, "Medium": 2, "Low": 1}

def compute_severity(row: pd.Series) -> str:
    score = 0
    urgency = float(row.get("Audio_Urgency_Score") or 0)
    score += 3 if urgency >= 0.8 else (2 if urgency >= 0.5 else 1)
    if str(row.get("Audio_Sentiment", "")).lower() == "distressed":
        score += 2
    score += SEV_WEIGHT.get(str(row.get("Text_Severity_Label", "")), 0)
    img_conf = float(row.get("Image_Confidence_Score") or 0)
    score += 2 if img_conf >= 0.85 else (1 if img_conf >= 0.5 else 0)
    vid_conf = float(row.get("Video_Confidence") or 0)
    score += 2 if vid_conf >= 0.85 else (1 if vid_conf >= 0.5 else 0)
    return "High" if score >= 8 else ("Medium" if score >= 5 else "Low")

# ── Helpers ──────────────────────────────────────────────────────────────────
def safe_load(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"  ✅ Loaded {len(df):>3} rows  ← {path}")
        return df
    print(f"  ⚠️  Not found: {path}  (run the module first)")
    return pd.DataFrame()

def val(df: pd.DataFrame, idx: int, col: str, default="N/A"):
    if df.empty or idx >= len(df):
        return default
    v = df.iloc[idx].get(col, default)
    return v if pd.notna(v) else default

# ── Main ─────────────────────────────────────────────────────────────────────
def run() -> pd.DataFrame:
    print("\n[Integration] ── Step 1: Loading all 5 module outputs ──")
    audio  = safe_load(OUTPUTS["audio"])
    pdf    = safe_load(OUTPUTS["pdf"])
    images = safe_load(OUTPUTS["images"])
    video  = safe_load(OUTPUTS["video"])
    text   = safe_load(OUTPUTS["text"])

    n = min(max(len(audio), len(pdf), len(images), len(video), len(text), 1), 20)

    print("\n[Integration] ── Step 2: Merging DataFrames on Incident_ID ──")
    rows = []
    for i in range(n):
        # Which modalities contributed to this incident
        sources = []
        if not audio.empty  and i < len(audio):  sources.append("Audio")
        if not pdf.empty    and i < len(pdf):     sources.append("PDF")
        if not images.empty and i < len(images):  sources.append("Image")
        if not video.empty  and i < len(video):   sources.append("Video")
        if not text.empty   and i < len(text):    sources.append("Text")

        # Best available top-level fields (assignment schema)
        event_val    = val(audio, i, "Extracted_Event")
        if event_val == "N/A":
            event_val = val(text, i, "Crime_Type")

        location_val = val(audio, i, "Location")
        if location_val in ("N/A", "Not mentioned"):
            location_val = val(text, i, "Location_Entity")
        if location_val in ("N/A", "Unknown"):
            location_val = val(pdf, i, "Location")

        time_val = val(video, i, "Timestamp")
        if time_val == "N/A":
            time_val = val(pdf, i, "Date")

        row = {
            # ── Assignment required columns ────────────────────────────────
            "Incident_ID":            f"INC_{i+1:03d}",
            "Source":                 " + ".join(sources) or "N/A",
            "Event":                  event_val,
            "Location":               location_val,
            "Time":                   time_val,
            # ── Audio ──────────────────────────────────────────────────────
            "Audio_Event":            val(audio,  i, "Extracted_Event"),
            "Audio_Location":         val(audio,  i, "Location"),
            "Audio_Sentiment":        val(audio,  i, "Sentiment"),
            "Audio_Urgency_Score":    val(audio,  i, "Urgency_Score", 0.0),
            # ── PDF ────────────────────────────────────────────────────────
            "PDF_Department":         val(pdf,    i, "Department"),
            "PDF_Incident_Type":      val(pdf,    i, "Incident_Type"),
            "PDF_Doc_Type":           val(pdf,    i, "Doc_Type"),
            "PDF_Date":               val(pdf,    i, "Date"),
            "PDF_Officer":            val(pdf,    i, "Officer"),
            "PDF_Key_Detail":         val(pdf,    i, "Key_Detail"),
            # ── Images ─────────────────────────────────────────────────────
            "Image_Scene_Type":       val(images, i, "Scene_Type"),
            "Image_Objects":          val(images, i, "Objects_Detected"),
            "Image_Text":             val(images, i, "Text_Extracted"),
            "Image_Confidence_Score": val(images, i, "Confidence_Score", 0.0),
            # ── Video ──────────────────────────────────────────────────────
            "Video_Clip":             val(video,  i, "Clip_ID"),
            "Video_Timestamp":        val(video,  i, "Timestamp"),
            "Video_Event":            val(video,  i, "Event_Detected"),
            "Video_Persons":          val(video,  i, "Persons_Count", 0),
            "Video_Confidence":       val(video,  i, "Confidence", 0.0),
            # ── Text ───────────────────────────────────────────────────────
            "Text_Crime_Type":        val(text,   i, "Crime_Type"),
            "Text_Location":          val(text,   i, "Location_Entity"),
            "Text_Sentiment":         val(text,   i, "Sentiment"),
            "Text_Severity_Label":    val(text,   i, "Severity_Label"),
            "Text_Source":            val(text,   i, "Source"),
        }

        print(f"\n[Integration] ── Step 3: Handling missing values ──")
        row["Severity"] = compute_severity(pd.Series(row))
        rows.append(row)

    print(f"\n[Integration] ── Step 4: Severity classification applied ──")
    df = pd.DataFrame(rows)
    df.fillna("N/A", inplace=True)

    print(f"\n[Integration] ── Step 5: Saving structured dataset ──")
    df.to_csv(INTEGRATED_CSV, index=False)

    print(f"\n[Integration] ✅ {len(df)} unified incidents → {INTEGRATED_CSV}")
    print("\n── Final Output (Assignment Schema) ──")
    print(df[["Incident_ID", "Source", "Event", "Location", "Time", "Severity"]].to_string(index=False))
    return df

if __name__ == "__main__":
    run()
