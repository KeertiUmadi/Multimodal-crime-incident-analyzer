"""
Integration — Full Team
========================
Merges all 5 modality CSV outputs into one unified incident dataset.

Assignment Steps:
  1. Define common Incident_ID across all 5 CSVs
  2. Merge DataFrames on Incident_ID
  3. Handle missing values with N/A
  4. Generate severity classification (Low / Medium / High)
  5. Save integration_output.csv

Output  : integration/integration_output.csv
Final Schema : Incident_ID | Audio_Event | PDF_Doc_Type | Image_Objects | Video_Event | Text_Crime_Type | Severity
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
os.makedirs("integration", exist_ok=True)

# ── Severity scoring ──────────────────────────────────────────────────────────
SEV_WEIGHT = {"High": 3, "Medium": 2, "Low": 1}

def compute_severity(row: pd.Series) -> str:
    score = 0

    urgency = float(row.get("Audio_Urgency_Score") or 0)
    score += 3 if urgency >= 0.8 else (2 if urgency >= 0.5 else 1)

    if str(row.get("Audio_Sentiment", "")).lower() == "distressed":
        score += 2

    score += SEV_WEIGHT.get(str(row.get("Text_Severity_Label", "")), 0)

    img_conf = float(row.get("Image_Confidence") or 0)
    score += 2 if img_conf >= 0.85 else (1 if img_conf >= 0.5 else 0)

    vid_conf = float(row.get("Video_Confidence") or 0)
    score += 2 if vid_conf >= 0.85 else (1 if vid_conf >= 0.5 else 0)

    return "High" if score >= 8 else ("Medium" if score >= 5 else "Low")

# ── Helpers ───────────────────────────────────────────────────────────────────
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

def clean(v, default="Unknown"):
    if pd.isna(v) or str(v).strip() in ["", "N/A"]:
        return default
    return str(v)

# ── Main ──────────────────────────────────────────────────────────────────────
def run() -> pd.DataFrame:
    print("\n[Integration] ── Step 1: Loading all 5 module outputs ──")
    audio  = safe_load(OUTPUTS["audio"])
    pdf    = safe_load(OUTPUTS["pdf"])
    images = safe_load(OUTPUTS["images"])
    video  = safe_load(OUTPUTS["video"])
    text   = safe_load(OUTPUTS["text"])

    n = max(len(audio), len(pdf), len(images), len(video), len(text), 1)

    print("\n[Integration] ── Step 2: Merging DataFrames on Incident_ID ──")
    rows = []
    for i in range(n):
        sources = []
        if not audio.empty  and i < len(audio):  sources.append("Audio")
        if not pdf.empty    and i < len(pdf):     sources.append("PDF")
        if not images.empty and i < len(images):  sources.append("Image")
        if not video.empty  and i < len(video):   sources.append("Video")
        if not text.empty   and i < len(text):    sources.append("Text")

        event_val = val(audio, i, "Extracted_Event")
        if event_val == "N/A":
            event_val = val(text, i, "Crime_Type")

        location_val = val(audio, i, "Location")
        if location_val in ("N/A", "Not mentioned"):
            location_val = val(text, i, "Location_Entity")
        if location_val in ("N/A", "Unknown"):
            location_val = val(pdf, i, "Department")

        time_val = val(video, i, "Timestamp")
        if time_val == "N/A":
            time_val = val(pdf, i, "Date")

        img_objects = val(images, i, "Objects_Detected")
        img_conf    = val(images, i, "Confidence", 0.0)
        img_display = f"{img_objects} ({img_conf})" if img_objects not in ("N/A", "None", "") else "N/A"

        row = {
            "Incident_ID":         f"INC_{i+1:03d}",
            "Source":              " + ".join(sources) or "N/A",
            "Event":               event_val,
            "Location":            location_val,
            "Time":                time_val,
            "Audio_Event":         val(audio, i, "Extracted_Event"),
            "Audio_Location":      val(audio, i, "Location"),
            "Audio_Sentiment":     val(audio, i, "Sentiment"),
            "Audio_Urgency_Score": val(audio, i, "Urgency_Score", 0.0),
            "PDF_Department":      val(pdf, i, "Department"),
            "PDF_Doc_Type":        val(pdf, i, "Doc_Type"),
            "PDF_Date":            val(pdf, i, "Date"),
            "PDF_Program":         val(pdf, i, "Program"),
            "PDF_Key_Detail":      val(pdf, i, "Key_Detail"),
            "Image_Scene_Type":    val(images, i, "Scene_Type"),
            "Image_Objects":       img_display,
            "Image_Bboxes":        val(images, i, "Bounding_Boxes"),
            "Image_Confidence":    img_conf,
            "Video_Clip":          val(video, i, "Clip_ID"),
            "Video_Timestamp":     val(video, i, "Timestamp"),
            "Video_Event":         val(video, i, "Event_Detected"),
            "Video_Persons":       val(video, i, "Persons_Count", "0 persons"),
            "Video_Confidence":    val(video, i, "Confidence", 0.0),
            "Text_Crime_Type":     val(text, i, "Crime_Type"),
            "Text_Location":       val(text, i, "Location_Entity"),
            "Text_Sentiment":      val(text, i, "Sentiment"),
            "Text_Topic":          val(text, i, "Topic"),
            "Text_Severity_Label": val(text, i, "Severity_Label"),
        }

        row["Severity"] = compute_severity(pd.Series(row))
        rows.append(row)

    print(f"\n[Integration] ── Step 3: Handling missing values ──")
    df = pd.DataFrame(rows)
    df.fillna("N/A", inplace=True)

    print(f"\n[Integration] ── Step 4: Severity classification applied ──")
    print(f"\n[Integration] ── Step 5: Saving final dataset ──")

    # ── FINAL DATASET (ASSIGNMENT FORMAT) ───────────────────────────────────
    final_df = pd.DataFrame({
        "Incident_ID":     df["Incident_ID"],
        "Audio_Event":     df["Audio_Event"].apply(lambda x: clean(x)),
        "PDF_Doc_Type":    df["PDF_Doc_Type"].apply(lambda x: clean(x)),
        "Image_Objects":   df["Image_Objects"].apply(lambda x: clean(x)),
        "Video_Event":     df["Video_Event"].apply(lambda x: clean(x, "No significant event")),
        "Text_Crime_Type": df["Text_Crime_Type"].apply(lambda x: clean(x)),
        "Severity":        df["Severity"],
    })

    final_df.to_csv("integration/integration_output.csv", index=False)
    print("\n[Integration] Final dataset saved → integration/integration_output.csv")
    print("\n── Final Output (Assignment Format) ──")
    print(final_df.to_string(index=False))

    return final_df

# ── Terminal Query Interface ──────────────────────────────────────────────────
def query_interface(df: pd.DataFrame):
    print("\n" + "="*60)
    print("  🔍 INCIDENT QUERY INTERFACE")
    print("="*60)
    print("  Filter and search the integrated incident dataset.")
    print("  Press Enter to skip a filter (show all).")
    print("="*60)

    while True:
        print("\n  OPTIONS:")
        print("  [1] Filter by Severity")
        print("  [2] Filter by Event Type")
        print("  [3] Filter by Crime Type")
        print("  [4] Show all incidents")
        print("  [5] Show summary statistics")
        print("  [6] Exit")

        choice = input("\n  Enter choice (1-6): ").strip()

        if choice == "1":
            sev = input("  Severity (High/Medium/Low): ").strip().capitalize()
            result = df[df["Severity"] == sev]
            print(f"\n  Found {len(result)} incidents with severity: {sev}")
            print(result.to_string(index=False))

        elif choice == "2":
            event = input("  Event type (e.g. Shooting/Fire/Robbery): ").strip()
            result = df[
                df["Audio_Event"].str.contains(event, case=False, na=False) |
                df["Image_Objects"].str.contains(event, case=False, na=False) |
                df["Video_Event"].str.contains(event, case=False, na=False)
            ]
            print(f"\n  Found {len(result)} incidents with event: {event}")
            print(result.to_string(index=False))

        elif choice == "3":
            crime = input("  Crime type (e.g. Theft/Murder/Drug): ").strip()
            result = df[df["Text_Crime_Type"].str.contains(crime, case=False, na=False)]
            print(f"\n  Found {len(result)} incidents with crime type: {crime}")
            print(result.to_string(index=False))

        elif choice == "4":
            print(f"\n  All {len(df)} incidents:")
            print(df.to_string(index=False))

        elif choice == "5":
            print("\n  ── Summary Statistics ──")
            print(f"  Total Incidents : {len(df)}")
            print(f"  High Severity   : {len(df[df['Severity'] == 'High'])}")
            print(f"  Medium Severity : {len(df[df['Severity'] == 'Medium'])}")
            print(f"  Low Severity    : {len(df[df['Severity'] == 'Low'])}")
            print(f"\n  Top Audio Events:")
            print(df["Audio_Event"].value_counts().head(5).to_string())
            print(f"\n  Top Crime Types:")
            print(df["Text_Crime_Type"].value_counts().head(5).to_string())

        elif choice == "6":
            print("\n  Exiting query interface. Goodbye!")
            break

        else:
            print("  Invalid choice. Please enter 1-6.")

if __name__ == "__main__":
    df = run()
    query_interface(df)