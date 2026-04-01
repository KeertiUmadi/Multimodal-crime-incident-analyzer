"""
Integration — Full Team
========================
Merges all 5 modality CSV outputs into one unified incident dataset.

Assignment steps:
  **Step 1** — `Incident_ID` on every modality (from `Report_ID`, `Call_ID`, `Text_ID`,
     `Image_ID`, or `Clip_ID` trailing digits; else stable order per modality).
  **Step 2** — `pandas.merge(..., on="Incident_ID", how="outer")` across five frames.
  **Step 3** — Missing text fields → `N/A` in the export columns below.
  **Step 4** — `Severity` ∈ {Low, Medium, High} via `compute_severity()`.
  **Step 5** — Streamlit `integration/dashboard.py` (primary). Optional CLI: `integrate.py --cli-query`.

Output  : integration/integration_output.csv
Schema  : Incident_ID | Audio_Event | PDF_Doc_Type | Image_Objects | Video_Event |
          Text_Crime_Type | Severity
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Optional

import pandas as pd


OUTPUTS = {
    "audio": "audio/output_audio.csv",
    "pdf": "pdf/output_pdf.csv",
    "images": "images/output_images.csv",
    "video": "video/output_video.csv",
    "text": "text/output_text.csv",
}

os.makedirs("integration", exist_ok=True)

SEV_WEIGHT = {"High": 3, "Medium": 2, "Low": 1}


def compute_severity(row: pd.Series) -> str:
    """
    Step 4 — Combined-signal severity (Low / Medium / High):
      - Audio urgency + distressed sentiment
      - Text severity label (High/Medium/Low weight)
      - Image and video detection confidence
      - PDF doc row present (non-N/A Doc_Type) as a weak extra modality signal
    """
    score = 0

    urgency = float(row.get("Audio_Urgency_Score", 0) or 0)
    score += 3 if urgency >= 0.8 else (2 if urgency >= 0.5 else 1)

    if str(row.get("Audio_Sentiment", "")).lower() == "distressed":
        score += 2

    txt_sev = str(row.get("Text_Severity_Label", "")).strip()
    if txt_sev.upper() != "N/A":
        score += SEV_WEIGHT.get(txt_sev, 0)

    img_conf = float(row.get("Image_Confidence_Score", 0) or 0)
    score += 2 if img_conf >= 0.85 else (1 if img_conf >= 0.5 else 0)

    vid_conf = float(row.get("Video_Confidence", 0) or 0)
    score += 2 if vid_conf >= 0.85 else (1 if vid_conf >= 0.5 else 0)

    pdf_dt = str(row.get("PDF_Doc_Type", "") or "").strip()
    if pdf_dt and pdf_dt.upper() != "N/A":
        score += 1

    return "High" if score >= 8 else ("Medium" if score >= 5 else "Low")


def _column_has_modality_data(series: pd.Series) -> pd.Series:
    """True where the cell is a real modality value (not blank / N/A)."""
    s = series.astype(str).str.strip()
    return ~(s.eq("") | s.str.upper().eq("N/A"))


# Whole-keyword only: maps user input → summary column (CLI keyword search).
_MODALITY_KEYWORD_TO_COL: dict[str, str] = {
    "audio": "Audio_Event",
    "pdf": "PDF_Doc_Type",
    "image": "Image_Objects",
    "images": "Image_Objects",
    "video": "Video_Event",
    "text": "Text_Crime_Type",
}


def filter_rows_by_keyword(df: pd.DataFrame, keyword: str) -> pd.DataFrame:
    """
    If keyword is exactly a modality name (audio, pdf, image, video, text), return rows
    where that column has non-N/A data. Otherwise substring-match (case-insensitive) across
    all five text columns (regex off so special characters are literal).
    """
    raw = keyword.strip()
    if not raw:
        return df.iloc[0:0]
    col = _MODALITY_KEYWORD_TO_COL.get(raw.lower())
    if col and col in df.columns:
        return df[_column_has_modality_data(df[col])]

    def _col_match(c: str) -> pd.Series:
        return df[c].astype(str).str.contains(raw, case=False, na=False, regex=False)

    mask = _col_match("Audio_Event") | _col_match("PDF_Doc_Type") | _col_match("Image_Objects")
    mask = mask | _col_match("Video_Event") | _col_match("Text_Crime_Type")
    return df[mask]


def safe_load(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"[Integration] Loaded {len(df)} rows <- {path}")
        return df
    print(f"[Integration] WARNING: Not found: {path}")
    return pd.DataFrame()


def _keys_to_incident_ids(series: pd.Series) -> pd.Series:
    """Map RPT_001 / C001 / TXT_001 / IMG_001 / clip12 → INC_001 (aligned across modalities)."""
    vals = series.astype(str).str.strip()

    def trailing_num(s: str) -> Optional[int]:
        m = re.search(r"(\d{1,4})\s*$", s)
        return int(m.group(1)) if m else None

    nums = [trailing_num(v) for v in vals]
    if all(n is not None for n in nums):
        return pd.Series([f"INC_{n:03d}" for n in nums], index=series.index)

    uniques = list(dict.fromkeys(vals.tolist()))
    mp = {u: f"INC_{i + 1:03d}" for i, u in enumerate(uniques)}
    return vals.map(mp)


def assign_incident_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1 — Ensure a single `Incident_ID` per row from the modality's native key.
    Priority: explicit Incident_ID → Report_ID (PDF) → Call_ID (audio) → Text_ID
    → Image_ID → sequential INC_###.
    """
    if df.empty:
        return df
    d = df.copy()
    if "Incident_ID" in d.columns:
        d["Incident_ID"] = d["Incident_ID"].astype(str).str.strip()
        return d

    for col in ("Report_ID", "Call_ID", "Text_ID", "Image_ID"):
        if col in d.columns:
            d["Incident_ID"] = _keys_to_incident_ids(d[col])
            return d

    d["Incident_ID"] = [f"INC_{i + 1:03d}" for i in range(len(d))]
    return d


def prepare_audio(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Incident_ID", "Audio_Event", "Audio_Sentiment", "Audio_Urgency_Score"])
    d = assign_incident_id_column(df)
    d = d.rename(
        columns={
            "Extracted_Event": "Audio_Event",
            "Sentiment": "Audio_Sentiment",
            "Urgency_Score": "Audio_Urgency_Score",
        }
    )
    keep = [c for c in ["Incident_ID", "Audio_Event", "Audio_Sentiment", "Audio_Urgency_Score"] if c in d.columns]
    return d[keep]


def prepare_pdf(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Incident_ID", "PDF_Doc_Type", "PDF_Date", "PDF_Location", "PDF_Summary"])
    d = df.copy().reset_index(drop=True)
    d = assign_incident_id_column(d)

    if "Doc_Type" in d.columns:
        d = d.rename(
            columns={
                "Doc_Type": "PDF_Doc_Type",
                "Date": "PDF_Date",
                "Department": "PDF_Location",
                "Key_Detail": "PDF_Summary",
            }
        )
    elif "Incident_Type" in d.columns:
        # Legacy PDF CSV shape
        d = d.rename(
            columns={
                "Incident_Type": "PDF_Doc_Type",
                "Date": "PDF_Date",
                "Location": "PDF_Location",
                "Officer": "PDF_Officer",
                "Summary": "PDF_Summary",
            }
        )
    keep_cols = [c for c in ["Incident_ID", "PDF_Doc_Type", "PDF_Date", "PDF_Location", "PDF_Officer", "PDF_Summary"] if c in d.columns]
    return d[keep_cols]


def prepare_images(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Incident_ID", "Image_Objects", "Image_Confidence_Score"])
    d = assign_incident_id_column(df)

    # objects
    if "Objects_Detected" in d.columns:
        d = d.rename(columns={"Objects_Detected": "Image_Objects_Detected"})

    # Primary: `Confidence_Score`; legacy CSVs may use `Confidence`
    if "Confidence_Score" in d.columns:
        d = d.rename(columns={"Confidence_Score": "Image_Confidence_Score"})
    elif "Confidence" in d.columns:
        d = d.rename(columns={"Confidence": "Image_Confidence_Score"})

    # keep for export
    if "Image_Objects_Detected" in d.columns and "Image_Confidence_Score" in d.columns:
        d["Image_Objects"] = d.apply(
            lambda r: "N/A"
            if pd.isna(r.get("Image_Objects_Detected")) or str(r.get("Image_Objects_Detected")).strip() in ("", "N/A", "None")
            else f"{r.get('Image_Objects_Detected')} ({float(r.get('Image_Confidence_Score') or 0):.2f})",
            axis=1,
        )
    elif "Image_Objects_Detected" in d.columns:
        d["Image_Objects"] = d["Image_Objects_Detected"].astype(str)
    else:
        d["Image_Objects"] = "N/A"

    keep_cols = [c for c in ["Incident_ID", "Image_Objects", "Image_Confidence_Score"] if c in d.columns or c == "Image_Objects"]
    return d[keep_cols]


def prepare_video(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Incident_ID", "Video_Event", "Video_Confidence"])
    d = df.copy()
    if "Clip_ID" not in d.columns:
        return pd.DataFrame(columns=["Incident_ID", "Video_Event", "Video_Confidence"])
    d = d.rename(
        columns={
            "Event_Detected": "Video_Event",
            "Confidence": "Video_Confidence",
        },
        errors="ignore",
    )
    if "Video_Confidence" not in d.columns and "Confidence" in d.columns:
        d = d.rename(columns={"Confidence": "Video_Confidence"})
    vc = d["Video_Confidence"] if "Video_Confidence" in d.columns else pd.Series(0.0, index=d.index)
    d["Video_Confidence"] = pd.to_numeric(vc, errors="coerce").fillna(0.0)
    # Many frame rows per clip — one incident summary row per clip (highest-confidence frame).
    idx = d.groupby("Clip_ID", sort=False)["Video_Confidence"].idxmax()
    d = d.loc[idx].reset_index(drop=True)
    d["Incident_ID"] = _keys_to_incident_ids(d["Clip_ID"])
    keep = [c for c in ["Incident_ID", "Video_Event", "Video_Confidence"] if c in d.columns]
    return d[keep]


def prepare_text(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Incident_ID", "Text_Crime_Type", "Text_Severity_Label"])
    d = assign_incident_id_column(df)
    d = d.rename(
        columns={
            "Crime_Type": "Text_Crime_Type",
            "Severity_Label": "Text_Severity_Label",
        }
    )
    keep = [c for c in ["Incident_ID", "Text_Crime_Type", "Text_Severity_Label"] if c in d.columns]
    return d[keep]


def merge_on_incident_id(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Step 2 — outer join on Incident_ID so every incident keeps all modality columns."""
    nonempty = [f for f in frames if not f.empty and "Incident_ID" in f.columns]
    if not nonempty:
        return pd.DataFrame(columns=["Incident_ID"])
    out = nonempty[0]
    for f in nonempty[1:]:
        out = pd.merge(out, f, on="Incident_ID", how="outer")
    return out


def run() -> pd.DataFrame:
    print("[Integration] Load modality CSVs (Step 1 prep: Incident_ID in each module)")
    audio = safe_load(OUTPUTS["audio"])
    pdf = safe_load(OUTPUTS["pdf"])
    images = safe_load(OUTPUTS["images"])
    video = safe_load(OUTPUTS["video"])
    text = safe_load(OUTPUTS["text"])

    print("[Integration] Step 2: merge outer on Incident_ID")
    merged = merge_on_incident_id(
        [
            prepare_audio(audio),
            prepare_pdf(pdf),
            prepare_images(images),
            prepare_video(video),
            prepare_text(text),
        ]
    )

    if merged.empty:
        final_df = pd.DataFrame(
            columns=[
                "Incident_ID",
                "Audio_Event",
                "PDF_Doc_Type",
                "Image_Objects",
                "Video_Event",
                "Text_Crime_Type",
                "Severity",
            ]
        )
        final_df.to_csv("integration/integration_output.csv", index=False)
        return final_df

    # Step 3 — text/object gaps → N/A (numeric columns filled for scoring below)
    obj_cols = merged.select_dtypes(include=["object"]).columns
    merged[obj_cols] = merged[obj_cols].fillna("N/A")

    # Convert numeric confidence columns to float for severity scoring
    if "Image_Confidence_Score" in merged.columns:
        merged["Image_Confidence_Score"] = pd.to_numeric(merged["Image_Confidence_Score"], errors="coerce").fillna(0.0)
    if "Video_Confidence" in merged.columns:
        merged["Video_Confidence"] = pd.to_numeric(merged["Video_Confidence"], errors="coerce").fillna(0.0)
    if "Audio_Urgency_Score" in merged.columns:
        merged["Audio_Urgency_Score"] = pd.to_numeric(merged["Audio_Urgency_Score"], errors="coerce").fillna(0.0)

    print("[Integration] Step 4: severity classification")
    merged["Severity"] = merged.apply(compute_severity, axis=1)

    final_df = pd.DataFrame(
        {
            "Incident_ID": merged.get("Incident_ID", pd.Series([], dtype=str)),
            "Audio_Event": merged.get("Audio_Event", pd.Series(["N/A"] * len(merged))).apply(lambda x: "N/A" if pd.isna(x) else str(x)),
            "PDF_Doc_Type": merged.get("PDF_Doc_Type", pd.Series(["N/A"] * len(merged))).apply(lambda x: "N/A" if pd.isna(x) else str(x)),
            "Image_Objects": merged.get("Image_Objects", pd.Series(["N/A"] * len(merged))).apply(lambda x: "N/A" if pd.isna(x) else str(x)),
            "Video_Event": merged.get("Video_Event", pd.Series(["N/A"] * len(merged))).apply(
                lambda x: "N/A" if pd.isna(x) else str(x)
            ),
            "Text_Crime_Type": merged.get("Text_Crime_Type", pd.Series(["N/A"] * len(merged))).apply(
                lambda x: "N/A" if pd.isna(x) else str(x)
            ),
            "Severity": merged["Severity"],
        }
    )

    final_df.to_csv("integration/integration_output.csv", index=False)
    print("[Integration] Step 5: `streamlit run integration/dashboard.py` (optional: `--cli-query`).")
    print(f"[Integration] Saved -> integration/integration_output.csv ({len(final_df)} rows)")
    return final_df


def query_interface(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("INCIDENT QUERY INTERFACE")
    print("=" * 60)

    while True:
        print("\nOPTIONS:")
        print("  [1] Filter by Severity")
        print("  [2] Filter by keyword (text in columns, or: audio/pdf/image/video/text)")
        print("  [3] Filter by Crime Type")
        print("  [4] Show all incidents")
        print("  [5] Show summary statistics")
        print("  [6] Exit")

        choice = input("\nEnter choice (1-6): ").strip()

        if choice == "1":
            sev = input("Severity (High/Medium/Low): ").strip().capitalize()
            result = df[df["Severity"] == sev]
            print(f"\nFound {len(result)} incidents with severity: {sev}")
            print(result.to_string(index=False))
        elif choice == "2":
            event = input(
                "Keyword — substring in any column, or alone: audio / pdf / image / video / text: "
            ).strip()
            result = filter_rows_by_keyword(df, event)
            print(f"\nFound {len(result)} incidents matching: {event}")
            print(result.to_string(index=False))
        elif choice == "3":
            crime = input("Crime type (e.g. Theft/Murder/Drug): ").strip()
            result = df[df["Text_Crime_Type"].astype(str).str.contains(crime, case=False, na=False)]
            print(f"\nFound {len(result)} incidents with crime type: {crime}")
            print(result.to_string(index=False))
        elif choice == "4":
            print(f"\nAll {len(df)} incidents:")
            print(df.to_string(index=False))
        elif choice == "5":
            print("\n-- Summary Statistics --")
            print(f"Total Incidents : {len(df)}")
            print(f"High Severity   : {len(df[df['Severity'] == 'High'])}")
            print(f"Medium Severity : {len(df[df['Severity'] == 'Medium'])}")
            print(f"Low Severity    : {len(df[df['Severity'] == 'Low'])}")
        elif choice == "6":
            print("\nExiting query interface. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1-6.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge modality CSVs into integration_output.csv")
    parser.add_argument(
        "--cli-query",
        action="store_true",
        help="After merge, open the text-based query menu (default: CSV only; use Streamlit dashboard for Step 5).",
    )
    args = parser.parse_args()

    out = run()
    if args.cli_query:
        query_interface(out)

