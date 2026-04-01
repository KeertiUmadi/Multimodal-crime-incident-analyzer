"""
Deliverable 5 helper: print how each unstructured source becomes structured data,
then show sample rows from modality CSVs and the merged incident report.

Run from repo root (after pipeline if needed):
  python scripts/demo_story.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _first_glob(rel_dir: str, patterns: tuple[str, ...]) -> str | None:
    base = ROOT / rel_dir
    if not base.is_dir():
        return None
    for pat in patterns:
        matches = sorted(base.glob(pat))
        if matches:
            return str(matches[0].relative_to(ROOT))
    return None


def _preview_csv(rel_path: str, max_chars: int = 200) -> None:
    p = ROOT / rel_path
    if not p.is_file():
        print("  (file missing - run: python run_pipeline.py)")
        return
    df = pd.read_csv(p)
    print(f"  Rows: {len(df)}  |  Columns: {list(df.columns)}")
    if df.empty:
        return
    row = df.iloc[0].to_dict()
    for k, v in row.items():
        s = str(v).replace("\n", " ").strip()
        if len(s) > max_chars:
            s = s[: max_chars - 3] + "..."
        print(f"    {k}: {s}")


def main() -> None:
    print("=" * 72)
    print("DEMONSTRATION STORYLINE - Unstructured inputs -> Structured incident report")
    print("=" * 72)

    print(
        """
WHAT TO SAY (short version for one presenter):
  We ingest five kinds of raw evidence (audio, PDFs, images, video, and text).
  Each folder runs its own analyzer script. Each script writes one CSV with
  structured fields. Integration merges those CSVs on Incident_ID, adds
  Severity, and produces one final incident table. The Streamlit dashboard
  lets you filter and export that report.
"""
    )

    flows = [
        (
            "AUDIO",
            "Emergency call recordings (.wav / .mp3 in audio/data/)",
            ("*.wav", "*.mp3", "*.m4a"),
            "audio/audio_analyzer.py",
            "Whisper transcription -> event, location, sentiment, urgency",
            "audio/output_audio.csv",
        ),
        (
            "PDF",
            "Police / agency PDFs (pdf/data/*.pdf)",
            ("*.pdf",),
            "pdf/pdf_analyzer.py",
            "Text + OCR -> department, doc type, date, program, key detail",
            "pdf/output_pdf.csv",
        ),
        (
            "IMAGES",
            "Scene photos (images/data/*.jpg, *.png; optional YOLO labels in images/labels/)",
            ("*.jpg", "*.jpeg", "*.png"),
            "images/image_analyzer.py",
            "Object detection -> scene type, objects, boxes, confidence score",
            "images/output_images.csv",
        ),
        (
            "VIDEO",
            "CCTV clips (video/data/*.mpg, *.mp4, ...)",
            ("*.mpg", "*.mp4", "*.avi"),
            "video/video_analyzer.py",
            "Sampled frames + motion + YOLO -> events, person counts, confidence",
            "video/output_video.csv",
        ),
        (
            "TEXT",
            "Unstructured crime text (text/data/CrimeReport.txt or .csv from Kaggle)",
            ("*.txt", "*.csv"),
            "text/text_analyzer.py",
            "NER + sentiment + zero-shot topic -> crime type, location, severity label",
            "text/output_text.csv",
        ),
    ]

    for name, desc, pats, script, transform, out_csv in flows:
        rel_dir = str(Path(out_csv).parent / "data")
        example = _first_glob(rel_dir, pats)
        print(f"\n--- {name} ---")
        print(f"  Raw:        {desc}")
        if example:
            print(f"  Example:    {example}")
        print(f"  Script:     {script}")
        print(f"  Transform:  {transform}")
        print(f"  Output:     {out_csv}")
        print("  Sample row (first row of output CSV):")
        _preview_csv(out_csv)

    print("\n--- INTEGRATION (merged structured incident report) ---")
    print("  Script:     integration/integrate.py")
    print("  Transform:  Outer-merge all modality CSVs on Incident_ID -> Severity")
    print("  Output:     integration/integration_output.csv")
    print("  Sample row:")
    _preview_csv("integration/integration_output.csv", max_chars=120)

    print("\n--- STEP 5 - DASHBOARD ---")
    print("  Run:        streamlit run integration/dashboard.py")
    print("  Or:         run_dashboard.bat")
    print("  Shows:      Filter by severity, search field + value, chart, download CSV.")
    print("=" * 72)


if __name__ == "__main__":
    main()
