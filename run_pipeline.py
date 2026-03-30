"""
run_pipeline.py — One-command full pipeline runner
===================================================
Runs all five modality analyzers, then integration (merged CSV + severity).

Step 5 (display / filter incident summaries): streamlit run integration/dashboard.py

Usage:
    python run_pipeline.py                              # run everything
    python run_pipeline.py --only text integration      # specific modules
"""

import argparse
import subprocess
import sys
import os

STEPS = [
    ("audio",       "audio/audio_analyzer.py"),
    ("pdf",         "pdf/pdf_analyzer.py"),
    ("images",      "images/image_analyzer.py"),
    ("video",       "video/video_analyzer.py"),
    ("text",        "text/text_analyzer.py"),
    ("integration", "integration/integrate.py"),
]
FINAL_CSV = "integration/integration_output.csv"

def run_step(name: str, script: str):
    print(f"\n{'='*60}")
    print(f"  MODULE: {name.upper()}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable] + script.split())
    if result.returncode != 0:
        print(f" {name} finished with errors - continuing...")
    else:
        print(f" [OK] {name} complete.")

def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Crime Incident Analyzer — Pipeline Runner"
    )
    parser.add_argument("--only", nargs="+",
                        choices=[s[0] for s in STEPS],
                        help="Run only these modules")
    args = parser.parse_args()

    targets = args.only or [s[0] for s in STEPS]

    print("\n MULTIMODAL CRIME INCIDENT ANALYZER — PIPELINE START")
    print(f"   Modules: {targets}\n")

    for name, script in STEPS:
        if name in targets:
            run_step(name, script)
    
    print(f"\n{'='*60}")
    print("  [OK] PIPELINE COMPLETE")
    print(f"{'='*60}")

    if os.path.exists(FINAL_CSV):
        print(f"\nMerged incidents: {FINAL_CSV}")
        print(
            "\nStep 5 — Dashboard (display / filter summaries):\n"
            "  streamlit run integration/dashboard.py\n"
            "Opening CSV in the default app…"
        )
        try:
            os.startfile(os.path.abspath(FINAL_CSV))
        except OSError:
            pass
    else:
        print(f"\nFinal CSV not found: {FINAL_CSV}")

if __name__ == "__main__":
    main()
