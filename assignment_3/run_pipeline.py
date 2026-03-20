"""
run_pipeline.py — One-command full pipeline runner
===================================================
Runs all 5 modality analyzers then integration, then launches dashboard.

Usage:
    python run_pipeline.py                              # run everything
    python run_pipeline.py --only text integration      # specific modules
    python run_pipeline.py --no-dashboard               # skip Streamlit
    python run_pipeline.py --demo                       # run demo.py after
"""

import argparse
import subprocess
import sys

STEPS = [
    ("audio",       "audio/audio_analyzer.py"),
    ("pdf",         "pdf/pdf_analyzer.py"),
    ("images",      "images/image_analyzer.py"),
    ("video",       "video/video_analyzer.py"),
    ("text",        "text/text_analyzer.py"),
    ("integration", "integration/integrate.py"),
]

def run_step(name: str, script: str):
    print(f"\n{'='*60}")
    print(f"  MODULE: {name.upper()}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"  ⚠️  {name} finished with errors — continuing...")

def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Crime Incident Analyzer — Pipeline Runner"
    )
    parser.add_argument("--only", nargs="+",
                        choices=[s[0] for s in STEPS],
                        help="Run only these modules")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Skip Streamlit dashboard launch")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo.py after pipeline")
    args = parser.parse_args()

    targets = args.only or [s[0] for s in STEPS]

    print("\n🚨 MULTIMODAL CRIME INCIDENT ANALYZER — PIPELINE START")
    print(f"   Modules: {targets}\n")
    print("   ℹ️  No data files? Each module uses built-in demo data automatically.\n")

    for name, script in STEPS:
        if name in targets:
            run_step(name, script)

    if args.demo:
        print("\n\n🎬 Running demonstration script ...")
        subprocess.run([sys.executable, "demo.py"])

    if not args.no_dashboard and "integration" in targets:
        print("\n\n🖥️  Launching Streamlit Dashboard ...")
        print("   Open your browser at http://localhost:8501")
        print("   Press Ctrl+C to stop.\n")
        subprocess.run(["streamlit", "run", "integration/dashboard.py"])

if __name__ == "__main__":
    main()
