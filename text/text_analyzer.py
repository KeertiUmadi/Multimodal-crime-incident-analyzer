"""
Student 5 — Text Analyst
=========================
CrimeReport (Kaggle) — real crime text reports.

  Dataset : https://www.kaggle.com/datasets/cameliasiadat/crimereport
  Place   : text/data/crimereport.txt, CrimeReport.txt (Kaggle), or crimereport.csv
            If missing: pip install kagglehub + Kaggle API token → auto-download via
            kagglehub.dataset_download("cameliasiadat/crimereport")

Pipeline: spaCy NER (locations) · HuggingFace sentiment (3-class) · HF zero-shot topic

Output  : text/output_text.csv
Columns : Text_ID, Crime_Type, Location_Entity, Sentiment, Topic, Severity_Label
"""

from __future__ import annotations

import csv
import json
import os
import re
from typing import List

import nltk
import pandas as pd

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(_SCRIPT_DIR, "data")
DEFAULT_OUTPUT_CSV = os.path.join(_SCRIPT_DIR, "output_text.csv")

STOP_WORDS = set(stopwords.words("english"))

TOPIC_ZS_LABELS = [
    "Accident",
    "Fire",
    "Theft/Robbery",
    "Assault",
    "Drug Activity",
    "Disturbance",
]


def _load_models():
    import spacy
    from transformers import pipeline as hf_pipeline

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    print("[Text] Loading sentiment model (3-class)...")
    sent_pipe = hf_pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        truncation=True,
        max_length=512,
    )
    print("[Text] Loading zero-shot topic classifier ...")
    zs_pipe = hf_pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
    )
    return nlp, sent_pipe, zs_pipe


def clean_text(text: str) -> str:
    """Lowercase, strip noise/URLs, remove punctuation, NLTK tokenize, drop stopwords."""
    if not text or not str(text).strip():
        return ""
    t = str(text).lower()
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    tokens = word_tokenize(t)
    return " ".join(w for w in tokens if w not in STOP_WORDS and len(w) > 2)


def extract_entities(raw_text: str, nlp) -> List[str]:
    """spaCy NER: GPE, LOC, FAC from original-casing text."""
    if not raw_text or not raw_text.strip():
        return []
    doc = nlp(str(raw_text)[:5000])
    locs = list(
        dict.fromkeys(
            e.text.strip() for e in doc.ents if e.label_ in ("GPE", "LOC", "FAC")
        )
    )
    return locs[:5]


def detect_crime_type(cleaned: str) -> str:
    """Keyword map → Robbery, Theft, Fire, Accident, Assault, Drug Activity, Disturbance, else Unknown."""
    if not cleaned:
        return "Unknown"
    tl = cleaned.lower()
    rules = [
        ("robbery", "Robbery"),
        ("theft", "Theft"),
        ("burglar", "Theft"),
        ("burglary", "Theft"),
        ("stolen", "Theft"),
        ("fire", "Fire"),
        ("arson", "Fire"),
        ("blaze", "Fire"),
        ("accident", "Accident"),
        ("crash", "Accident"),
        ("collision", "Accident"),
        ("shooting", "Assault"),
        ("shoot ", "Assault"),
        (" shooter", "Assault"),
        ("shot ", "Assault"),
        ("assault", "Assault"),
        ("attack", "Assault"),
        ("fight", "Assault"),
        ("drug", "Drug Activity"),
        ("narcotic", "Drug Activity"),
        ("overdose", "Drug Activity"),
        ("disturbance", "Disturbance"),
        ("disorderly", "Disturbance"),
    ]
    for kw, label in rules:
        if kw in tl:
            return label
    return "Unknown"


def get_sentiment(raw_text: str, sent_pipe) -> str:
    """HuggingFace sentiment → Positive / Negative / Neutral only."""
    if not raw_text or not str(raw_text).strip():
        return "Neutral"
    r = sent_pipe(str(raw_text)[:512])[0]
    lab = str(r.get("label", "")).lower()
    if "neutral" in lab or "label_1" in lab:
        return "Neutral"
    if "neg" in lab or "label_0" in lab:
        return "Negative"
    if "pos" in lab or "label_2" in lab:
        return "Positive"
    return "Neutral"


def classify_topic(raw_text: str, zs_pipe) -> str:
    """Zero-shot; labels fixed list; highest score wins."""
    if not raw_text or not str(raw_text).strip():
        return "Disturbance"
    r = zs_pipe(str(raw_text)[:512], candidate_labels=TOPIC_ZS_LABELS)
    return r["labels"][0]


def assign_severity(text_lower: str) -> str:
    """High / Medium / Low from custom keyword logic."""
    if not text_lower:
        return "Low"
    tl = text_lower
    high_kw = (
        "shooting",
        "shoot ",
        " shot",
        "fire",
        "weapon",
        " gun",
        "knife attack",
        "severe injury",
        "critically",
        "homicide",
        "murder",
        "killed",
        "dead",
        "bomb",
        "explosion",
    )
    if any(k in tl for k in high_kw):
        return "High"
    med_kw = (
        "robbery",
        "assault",
        "drug",
        "narcotic",
        "stabbing",
        "rape",
        "kidnap",
    )
    if any(k in tl for k in med_kw):
        return "Medium"
    low_kw = ("disturbance", "minor", "noise complaint", "loud")
    if any(k in tl for k in low_kw):
        return "Low"
    return "Low"


def _parse_txt_line(line: str) -> str:
    line = line.strip()
    if not line:
        return ""
    try:
        row = next(csv.reader([line]))
    except Exception:
        return line
    if len(row) <= 1:
        return row[0].strip() if row else ""
    chunks = [c.strip() for c in row if len(c.strip()) > 8]
    if not chunks:
        return " ".join(row).strip()
    return max(chunks, key=len)


def _tweet_text_from_json_obj(obj: dict) -> str:
    """Kaggle CrimeReport .txt is JSON Lines (Twitter export): use `text` field."""
    t = (obj.get("text") or "").strip()
    if not t and isinstance(obj.get("retweeted_status"), dict):
        t = (obj["retweeted_status"].get("text") or "").strip()
    return t


def _read_records_from_file(path: str) -> List[str]:
    """Load text rows from one .csv or .txt path (.txt may be plain lines or JSON Lines)."""
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
        col = "text" if "text" in df.columns else df.columns[0]
        return [
            str(x).strip()
            for x in df[col].tolist()
            if pd.notna(x) and str(x).strip()
        ]
    records: List[str] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        pos0 = f.tell()
        first_nonempty = ""
        for line in f:
            s = line.strip()
            if s:
                first_nonempty = s
                break
        if first_nonempty.startswith("{"):
            f.seek(pos0)
            for line in f:
                s = line.strip()
                if not s or not s.startswith("{"):
                    continue
                try:
                    obj = json.loads(s)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                t = _tweet_text_from_json_obj(obj)
                if t:
                    records.append(t)
            return records
        f.seek(pos0)
        for line in f:
            rec = _parse_txt_line(line)
            if rec:
                records.append(rec)
    return records


def _scan_dataset_directory(root: str) -> List[str] | None:
    """Find .csv / .txt under a Kaggle extract folder and load the best match."""
    csv_paths: list[str] = []
    txt_paths: list[str] = []
    for walk_root, _, files in os.walk(root):
        for fn in files:
            fp = os.path.join(walk_root, fn)
            low = fn.lower()
            if low.endswith(".csv"):
                csv_paths.append(fp)
            elif low.endswith(".txt"):
                txt_paths.append(fp)

    def csv_sort_key(p: str) -> tuple:
        base = os.path.basename(p).lower()
        pri = 0 if "crime" in base or "report" in base else 1
        return (pri, -os.path.getsize(p))

    csv_paths.sort(key=csv_sort_key)
    for fp in csv_paths:
        try:
            out = _read_records_from_file(fp)
            if out:
                print(f"[Text] Using dataset file: {fp}")
                return out
        except Exception:
            continue

    txt_paths.sort(key=lambda p: -os.path.getsize(p))
    for fp in txt_paths:
        try:
            out = _read_records_from_file(fp)
            if out:
                print(f"[Text] Using dataset file: {fp}")
                return out
        except Exception:
            continue
    return None


def load_input_records(data_dir: str) -> List[str]:
    """Local crimereport.txt / .csv, or download with kagglehub if missing."""
    os.makedirs(data_dir, exist_ok=True)
    for name in ("crimereport.txt", "CrimeReport.txt", "crimereport.csv"):
        path = os.path.join(data_dir, name)
        if os.path.isfile(path):
            print(f"[Text] Loading: {path}")
            rec = _read_records_from_file(path)
            if rec:
                return rec

    try:
        import kagglehub

        print("[Text] No local crimereport.* — downloading via kagglehub ...")
        kaggle_path = kagglehub.dataset_download("cameliasiadat/crimereport")
        print(f"[Text] Path to dataset files: {kaggle_path}")
        got = _scan_dataset_directory(kaggle_path)
        if got:
            return got
    except ImportError:
        print("[Text] Optional: pip install kagglehub for automatic download.")
    except Exception as e:
        print(f"[Text] kagglehub error: {e}")

    raise FileNotFoundError(
        f"[Text] No usable data in {data_dir}.\n"
        f"  • Add text/data/crimereport.txt, CrimeReport.txt, or crimereport.csv, or\n"
        f"  • pip install kagglehub and configure Kaggle API (kaggle.json), or\n"
        f"  • Download manually: https://www.kaggle.com/datasets/cameliasiadat/crimereport"
    )


def run(
    text_dir: str | None = None,
    output_csv: str | None = None,
) -> pd.DataFrame:
    text_dir = text_dir or DEFAULT_DATA_DIR
    output_csv = output_csv or DEFAULT_OUTPUT_CSV

    records = load_input_records(text_dir)
    if not records:
        df = pd.DataFrame(
            columns=[
                "Text_ID",
                "Crime_Type",
                "Location_Entity",
                "Sentiment",
                "Topic",
                "Severity_Label",
            ]
        )
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"[Text] No rows → wrote empty schema to {output_csv}")
        return df

    nlp, sent_pipe, zs_pipe = _load_models()
    rows: list[dict] = []

    for i, raw in enumerate(records):
        raw = str(raw).strip()
        if not raw:
            continue
        tid = f"TXT_{i + 1:03d}"
        print(f"[Text] {tid} ...")

        cleaned = clean_text(raw)
        locs = extract_entities(raw, nlp)
        location = ", ".join(locs) if locs else "Not mentioned"
        crime = detect_crime_type(cleaned) if cleaned else "Unknown"
        if crime == "Unknown":
            crime = detect_crime_type(raw.lower()) or "Unknown"
        sent = get_sentiment(raw, sent_pipe)
        topic = classify_topic(raw, zs_pipe)
        sev = assign_severity(raw.lower())

        rows.append(
            {
                "Text_ID": tid,
                "Crime_Type": crime if crime else "Unknown",
                "Location_Entity": location if location else "Not mentioned",
                "Sentiment": sent if sent else "Neutral",
                "Topic": topic if topic else "Disturbance",
                "Severity_Label": sev if sev else "Low",
            }
        )

    output_columns = [
        "Text_ID",
        "Crime_Type",
        "Location_Entity",
        "Sentiment",
        "Topic",
        "Severity_Label",
    ]
    df = pd.DataFrame(rows)[output_columns]
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[Text] {len(df)} records → {output_csv}\n")
    return df


if __name__ == "__main__":
    run()
