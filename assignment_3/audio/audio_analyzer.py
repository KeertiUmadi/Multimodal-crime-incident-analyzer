"""
Student 1 — Audio Analyst
=========================
Converts emergency audio calls to text using Whisper,
extracts incident keywords + locations with spaCy NER,
scores urgency and classifies caller sentiment.

Dataset : 911 Calls + Wav2Vec2 (Kaggle)
Link    : https://www.kaggle.com/code/stpeteishii/911-calls-wav2vec2
Place   : audio/data/*.wav  (also supports .mp3 .m4a .ogg .flac)

Output  : audio/output_audio.csv
Columns : Call_ID, Source_File, Transcript, Extracted_Event,
          Location, Sentiment, Urgency_Score
"""

import os
import pandas as pd

# ── Keyword maps ────────────────────────────────────────────────────────────
INCIDENT_KEYWORDS = {
    "Fire":        ["fire", "flames", "burning", "smoke", "trapped"],
    "Accident":    ["crash", "collision", "accident", "injured", "hit"],
    "Theft":       ["robbery", "stolen", "theft", "burglar", "broke in"],
    "Assault":     ["fight", "assault", "attack", "weapon", "gun", "knife"],
    "Medical":     ["heart", "breathing", "unconscious", "bleeding", "ambulance"],
    "Disturbance": ["noise", "disturbance", "argument", "loud", "crowd"],
}

URGENCY_PHRASES = [
    "help", "emergency", "hurry", "please", "dying", "now", "quick",
    "trapped", "gun", "shooting", "bleeding", "unconscious", "fire", "danger",
]

# ── Helpers ─────────────────────────────────────────────────────────────────
def extract_event(text: str) -> str:
    tl = text.lower()
    for event, kws in INCIDENT_KEYWORDS.items():
        if any(k in tl for k in kws):
            return event
    return "Unknown"

def extract_location(text: str, nlp) -> str:
    doc = nlp(text)
    locs = [e.text for e in doc.ents if e.label_ in ("GPE", "LOC", "FAC")]
    return ", ".join(dict.fromkeys(locs)) or "Not mentioned"

def urgency_score(text: str) -> float:
    tl = text.lower()
    hits = sum(1 for p in URGENCY_PHRASES if p in tl)
    return round(min(hits / len(URGENCY_PHRASES), 1.0), 2)

def sentiment_label(text: str, pipe) -> str:
    result = pipe(text[:512])[0]
    return "Distressed" if result["label"] == "NEGATIVE" else "Calm"

def _load_models():
    import whisper, spacy
    from transformers import pipeline as hf_pipeline
    print("[Audio] Loading Whisper (base) ...")
    w = whisper.load_model("base")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    print("[Audio] Loading sentiment model ...")
    sent = hf_pipeline("text-classification",
                        model="distilbert-base-uncased-finetuned-sst-2-english",
                        truncation=True, max_length=512)
    return w, nlp, sent

# ── Main ────────────────────────────────────────────────────────────────────
def run(audio_dir: str = "audio/data",
        output_csv: str = "audio/output_audio.csv") -> pd.DataFrame:

    supported = (".wav", ".mp3", ".m4a", ".ogg", ".flac")
    files = [f for f in os.listdir(audio_dir) if f.lower().endswith(supported)]

    if not files:
        print("[Audio] No audio files found — using built-in demo data.")
        return _demo(output_csv)

    whisper_model, nlp, sent_pipe = _load_models()
    rows = []
    for i, fname in enumerate(files, 1):
        cid  = f"C{i:03d}"
        path = os.path.join(audio_dir, fname)
        print(f"[Audio] {cid} ← {fname}")
        result     = whisper_model.transcribe(path)
        transcript = result["text"].strip()
        rows.append({
            "Call_ID":         cid,
            "Source_File":     fname,
            "Transcript":      transcript[:250],
            "Extracted_Event": extract_event(transcript),
            "Location":        extract_location(transcript, nlp),
            "Sentiment":       sentiment_label(transcript, sent_pipe),
            "Urgency_Score":   urgency_score(transcript),
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[Audio] ✅ {len(df)} records → {output_csv}\n")
    return df

def _demo(output_csv: str = "audio/output_audio.csv") -> pd.DataFrame:
    df = pd.DataFrame([
        {"Call_ID": "C001", "Source_File": "demo_fire.wav",
         "Transcript": "There is a fire! People are trapped on the second floor of Downtown Avenue! Please hurry!",
         "Extracted_Event": "Fire", "Location": "Downtown Avenue",
         "Sentiment": "Distressed", "Urgency_Score": 0.91},
        {"Call_ID": "C002", "Source_File": "demo_accident.wav",
         "Transcript": "There was a car crash on Main Street, one person is injured and unconscious.",
         "Extracted_Event": "Accident", "Location": "Main Street",
         "Sentiment": "Distressed", "Urgency_Score": 0.75},
        {"Call_ID": "C003", "Source_File": "demo_theft.wav",
         "Transcript": "Someone just broke into my store on Oak Avenue, they stole the register.",
         "Extracted_Event": "Theft", "Location": "Oak Avenue",
         "Sentiment": "Distressed", "Urgency_Score": 0.60},
    ])
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[Audio] ✅ Demo data saved → {output_csv}\n")
    return df

if __name__ == "__main__":
    run()
