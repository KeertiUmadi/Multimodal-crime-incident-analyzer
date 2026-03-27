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
    if not os.path.exists(audio_dir):
        print("[Audio] No audio data directory found — using built-in demo data.")
        return _demo(output_csv)
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
    df.to_csv(output_csv,import os
import pandas as pd

# -----------------------------
# EVENT EXTRACTION (KEYWORD-BASED)
# -----------------------------
def extract_event(text: str) -> str:
    tl = text.lower()

    if any(k in tl for k in ["shoot", "gunshot", "firing"]):
        return "Shooting"
    elif "drug" in tl:
        return "Drug Activity"
    elif any(k in tl for k in ["escort", "problem", "disturb"]):
        return "Disturbance"
    elif any(k in tl for k in ["alone", "someone here"]):
        return "Suspicious Situation"
    elif any(k in tl for k in ["fire", "smoke"]):
        return "Fire"

    return "Unknown"

# -----------------------------
# LOCATION EXTRACTION (NO GENERIC)
# -----------------------------
def extract_location(text: str, nlp) -> str:
    doc = nlp(text)
    locs = [e.text for e in doc.ents if e.label_ in ("GPE", "LOC", "FAC", "ORG")]

    if locs:
        return ", ".join(dict.fromkeys(locs))

    return "Not mentioned"

# -----------------------------
# URGENCY SCORING
# -----------------------------
def urgency_score(text: str, event: str) -> float:

    if event == "Shooting":
        return 0.95
    elif event == "Fire":
        return 0.9
    elif event == "Drug Activity":
        return 0.7
    elif event == "Disturbance":
        return 0.7
    elif event == "Suspicious Situation":
        return 0.8

    return 0.6
# -----------------------------
# SENTIMENT ANALYSIS
# -----------------------------
def sentiment_label(text: str, pipe) -> str:
    result = pipe(text[:512])[0]
    return "Distressed" if result["label"] == "NEGATIVE" else "Calm"

# -----------------------------
# LOAD MODELS
# -----------------------------
def load_models():
    import whisper, spacy
    from transformers import pipeline

    print("[Audio] Loading Whisper...")
    whisper_model = whisper.load_model("base")

    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    print("[Audio] Loading Sentiment Model...")
    sentiment_pipe = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    return whisper_model, nlp, sentiment_pipe

# -----------------------------
# MAIN FUNCTION
# -----------------------------
def run(
    audio_dir=r"C:\Users\umadi\Multimodal-crime-incident-analyzer\audio\data",
    output_csv=r"C:\Users\umadi\Multimodal-crime-incident-analyzer\audio\output_audio.csv"
):

    supported = (".wav", ".mp3", ".m4a", ".ogg", ".flac")

    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"Directory not found: {audio_dir}")

    files = [f for f in os.listdir(audio_dir) if f.lower().endswith(supported)]

    if not files:
        raise ValueError("No audio files found.")

    whisper_model, nlp, sentiment_pipe = load_models()

    data = []

    for i, file in enumerate(files, 1):
        path = os.path.join(audio_dir, file)
        call_id = f"C{i:03d}"

        print(f"[Audio] Processing {file}")

        result = whisper_model.transcribe(path)
        transcript = result["text"].strip()

        event = extract_event(transcript)
        location = extract_location(transcript, nlp)
        sentiment = sentiment_label(transcript, sentiment_pipe)
        urgency = urgency_score(transcript, event)

        data.append({
            "Call_ID": call_id,
            "Source_File": file,
            "Transcript": transcript,
            "Extracted_Event": event,
            "Location": location,
            "Sentiment": sentiment,
            "Urgency_Score": urgency
        })

    df = pd.DataFrame(data)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"\n✅ Output saved to: {output_csv}")
    print(df)

# -----------------------------
# RUN SCRIPT
# -----------------------------
if __name__ == "__main__":
    run()
