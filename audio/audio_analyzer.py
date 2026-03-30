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
Columns : Call_ID, Transcript, Extracted_Event, Location, Sentiment, Urgency_Score
"""
import os
import re
import pandas as pd

# -----------------------------
# EVENT EXTRACTION (KEYWORD-BASED)
# -----------------------------
def extract_event(text: str) -> str:
    tl = text.lower()

    if any(k in tl for k in ["shoot", "gunshot", "firing", "shot", "shots"]):
        return "Shooting"
    elif any(k in tl for k in ["rob", "robbed", "robbery", "stolen", "burglar", "burglary"]):
        return "Robbery"
    elif any(k in tl for k in ["accident", "crash", "collision", "crashed"]):
        return "Accident"
    elif any(k in tl for k in ["strangle", "strangled", "assault", "attack", "beat", "stab"]):
        return "Assault"
    elif any(k in tl for k in ["dead", "body", "homicide", "murder", "killed"]):
        return "Homicide"
    elif any(k in tl for k in ["break in", "breaking in", "broke in", "kicked in", "kicking in"]):
        return "Break-in"
    elif any(k in tl for k in ["drug", "drugs", "narco"]):
        return "Drug Activity"
    elif any(k in tl for k in ["fire", "smoke", "burning", "flames"]):
        return "Fire"
    elif any(k in tl for k in ["escort", "problem", "disturb", "naked", "fight"]):
        return "Disturbance"
    elif any(k in tl for k in ["alone", "someone here", "break into", "hiding"]):
        return "Suspicious Situation"
    elif any(k in tl for k in ["suicid", "kill myself", "hurt myself", "overdose", "ambulance", "medical", "unconscious", "not breathing", "heart", "bleeding"]):
        return "Medical Emergency"
    
    return "Unknown"

# -----------------------------
# LOCATION EXTRACTION
# -----------------------------
def extract_location(text: str, nlp) -> str:
    doc = nlp(text)
    locs = [e.text for e in doc.ents if e.label_ in ("GPE", "LOC", "FAC")]
    if locs:
        return ", ".join(dict.fromkeys(locs))
    
    return "Not mentioned"

# -----------------------------
# URGENCY SCORING
# -----------------------------
def urgency_score(text: str, event: str) -> float:
    if event == "Shooting":
        return 0.95
    elif event in ("Homicide", "Assault"):
        return 0.92
    elif event == "Fire":
        return 0.9
    elif event == "Robbery":
        return 0.85
    elif event == "Break-in":
        return 0.80
    elif event == "Suspicious Situation":
        return 0.8
    elif event in ("Drug Activity", "Disturbance"):
        return 0.7
    elif event == "Accident":
        return 0.75
    elif event == "Medical Emergency":
        return 0.93
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
    audio_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
    output_csv=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_audio.csv")
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
            "Transcript": transcript,
            "Extracted_Event": event,
            "Location": location,
            "Sentiment": sentiment,
            "Urgency_Score": urgency
        })

    output_columns = [
        "Call_ID",
        "Transcript",
        "Extracted_Event",
        "Location",
        "Sentiment",
        "Urgency_Score",
    ]
    df = pd.DataFrame(data)[output_columns]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"\n✅ Output saved to: {output_csv}")
    print(df)

# -----------------------------
# RUN SCRIPT
# -----------------------------
if __name__ == "__main__":
    run()