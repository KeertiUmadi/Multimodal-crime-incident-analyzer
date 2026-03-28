"""
Student 5 — Text Analyst
=========================
Processes social media posts / news articles.
Runs spaCy NER, HuggingFace sentiment, BART zero-shot crime classification.

Dataset : CrimeReport (Kaggle)
Link    : https://www.kaggle.com/datasets/cameliasiadat/crimereport
Place   : text/data/crimereport.csv   (column: 'text')

Fallback: sample_data/sample_text.csv (built-in, no download needed)

Output  : text/output_text.csv
Columns : Text_ID, Source, Raw_Text, Cleaned_Text, Sentiment,
          Entities, Location_Entity, Crime_Type, Topic, Severity_Label
"""

import os, re
import pandas as pd
import nltk

nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

from nltk.corpus   import stopwords
from nltk.tokenize import word_tokenize

STOP_WORDS = set(stopwords.words("english"))

CRIME_LABELS = [
    "fire", "robbery", "theft", "assault", "murder",
    "accident", "drug offense", "vandalism", "disturbance",
]

SEVERITY_MAP = {
    "murder": "High", "assault": "High", "fire": "High", "robbery": "High",
    "theft": "Medium", "drug offense": "Medium", "accident": "Medium",
    "disturbance": "Low", "vandalism": "Low",
}

TOPIC_MAP = {
    "robbery":      "Theft / Robbery",
    "theft":        "Theft / Robbery",
    "fire":         "Fire / Emergency",
    "assault":      "Violence / Assault",
    "murder":       "Violence / Murder",
    "accident":     "Accident",
    "drug offense": "Drug Offense",
    "disturbance":  "Public Disturbance",
    "vandalism":    "Vandalism",
}

def _load_models():
    import spacy
    from transformers import pipeline as hf_pipeline
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        os.system("python -m spacy download en_core_web_sm")
        import spacy
        nlp = spacy.load("en_core_web_sm")
    print("[Text] Loading sentiment model ...")
    sent_pipe = hf_pipeline("text-classification",
                             model="distilbert-base-uncased-finetuned-sst-2-english",
                             truncation=True, max_length=512)
    print("[Text] Loading zero-shot classifier ...")
    zs_pipe = hf_pipeline("zero-shot-classification",
                           model="facebook/bart-large-mnli")
    return nlp, sent_pipe, zs_pipe

def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = word_tokenize(text)
    return " ".join(t for t in tokens if t not in STOP_WORDS and len(t) > 2)

def ner(text: str, nlp) -> dict:
    doc  = nlp(text[:5000])
    locs = list(dict.fromkeys(e.text for e in doc.ents if e.label_ in ("GPE", "LOC", "FAC")))
    pers = list(dict.fromkeys(e.text for e in doc.ents if e.label_ == "PERSON"))
    orgs = list(dict.fromkeys(e.text for e in doc.ents if e.label_ == "ORG"))
    return {"locations": locs[:3], "persons": pers[:3], "orgs": orgs[:3]}

def sentiment(text: str, pipe) -> str:
    r = pipe(text[:512])[0]
    return "Negative" if r["label"] == "NEGATIVE" else "Positive"

def crime_type(text: str, pipe) -> str:
    r = pipe(text[:512], candidate_labels=CRIME_LABELS)
    return r["labels"][0].title()

def run(text_dir: str  = "text/data",
        output_csv: str = "text/output_text.csv") -> pd.DataFrame:

    kaggle_csv = os.path.join(text_dir, "crimereport.csv")
    sample_csv = "sample_data/sample_text.csv"

    if os.path.exists(kaggle_csv):
        df_raw   = pd.read_csv(kaggle_csv, nrows=50)
        text_col = "text" if "text" in df_raw.columns else df_raw.columns[0]
        src_col  = "source" if "source" in df_raw.columns else None
        print(f"[Text] Loaded Kaggle CrimeReport ({len(df_raw)} rows)")
    elif os.path.exists(sample_csv):
        df_raw   = pd.read_csv(sample_csv)
        text_col, src_col = "text", "source"
        print(f"[Text] Using sample_data/sample_text.csv ({len(df_raw)} rows)")
    else:
        """print("[Text] No data found — using built-in demo data.")
        return _demo(output_csv)"""
        raise FileNotFoundError("[Text] No dataset found. Please add crimereport.csv to text/data/")

    nlp, sent_pipe, zs_pipe = _load_models()
    rows = []
    for i, row in df_raw.iterrows():
        raw = str(row.get(text_col, "")).strip()
        if not raw:
            continue
        tid = f"TXT_{i+1:03d}"
        src = str(row.get(src_col, "Unknown")) if src_col else "Unknown"
        print(f"[Text] {tid} ...")
        entities = ner(raw, nlp)
        ctype    = crime_type(raw, zs_pipe)
        rows.append({
            "Text_ID":         tid,
            "Crime_Type":      ctype,
            "Location_Entity": ", ".join(entities["locations"]) or "Unknown",
            "Sentiment":       sentiment(raw, sent_pipe),
            "Topic":           TOPIC_MAP.get(ctype.lower(), ctype),
            "Severity_Label":  SEVERITY_MAP.get(ctype.lower(), "Low"),
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[Text] ✅ {len(df)} records → {output_csv}\n")
    return df


if __name__ == "__main__":
    run()
