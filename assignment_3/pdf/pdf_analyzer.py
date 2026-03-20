"""
Student 2 — Document Analyst
=============================
Extracts structured info from police/incident PDFs.
Handles text-based PDFs (pdfplumber) and scanned PDFs (pytesseract OCR).
Uses spaCy NER for persons, orgs, locations, dates.

Dataset : Arkansas PD 1033 Training Plan Proposals (MuckRock FOIA)
Link    : https://www.muckrock.com/foi/arkansas-114/arkansas-police-departments-1033-training-plan-proposals-20493/
Place   : pdf/data/*.pdf

Output  : pdf/output_pdf.csv
Columns : Report_ID, Source_File, Department, Incident_Type, Doc_Type,
          Date, Location, Officer, Program, Key_Detail, Summary
"""

import os, re, io
import pandas as pd
import pdfplumber
import fitz          # PyMuPDF
import pytesseract
from PIL import Image

def _load_nlp():
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        os.system("python -m spacy download en_core_web_sm")
        import spacy
        return spacy.load("en_core_web_sm")

# ── Text extraction ──────────────────────────────────────────────────────────
def _plumber(path: str) -> str:
    parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
    return "\n".join(parts)

def _ocr(path: str) -> str:
    parts = []
    doc = fitz.open(path)
    for page in doc:
        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        parts.append(pytesseract.image_to_string(img))
    return "\n".join(parts)

def get_text(path: str) -> str:
    text = _plumber(path)
    if len(text.strip()) < 50:
        print(f"  [PDF] Sparse text — switching to OCR: {os.path.basename(path)}")
        text = _ocr(path)
    return text

# ── Field extraction ─────────────────────────────────────────────────────────
def classify_incident_type(text: str) -> str:
    tl = text.lower()
    if any(w in tl for w in ["fire", "arson", "blaze", "smoke"]):         return "Fire"
    if any(w in tl for w in ["robbery", "theft", "stolen", "burglar"]):   return "Theft/Robbery"
    if any(w in tl for w in ["assault", "attack", "fight", "weapon"]):    return "Assault"
    if any(w in tl for w in ["accident", "crash", "collision"]):           return "Accident"
    if any(w in tl for w in ["drug", "narcotic", "substance"]):            return "Drug Offense"
    if any(w in tl for w in ["homicide", "murder", "shooting"]):           return "Homicide"
    if any(w in tl for w in ["training", "proposal", "equipment"]):        return "Administrative"
    return "General Incident"

def classify_doc_type(text: str) -> str:
    tl = text.lower()
    if any(w in tl for w in ["training", "plan", "proposal"]): return "Training Proposal"
    if any(w in tl for w in ["arrest", "suspect", "offense"]): return "Arrest Report"
    if any(w in tl for w in ["incident", "report", "complaint"]): return "Incident Report"
    if any(w in tl for w in ["insurance", "claim", "damage"]): return "Insurance Form"
    return "General Document"

def extract_date(text: str) -> str:
    for pat in [r"\b\d{4}-\d{2}-\d{2}\b",
                r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
                r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s+\d{4}\b"]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(0)
    return "Not found"

def extract_program(text: str) -> str:
    m = re.search(r"(?:program|project|initiative)[:\s]+([A-Z][^\n]{3,50})", text, re.IGNORECASE)
    return m.group(1).strip() if m else "N/A"

def ner_fields(text: str, nlp) -> dict:
    doc = nlp(text[:8000])
    persons = list(dict.fromkeys(e.text for e in doc.ents if e.label_ == "PERSON"))
    orgs    = list(dict.fromkeys(e.text for e in doc.ents if e.label_ == "ORG"))
    locs    = list(dict.fromkeys(e.text for e in doc.ents if e.label_ in ("GPE", "LOC", "FAC")))
    return {"persons": persons, "orgs": orgs, "locations": locs}

# ── Main ─────────────────────────────────────────────────────────────────────
def run(pdf_dir: str = "pdf/data",
        output_csv: str = "pdf/output_pdf.csv") -> pd.DataFrame:

    files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not files:
        print("[PDF] No PDF files found — using built-in demo data.")
        return _demo(output_csv)

    nlp  = _load_nlp()
    rows = []
    for i, fname in enumerate(files, 1):
        rid  = f"RPT_{i:03d}"
        path = os.path.join(pdf_dir, fname)
        print(f"[PDF] {rid} ← {fname}")
        text = get_text(path)
        ner  = ner_fields(text, nlp)
        rows.append({
            "Report_ID":      rid,
            "Source_File":    fname,
            "Department":     ner["orgs"][0]    if ner["orgs"]    else "Unknown",
            "Incident_Type":  classify_incident_type(text),
            "Doc_Type":       classify_doc_type(text),
            "Date":           extract_date(text),
            "Location":       ", ".join(ner["locations"][:3]),
            "Officer":        ner["persons"][0] if ner["persons"] else "Not mentioned",
            "Program":        extract_program(text),
            "Key_Detail":     text.replace("\n", " ")[:200],
            "Summary":        text.replace("\n", " ")[:400],
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[PDF] ✅ {len(df)} records → {output_csv}\n")
    return df

def _demo(output_csv: str = "pdf/output_pdf.csv") -> pd.DataFrame:
    df = pd.DataFrame([
        {"Report_ID": "RPT_001", "Source_File": "arkansas_pd_1033.pdf",
         "Department": "Arkansas PD", "Incident_Type": "Administrative",
         "Doc_Type": "Training Proposal", "Date": "2015-04-10",
         "Location": "Little Rock, AR", "Officer": "Officer Johnson",
         "Program": "Law Enforcement Support",
         "Key_Detail": "Equipment request: tactical gear listed for law enforcement support",
         "Summary": "Arkansas PD 1033 training plan proposal. Requesting tactical equipment for officer safety."},
        {"Report_ID": "RPT_002", "Source_File": "incident_report_002.pdf",
         "Department": "City Police Dept", "Incident_Type": "Theft/Robbery",
         "Doc_Type": "Incident Report", "Date": "2024-03-15",
         "Location": "Downtown, Chicago", "Officer": "Officer Martinez",
         "Program": "N/A",
         "Key_Detail": "Robbery at convenience store on 5th Avenue at 22:30",
         "Summary": "Suspect threatened cashier and stole register contents. Fled on foot northbound."},
        {"Report_ID": "RPT_003", "Source_File": "fire_report_003.pdf",
         "Department": "Fire Department", "Incident_Type": "Fire",
         "Doc_Type": "Incident Report", "Date": "2024-03-16",
         "Location": "Warehouse District, Oak Street", "Officer": "Captain Williams",
         "Program": "N/A",
         "Key_Detail": "Warehouse fire reported at 14:30. Possible arson. Two units dispatched.",
         "Summary": "Large fire at industrial warehouse. Fire department responded in 8 minutes. Arson suspected."},
    ])
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[PDF] ✅ Demo data saved → {output_csv}\n")
    return df

if __name__ == "__main__":
    run()
