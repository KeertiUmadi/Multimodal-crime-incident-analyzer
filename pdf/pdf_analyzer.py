"""
Student 2 — Document Analyst
=============================
Extracts structured info from police/incident PDFs.
Handles text-based PDFs (pdfplumber) and scanned PDFs (pytesseract OCR).
Uses spaCy NER for persons, orgs, locations, dates.

Dataset : Arkansas Police Department 1033 Training Plan Proposals (MuckRock FOIA)
Link    : https://www.muckrock.com/foi/arkansas-114/arkansas-police-departments-1033-training-plan-proposals-20493/
Place   : pdf/data/*.pdf

Output  : pdf/output_pdf.csv
Columns : Report_ID, Department, Doc_Type, Date, Program, Key_Detail,
          Incident_Type, Location, Officer, Summary
"""

import os, re, io
import pandas as pd
import pdfplumber
import fitz
import pytesseract
from PIL import Image

def _load_nlp():
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        os.system("python -m spacy download en_core_web_sm")
        import spacy as _s
        return _s.load("en_core_web_sm")

# ── Text extraction ──────────────────────────────────────────────────────────
def _plumber(path):
    parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
    return "\n".join(parts)

def _ocr(path):
    parts = []
    doc = fitz.open(path)
    for page in doc:
        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        parts.append(pytesseract.image_to_string(img))
    return "\n".join(parts)

def get_text(path):
    text = _plumber(path)
    if len(text.strip()) < 50:
        print(f"  [PDF] Sparse text — switching to OCR: {os.path.basename(path)}")
        text = _ocr(path)
    return text

# ── Field extraction ─────────────────────────────────────────────────────────
def classify_incident_type(text):
    tl = text.lower()
    if any(w in tl for w in ["training", "proposal", "1033", "mrap", "equipment"]):
        return "Administrative"
    if any(w in tl for w in ["homicide", "murder", "shooting"]):
        return "Homicide"
    if any(w in tl for w in ["robbery", "theft", "stolen", "burglar"]):
        return "Theft/Robbery"
    if any(w in tl for w in ["assault", "attack", "fight"]):
        return "Assault"
    if any(w in tl for w in ["drug", "narcotic"]):
        return "Drug Offense"
    if any(w in tl for w in ["accident", "crash", "collision"]):
        return "Accident"
    if any(w in tl for w in ["arson", "blaze"]):
        return "Fire"
    return "General Incident"

def classify_doc_type(text):
    tl = text.lower()
    if any(w in tl for w in ["training", "plan", "proposal"]): return "Training Proposal"
    if any(w in tl for w in ["arrest", "suspect", "offense"]):  return "Arrest Report"
    if any(w in tl for w in ["incident", "report", "complaint"]): return "Incident Report"
    return "General Document"

def extract_date(text):
    # Try full date formats
    for pat in [
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s+\d{4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(0)
    return "Not found"

def extract_program(text):
    """Extract program name — look for 1033, MRAP, or named programs."""
    # Check for 1033 program specifically
    if "1033" in text:
        return "1033 Program — Law Enforcement Support"
    m = re.search(r"(?:program|initiative)[:\s]+([A-Za-z][^\n]{5,60})", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()[:80]
    return "N/A"

def extract_key_detail(text):
    """Extract the most meaningful sentence — skip letter headers."""
    # Skip common letter header lines
    skip = ["to:", "from:", "date:", "ref:", "dear", "whom it may"]
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 40]
    for line in lines:
        if not any(line.lower().startswith(s) for s in skip):
            return line[:200]
    return text.replace("\n", " ")[:200]

def clean_location(locs):
    """Filter out noise from location list — remove person names, short strings."""
    clean = []
    noise = ["rostan", "united states government", "government"]
    for loc in locs:
        loc_clean = loc.split("\n")[0].strip()
        if len(loc_clean) > 3 and loc_clean.lower() not in noise:
            clean.append(loc_clean)
    return ", ".join(clean[:3]) if clean else "Not mentioned"

def ner_fields(text, nlp):
    doc     = nlp(text[:8000])
    persons = list(dict.fromkeys(e.text.split("\n")[0].strip() for e in doc.ents if e.label_ == "PERSON"))
    orgs    = list(dict.fromkeys(e.text.split("\n")[0].strip() for e in doc.ents if e.label_ == "ORG"))
    locs    = list(dict.fromkeys(e.text.split("\n")[0].strip() for e in doc.ents if e.label_ in ("GPE","LOC","FAC")))
    return {"persons": persons, "orgs": orgs, "locations": locs}

def clean_org(name):
    """Clean department name — first line only."""
    if not name:
        return "Unknown"
    name = name.split("\n")[0].strip()
    name = re.sub(r"\s*(Date|Ref|From|To):.*", "", name, flags=re.IGNORECASE)
    return name.strip()[:60]

# ── Main ─────────────────────────────────────────────────────────────────────
def run(pdf_dir="pdf/data", output_csv="pdf/output_pdf.csv"):
    files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not files:
        print("[PDF] No PDF files found — using built-in demo data.")
        return _demo(output_csv)

    nlp  = _load_nlp()
    rows = []
    for i, fname in enumerate(files, 1):
        rid  = f"RPT_{i:03d}"
        path = os.path.join(pdf_dir, fname)
        print(f"[PDF] {rid} <- {fname}")
        text = get_text(path)
        ner  = ner_fields(text, nlp)

        rows.append({
            "Report_ID":     rid,
            "Department":    clean_org(ner["orgs"][0]) if ner["orgs"] else "Unknown",
            "Doc_Type":      classify_doc_type(text),
            "Date":          extract_date(text),
            "Program":       extract_program(text),
            "Key_Detail":    extract_key_detail(text),
            "Incident_Type": classify_incident_type(text),
            "Location":      clean_location(ner["locations"]),
            "Officer":       ner["persons"][0] if ner["persons"] else "Not mentioned",
            "Summary":       text.replace("\n", " ")[:400],
            "Source_File":   fname,
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n[PDF] ✅ {len(df)} records -> {output_csv}\n")
    print("── Output (Assignment Schema) ──")
    print(df[["Report_ID","Department","Doc_Type","Date","Program","Key_Detail"]].to_string(index=False))
    return df

def _demo(output_csv="pdf/output_pdf.csv"):
    df = pd.DataFrame([
        {
            "Report_ID":     "RPT_001",
            "Department":    "Fort Smith Police Department",
            "Doc_Type":      "Training Proposal",
            "Date":          "January 19, 2015",
            "Program":       "1033 Program — Law Enforcement Support",
            "Key_Detail":    "Equipment request: MRAP vehicle allocated by US Government for law enforcement use",
            "Incident_Type": "Administrative",
            "Location":      "Fort Smith, AR",
            "Officer":       "Not mentioned",
            "Summary":       "Fort Smith PD documentation for MRAP vehicle use and training under 1033 Program. Vehicle allocated by the United States Government.",
            "Source_File":   "arkansas_pd_1033.pdf",
        },
    ])
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[PDF] ✅ Demo data saved -> {output_csv}\n")
    return df

if __name__ == "__main__":
    run()
