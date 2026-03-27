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
Columns : Report_ID, Department, Doc_Type, Date, Program, Key_Detail
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
    """Extract and standardize date to YYYY-MM-DD format."""
    months = {
        'january': '01', 'february': '02', 'march': '03', 'april': '04',
        'may': '05', 'june': '06', 'july': '07', 'august': '08',
        'september': '09', 'october': '10', 'november': '11', 'december': '12',
        'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 'may': '05',
        'jun': '06', 'jul': '07', 'aug': '08', 'sep': '09', 'oct': '10',
        'nov': '11', 'dec': '12'
    }
    
    # Pattern 1: "Day Month Year" format (e.g., "1 December 2015")
    match = re.search(r'(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})', text)
    if match:
        day = match.group(1).zfill(2)
        month_name = match.group(2).lower()[:3]
        month_num = months.get(month_name, '01')
        year = match.group(3)
        return f"{year}-{month_num}-{day}"
    
    # Pattern 2: "Month Day, Year" format (e.g., "January 19, 2015")
    match = re.search(r'([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})', text)
    if match:
        month_name = match.group(1).lower()[:3]
        month_num = months.get(month_name, '01')
        day = match.group(2).zfill(2)
        year = match.group(3)
        return f"{year}-{month_num}-{day}"
    
    # Pattern 3: Slash format (e.g., "5/5/15" or "01/14/2015")
    match = re.search(r'(\d{1,2})/(\d{1,2})/(\d{2,4})', text)
    if match:
        month = match.group(1).zfill(2)
        day = match.group(2).zfill(2)
        year = match.group(3)
        if len(year) == 2:
            year = "20" + year if int(year) <= 25 else "19" + year
        return f"{year}-{month}-{day}"
    
    # Pattern 4: Just year (e.g., "2015")
    match = re.search(r'\b(\d{4})\b', text)
    if match:
        year = match.group(1)
        return f"{year}-01-01"  # Default to Jan 1 if only year available
    
    # Pattern 5: ISO format (e.g., "2015-04-10")
    match = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', text)
    if match:
        return f"{match.group(1)}-{match.group(2).zfill(2)}-{match.group(3).zfill(2)}"
    
    return "Not found"

def extract_program(text):
    """Extract program name — look for 1033, MRAP, Ref field, or named programs."""
    # Check for Ref: field first (most reliable)
    ref_match = re.search(r'Ref:\s*([^\n]+)', text, re.IGNORECASE)
    if ref_match:
        ref = ref_match.group(1).strip()
        if len(ref) > 0 and ref != "MRAP":  # Ref contains the program name
            return ref[:60]
        elif "MRAP" in ref:
            return "MRAP Training"
    
    # Check for 1033 program specifically
    if "1033" in text:
        return "1033 Program — Law Enforcement Support"
    
    # Check for MRAP
    if "mrap" in text.lower():
        return "MRAP Training"
    
    # Try generic program extraction
    m = re.search(r"(?:program|initiative)[:\s]+([A-Za-z][^\n]{5,60})", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()[:60]
    
    return "General Training"

def extract_key_detail(text):
    """Extract the most meaningful sentence — skip letter headers."""
    # Skip common letter header lines
    skip = ["to:", "from:", "date:", "ref:", "dear", "whom it may"]
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 40]
    
    for line in lines:
        if not any(line.lower().startswith(s) for s in skip):
            detail = line[:200]
            # Clean up garbled OCR text (multiple spaces, weird characters)
            detail = re.sub(r'\s{2,}', ' ', detail)  # Multiple spaces to single
            detail = re.sub(r'([a-z])([A-Z])(?=[a-z])', r'\1 \2', detail)  # Fix merged words like "CountyThe" -> "County The"
            return detail
    
    # Fallback: try to extract first substantive sentence
    text_clean = text.replace('\n', ' ')
    text_clean = re.sub(r' {2,}', ' ', text_clean)
    sentences = re.split(r'[.!?]+', text_clean)
    for sent in sentences:
        sent = sent.strip()
        if 20 < len(sent) < 300 and not any(sent.lower().startswith(h) for h in skip):
            return sent[:200]
    
    return text_clean[:200]

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
    """Extract entities using spaCy NER — persons, organizations, locations."""
    doc     = nlp(text[:8000])
    persons = list(dict.fromkeys(e.text.split("\n")[0].strip() for e in doc.ents if e.label_ == "PERSON"))
    orgs    = list(dict.fromkeys(e.text.split("\n")[0].strip() for e in doc.ents if e.label_ == "ORG"))
    locs    = list(dict.fromkeys(e.text.split("\n")[0].strip() for e in doc.ents if e.label_ in ("GPE","LOC","FAC")))
    return {"persons": persons, "orgs": orgs, "locations": locs}

def clean_org(name):
    """Clean department name — extract first meaningful line."""
    if not name:
        return "Unknown"
    name = name.split("\n")[0].strip()
    name = re.sub(r"\s*(Date|Ref|From|To):.*", "", name, flags=re.IGNORECASE)
    # Fix common OCR issues
    name = re.sub(r'\s{2,}', ' ', name)  # Multiple spaces
    return name.strip()[:80]

def extract_department_from_context(text, ner_orgs):
    """Extract department by searching full text for agency names."""
    # Primary: Look for "From:" field 
    from_match = re.search(r'From:\s+([^\n]+)', text, re.IGNORECASE)
    if from_match:
        dept_candidate = from_match.group(1).strip()
        dept_candidate = re.sub(r'\s{2,}', ' ', dept_candidate)
        
        # If From: contains "Police Department" or "Sheriff's Office", it's a department name
        if any(word in dept_candidate.lower() for word in ["police", "sheriff", "department", "office"]):
            return dept_candidate[:80]
        
        # If From: contains a person's name (and NOT a department), search for department in text
        # Common police titles
        if any(title in dept_candidate.lower() for title in ["lt.", "chief", "captain", "sergeant", "officer"]):
            # Search for nearest department name after the From: field
            dept_patterns = [
                r"(Lonoke\s+County\s+Sheriff['']s\s+Office)",
                r"(Hot\s+Springs\s+Police\s+Department)",
                r"(Jacksonville\s+(?:AR\s+)?Police\s+Department)",
                r"(Fort\s+Smith\s+Police\s+Department)",
                r"([A-Z][a-zA-Z\s]+Police\s+Department)",
                r"([A-Z][a-zA-Z\s]+Sheriff['']s\s+Office)",
            ]
            for pattern in dept_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    dept = match.group(1).strip()
                    dept = re.sub(r'\s{2,}', ' ', dept)
                    return dept[:80]
    
    # Secondary: Look directly for department patterns in text
    dept_patterns = [
        r"(Lonoke\s+County\s+Sheriff['']s\s+Office)",
        r"(Hot\s+Springs\s+Police\s+Department)",
        r"(Jacksonville\s+(?:AR\s+)?Police\s+Department)",
        r"(Fort\s+Smith\s+Police\s+Department)",
        r"([A-Z][a-zA-Z\s]+Police\s+Department)",
        r"([A-Z][a-zA-Z\s]+Sheriff['']s\s+Office)",
    ]
    
    for pattern in dept_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            dept = match.group(1).strip()
            dept = re.sub(r'\s{2,}', ' ', dept)
            return dept[:80]
    
    # Fallback to NER
    if ner_orgs:
        org_name = clean_org(ner_orgs[0])
        if "police" in org_name.lower() or "sheriff" in org_name.lower():
            return org_name
    
    return "Unknown"

def extract_officer_name(text, ner_persons):
    """Extract officer name — prioritize names in signature blocks or from NER."""
    # Look for officer signatures
    officer_patterns = [
        r"(?:Officer|Officer Name|Submitted by|Prepared by|Signed by|By:)\s+([A-Z][a-zA-Z\s]+)",
        r"(?:Chief|Director|Captain|Lieutenant|Sergeant)\s+([A-Z][a-zA-Z\s]+)",
    ]
    for pattern in officer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()[:60]
    
    # Fallback to first person found in NER
    if ner_persons:
        return ner_persons[0][:60]
    return "Not mentioned"

def extract_location_detailed(text, ner_locations):
    """Extract location — from addresses or NER results."""
    # Look for address patterns
    address_match = re.search(
        r"(?:Address|Located at|Location|Agency|Department)[:\s]+([^\n]+)", 
        text, 
        re.IGNORECASE
    )
    if address_match:
        return address_match.group(1).strip()[:100]
    
    # Use NER locations
    if ner_locations:
        return ", ".join(ner_locations[:2])[:100]
    
    return "Not mentioned"

def extract_summary(text):
    """Extract a meaningful summary — skip headers, try first substantive paragraph."""
    skip_starts = ["to:", "from:", "date:", "ref:", "dear", "whom it may", "re:"]
    
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 30]
    
    for line in lines:
        if not any(line.lower().startswith(s) for s in skip_starts):
            # Found first substantive line, combine with next line if short
            summary = line
            return summary[:300]
    
    # Fallback: combine first sentences
    sentences = re.split(r'[.!?]+', text)
    valid_sentences = [s.strip() for s in sentences if 20 < len(s.strip()) < 200]
    if valid_sentences:
        return " ".join(valid_sentences[:2])[:300]
    
    return text.replace("\n", " ")[:300]

def split_pdf_into_sections(text):
    """Split multi-document PDFs by finding  department/document boundaries."""
    #  Simple approach: split by common document start markers
    lines = text.split('\n')
    sections = []
    current_section = []
    
    for i, line in enumerate(lines):
        # Check if this line starts a new document (From: or To: Whom)
        if re.match(r'^\s*(From|To):\s', line, re.IGNORECASE) and current_section and len('\n'.join(current_section)) > 400:
            # Save previous section and start new one
            sections.append('\n'.join(current_section).strip())
            current_section = [line]
        else:
            current_section.append(line)
    
    # Add last section
    if current_section and len('\n'.join(current_section)) > 400:
        sections.append('\n'.join(current_section).strip())
    
    #  Filter empty/tiny sections
    sections = [s for s in sections if len(s) > 500]
    
    # If only 1 section, try splitting by COURSE: marker  
    if len(sections) < 2:
        sections = []
        current = []
        for line in lines:
            if re.match(r'^\s*COURSE:\s+LESSON', line, re.IGNORECASE) and current and len('\n'.join(current)) > 400:
                sections.append('\n'.join(current).strip())
                current = [line]
            else:
                current.append(line)
        if current:
            sections.append('\n'.join(current).strip())
        sections = [s for s in sections if len(s) > 400]
    
    return sections if sections else [text]

# ── Main ─────────────────────────────────────────────────────────────────────
def _sanitize_text(text):
    """Clean text for CSV: remove newlines, fix encoding, collapse spaces."""
    if not text:
        return ""
    # Handle common encoding issues - check for literal UTF-8 sequences  
    text = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
    # Replace smart quotes and dashes
    text = text.replace('\u2019', "'")   # right single quote
    text = text.replace('\u201c', '"')   # left double quote
    text = text.replace('\u201d', '"')   # right double quote
    text = text.replace('\u2013', '-')   # en dash
    text = text.replace('\u2014', '-')   # em dash
    # Remove newlines and collapse multiple spaces
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def run(pdf_dir=None, output_csv=None):
    """Process PDFs and extract structured data."""
    # Default to script directory if not specified
    if pdf_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pdf_dir = os.path.join(script_dir, "data")
    if output_csv is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_csv = os.path.join(script_dir, "output_pdf.csv")
    
    files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not files:
        print("[PDF] No PDF files found")
        return

    nlp  = _load_nlp()
    rows = []
    report_id_counter = 1
    
    for fname in files:
        path = os.path.join(pdf_dir, fname)
        print(f"[PDF] Processing: {fname}")
        text = get_text(path)
        
        # For single-document PDFs or simple documents, process directly
        # For multi-document PDFs (like Arkansas 1033), try to split
        sections = split_pdf_into_sections(text)
        if len(sections) > 1:
            print(f"  --> Found {len(sections)} sections")
            process_texts = sections
        else:
            process_texts = [text]
        
        # Process each section/document
        for section_text in process_texts:
            if len(section_text.strip()) < 100:
                continue
                
            rid  = f"RPT_{report_id_counter:03d}"
            ner  = ner_fields(section_text, nlp)
            
            # Extract ONLY required fields per assignment
            rows.append({
                "Report_ID":        rid,
                "Department":       _sanitize_text(extract_department_from_context(section_text, ner["orgs"])),
                "Doc_Type":         _sanitize_text(classify_doc_type(section_text)),
                "Date":             _sanitize_text(extract_date(section_text)),
                "Program":          _sanitize_text(extract_program(section_text)),
                "Key_Detail":       _sanitize_text(extract_key_detail(section_text)),
            })
            
            report_id_counter += 1

    if not rows:
        print("[PDF] ⚠️  No data extracted")
        return None
    
    # Create DataFrame with ONLY required columns
    df = pd.DataFrame(rows)
    
    # Define output columns (assignment specification)
    output_columns = [
        "Report_ID", "Department", "Doc_Type", "Date", "Program", "Key_Detail"
    ]
    df = df[output_columns]
    
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    
    print(f"\n[PDF] ✅ {len(df)} records extracted -> {output_csv}\n")
    print("── Output Table ──")
    print(df.to_string(index=False))
    
    return df

if __name__ == "__main__":
    run()  # Uses default paths relative to script location
