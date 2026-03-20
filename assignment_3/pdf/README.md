# Student 2 — Document Analyst 📄

## Task
Extract structured info from police/incident PDFs — handles both text-based and scanned documents.

## Tools
- `pdfplumber` — text-based PDF extraction
- `PyMuPDF + pytesseract` — OCR fallback for scanned PDFs
- `spaCy` — NER for persons, organisations, locations, dates

## Dataset
**Arkansas PD 1033 Training Plan Proposals** — [MuckRock FOIA Link](https://www.muckrock.com/foi/arkansas-114/arkansas-police-departments-1033-training-plan-proposals-20493/#file-52365)
1. Open link → scroll to file section → download PDF
2. No account required
3. Place in `pdf/data/`

## Run
```bash
pip install -r pdf/requirements.txt
python pdf/pdf_analyzer.py
```
> No files in `pdf/data/`? Demo data runs automatically.

## Output: `pdf/output_pdf.csv`
| Report_ID | Department | Incident_Type | Doc_Type | Date | Location | Officer |
|-----------|------------|---------------|----------|------|----------|---------|
| RPT_001 | Arkansas PD | Administrative | Training Proposal | 2015-04-10 | Little Rock | Officer Johnson |
