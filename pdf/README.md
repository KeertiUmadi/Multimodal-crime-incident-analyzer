# 📄 Document Analyst — Student 2

**Modality: Police Reports / Official Incident Documents (PDF)**

This module extracts structured information from PDF documents such as police reports, insurance forms, and official incident filings — handling both text-based and scanned (OCR) PDFs.

---

## 🎯 Responsibilities

- Extract raw text from PDFs using a PDF parsing library
- Identify and extract: incident type, date, location, officer name, suspect description, and outcome
- Handle scanned PDFs using OCR (pytesseract)
- Output a structured CSV with extracted report fields

---

## 📤 Output Schema

| Report_ID | Department | Doc_Type | Date | Program | Key_Detail |
|-----------|------------|----------|------|---------|------------|
| RPT_001 | Arkansas PD | 1033 Training Proposal | 2015-04-10 | Law Enforcement Support | Equipment request: tactical gear listed |

---

## 🛠️ Tools & Libraries

| Library | Purpose | Install |
|---------|---------|---------|
| `PyMuPDF (fitz)` | Fast PDF text extraction | `pip install pymupdf` |
| `pdfplumber` | Table extraction from PDFs | `pip install pdfplumber` |
| `pytesseract` | OCR for scanned PDF images | `pip install pytesseract` |
| `spaCy` | Named Entity Recognition (NER) for names, locations, dates | `pip install spacy` |

---

## 📦 Dataset

**Arkansas Police Department 1033 Training Plan Proposals** — a real FOIA-released official police PDF from MuckRock.

- **Link:** [muckrock.com — Arkansas Police PDF](https://www.muckrock.com/foi/arkansas-114/arkansas-police-departments-1033-training-plan-proposals-20493/#file-52365)
- **Access:** Open the link → scroll to the file section → download the PDF directly. No account required.

> **Tip:** This is a text-based PDF so `pdfplumber` will extract it cleanly. Focus on extracting department names, dates, program names, and document structure.

---

## 🚀 Running the Module

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python pdf_analyzer.py
```

Output will be saved to `pdf_output.csv`.
