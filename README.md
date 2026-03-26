# 🚨 Multimodal Crime / Incident Report Analyzer

An AI-powered pipeline that ingests unstructured data from **5 modalities** and produces a unified structured incident report — helping emergency response teams respond faster and more accurately to incidents.

---

## 🧠 Background

A city's emergency response department faces hundreds of daily incident reports — road accidents, thefts, fires, and public disturbances — arriving from audio calls, written police reports, CCTV footage, scene photos, and social media. This system automates that analysis using AI, converting raw unstructured data into a clean, queryable incident dataset.

---

## 📁 Repository Structure

```
multimodal-crime-incident-analyzer/
├── audio/                  # Student 1 — Audio Analyst
│   ├── audio_analyzer.py
│   ├── requirements.txt
│   └── README.md
├── pdf/                    # Student 2 — Document Analyst
│   ├── pdf_analyzer.py
│   ├── requirements.txt
│   └── README.md
├── images/                 # Student 3 — Image Analyst
│   ├── image_analyzer.py
│   ├── requirements.txt
│   └── README.md
├── video/                  # Student 4 — Video Analyst
│   ├── video_analyzer.py
│   ├── requirements.txt
│   └── README.md
├── text/                   # Student 5 — Text Analyst
│   ├── text_analyzer.py
│   ├── requirements.txt
│   └── README.md
├── integration/            # Full Team — Merge + Dashboard
│   ├── integrate.py
│   ├── dashboard.py
│   ├── requirements.txt
│   └── README.md
├── sample_data/            # Built-in demo data (no downloads needed)
│   └── sample_text.csv
├── docs/                   # Deliverables 1 & 4
│   ├── pipeline_architecture.html
│   └── project_report.md
├── demo.py                 # Deliverable 5 — Demonstration script
├── run_pipeline.py         # One-command full pipeline runner
├── requirements.txt        # All dependencies combined
├── .gitignore
└── README.md
```

---

## 🔄 AI Pipeline

| # | Stage | Description |
|---|-------|-------------|
| 1 | **Unstructured Data Ingestion** | Audio files, PDFs, images, video clips, and text posts are loaded |
| 2 | **AI Processing per Modality** | Each module runs its AI model on its specific data type |
| 3 | **Information Extraction** | Key fields (event, location, time, entities, sentiment) are extracted |
| 4 | **Structured Dataset Generation** | All outputs are merged into a unified CSV |
| 5 | **Dashboard / Query System** | Final data is visualized to generate an incident summary |

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/KeertiUmadi/multimodal-crime-incident-analyzer.git
cd multimodal-crime-incident-analyzer

# 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Run full pipeline (works with built-in demo data — no downloads needed)
python run_pipeline.py

# 4. Launch dashboard only
streamlit run integration/dashboard.py

# 5. Run demonstration (Deliverable 5)
python demo.py
```

---

## 👥 Team Roles

| # | Role | Modality | Key Tools | Dataset |
|---|------|----------|-----------|---------|
| 1 | Audio Analyst | Emergency calls (.wav/.mp3) | Whisper, spaCy, HuggingFace | [911 Calls – Kaggle](https://www.kaggle.com/code/stpeteishii/911-calls-wav2vec2) |
| 2 | Document Analyst | Police PDFs (.pdf) | pdfplumber, PyMuPDF, pytesseract | [Arkansas PD – MuckRock](https://www.muckrock.com/foi/arkansas-114/arkansas-police-departments-1033-training-plan-proposals-20493/#file-52365) |
| 3 | Image Analyst | Scene photos (.jpg/.png) | YOLOv8, OpenCV, pytesseract | [Fire Detection – Roboflow](https://universe.roboflow.com/search?q=fire) |
| 4 | Video Analyst | CCTV footage (.mp4/.mpg) | OpenCV, YOLOv8, moviepy | [CAVIAR CCTV – Edinburgh](https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/) |
| 5 | Text Analyst | Social media / news (.csv) | spaCy, HuggingFace BART, NLTK | [CrimeReport – Kaggle](https://www.kaggle.com/datasets/cameliasiadat/crimereport) |

---

## 📊 Final Output Schema

| Incident_ID | Audio_Event | PDF_Doc_Type | Image_Objects | Video_Event | Text_Crime_Type | Severity |
|-------------|-------------|--------------|---------------|-------------|-----------------|----------|
| INC_001 | Building fire / trapped | 1033 Training Proposal | fire, smoke (0.94) | Person collapsing | Robbery / Theft | High |

---

## 📋 Deliverables

| # | Deliverable | File |
|---|-------------|------|
| 1 | AI Pipeline Architecture Diagram | `docs/pipeline_architecture.html` |
| 2 | Code Repository (GitHub) | This repo — all 6 folders with README + requirements |
| 3 | Structured Dataset | `integration/integrated_incidents.csv` (generated on run) |
| 4 | Project Report | `docs/project_report.md` |
| 5 | Demonstration | `python demo.py` |

---

## ⚙️ Tech Stack

Python 3.10+ · OpenAI Whisper · pdfplumber · PyMuPDF · YOLOv8 · OpenCV · spaCy · HuggingFace Transformers · NLTK · pandas · Streamlit · Plotly

---

## 🏆 Marking Rubric

| Criteria | Weight | Description |
|----------|--------|-------------|
| Problem Understanding | 10% | Clear understanding of the scenario, objectives, and each student's role |
| Data Collection | 15% | Quality and relevance of datasets across all five modalities |
| AI Model Implementation | 25% | Correct and working use of AI techniques for each data type |
| Pipeline Design | 15% | Quality and clarity of the end-to-end AI pipeline architecture |
| Data Integration | 15% | Ability to successfully merge outputs from all five modalities |
| Code Quality | 10% | Clean, documented, and reproducible code with a working GitHub repo |
| Final Demonstration | 10% | Clarity of the live demo showing raw data converted to structured output |

---

*Build something that could help save lives and respond faster to emergencies.*
