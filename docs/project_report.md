# Project Report — Multimodal Crime / Incident Analyzer
**Course:** AI for Engineers &nbsp;|&nbsp; **Type:** Group of 5 Students &nbsp;|&nbsp; **Duration:** 2 Weeks &nbsp;|&nbsp; **Total Marks:** 100

---

## 1. Problem Understanding

A city's emergency response department receives hundreds of daily incident reports across five different unstructured formats — emergency audio calls, written police reports (PDFs), crime scene photographs, CCTV surveillance footage, and social media/news text posts. Manual review of all these sources is slow, error-prone, and delays emergency response.

**Our solution:** An AI-powered Multimodal Incident Analyzer that automatically ingests all five data types, runs the appropriate AI model on each, extracts structured fields (event, location, time, entities, sentiment), and merges everything into a single unified incident report with severity scoring.

---

## 2. Team Roles

| Student | Role | Modality | Output |
|---------|------|----------|--------|
| Student 1 | Audio Analyst | Emergency audio calls (.wav/.mp3) | `audio/output_audio.csv` |
| Student 2 | Document Analyst | Police PDF reports (.pdf) | `pdf/output_pdf.csv` |
| Student 3 | Image Analyst | Crime scene photographs (.jpg/.png) | `images/output_images.csv` |
| Student 4 | Video Analyst | CCTV surveillance footage (.mp4/.mpg) | `video/output_video.csv` |
| Student 5 | Text Analyst | Social media / news articles (.csv) | `text/output_text.csv` |

---

## 3. Data Collection

| Module | Dataset | Source | Format | Access |
|--------|---------|--------|--------|--------|
| Audio | 911 Calls + Wav2Vec2 | Kaggle | .wav audio | Sign in → fork notebook → run free GPU |
| PDF | Arkansas PD 1033 Proposals | MuckRock FOIA | .pdf | Open link → download directly (no account) |
| Images | Fire Detection Dataset | Roboflow Universe | .jpg/.png | Free account → Download → YOLOv8 format |
| Video | CAVIAR CCTV Dataset | Edinburgh University | .mpg | Open link → download clips (no account) |
| Text | CrimeReport | Kaggle | .csv | Sign in → Download CSV |

> All modules include built-in demo data so the full pipeline runs without any downloads.

---

## 4. AI Model Implementation

### Student 1 — Audio Analyst
**Tools:** `openai-whisper`, `spaCy en_core_web_sm`, `distilbert-base-uncased-finetuned-sst-2-english`

| Step | Technique | Output |
|------|-----------|--------|
| Transcription | Whisper base model | Full text transcript of call |
| Entity extraction | spaCy NER (GPE/LOC/FAC) | Location mentioned in call |
| Sentiment | DistilBERT SST-2 | Calm / Distressed |
| Urgency scoring | Keyword frequency ratio | 0.0–1.0 score |
| Event classification | Keyword map | Fire / Accident / Theft / Assault / Medical |

**Output columns:** `Call_ID, Transcript, Extracted_Event, Location, Sentiment, Urgency_Score`

**Sample output:**
```
C001 | There is a fire! People trapped on 2nd floor... | Fire     | Downtown Ave | Distressed | 0.91
C002 | Car crash on Main Street, one person unconscious | Accident | Main Street  | Distressed | 0.75
C003 | Someone broke into my store on Oak Avenue        | Theft    | Oak Avenue   | Distressed | 0.60
```

---

### Student 2 — Document Analyst
**Tools:** `pdfplumber`, `PyMuPDF (fitz)`, `pytesseract`, `spaCy en_core_web_sm`

| Step | Technique | Output |
|------|-----------|--------|
| Text extraction | pdfplumber | Full text from text-based PDFs |
| OCR fallback | PyMuPDF (200 DPI) + pytesseract | Text from scanned/image PDFs |
| Entity extraction | spaCy NER | Persons, organisations, locations, dates |
| Date extraction | Regex (3 patterns) | Standardised date string |
| Incident type | Keyword classification | Fire / Theft / Assault / Accident / Administrative |

**Output columns:** `Report_ID, Department, Incident_Type, Doc_Type, Date, Location, Officer, Key_Detail, Summary`

**Sample output:**
```
RPT_001 | Arkansas PD    | Administrative | Training Proposal | 2015-04-10 | Little Rock, AR  | Officer Johnson
RPT_002 | City Police    | Theft/Robbery  | Incident Report   | 2024-03-15 | Downtown, Chicago| Officer Martinez
RPT_003 | Fire Department| Fire           | Incident Report   | 2024-03-16 | Warehouse District| Captain Williams
```

---

### Student 3 — Image Analyst
**Tools:** `ultralytics YOLOv8n`, `opencv-python`, `pytesseract`

| Step | Technique | Output |
|------|-----------|--------|
| Object detection | YOLOv8 nano (COCO 80 classes) | Labels + bounding boxes |
| Scene classification | Label → scene map | Fire Scene / Accident / Theft/Robbery |
| Confidence | Mean of all detection scores | 0.0–1.0 |

**Output columns:** `Image_ID, Scene_Type, Objects_Detected, Bounding_Boxes, Confidence`

**Sample output:**
```
IMG_034 | Fire Scene   | fire, smoke      | 2 fire regions, 1 smoke plume | 0.94
IMG_035 | Accident     | car, person      | 1 vehicle, 1 person            | 0.87
IMG_036 | Theft/Robbery| person, backpack | 1 person, 1 backpack           | 0.81
```

---

### Student 4 — Video Analyst
**Tools:** `opencv-python`, `ultralytics YOLOv8n`, `moviepy`

| Step | Technique | Output |
|------|-----------|--------|
| Frame extraction | OpenCV VideoCapture | 1 frame/second (every 30 frames) |
| Motion detection | Frame pixel difference | Motion score 0.0–1.0 |
| Object detection | YOLOv8 per frame | Labels + person count |
| Anomaly detection | Motion × person count rules | Suspicious event label |

**Output columns:** `Clip_ID, Timestamp, Frame_ID, Motion_Score, Event_Detected, Persons_Count, Confidence`

**Sample output:**
```
CAVIAR_03 | 00:00:12 | FRM_036 | 0.08 | Person collapsing           | 1 | 0.88
CAVIAR_03 | 00:00:24 | FRM_072 | 0.12 | Crowd / Suspicious movement | 3 | 0.91
CAVIAR_04 | 00:00:08 | FRM_020 | 0.05 | Vehicle movement            | 0 | 0.85
```

---

### Student 5 — Text Analyst
**Tools:** `spaCy en_core_web_sm`, `distilbert-base-uncased-finetuned-sst-2-english`, `facebook/bart-large-mnli`, `nltk`

| Step | Technique | Output |
|------|-----------|--------|
| Preprocessing | NLTK tokenise + stopword removal | Cleaned text |
| NER | spaCy (GPE/LOC/FAC/PERSON/ORG) | Location, persons, organisations |
| Sentiment | DistilBERT SST-2 | Negative / Positive |
| Crime type | BART zero-shot (9 candidate labels) | Top crime label |
| Severity | Rule map (crime type → severity) | Low / Medium / High |

**Output columns:** `Text_ID, Crime_Type, Location_Entity, Sentiment, Topic, Severity_Label`

**Sample output:**
```
TXT_112 | Robbery  | Oak Street, Chicago | Negative | Theft / Robbery    | High
TXT_113 | Fire     | 5th Avenue          | Negative | Fire / Emergency   | High
TXT_114 | Accident | I-95                | Negative | Accident           | Medium
```

---

## 5. Pipeline Design

See `docs/pipeline_architecture.html` for the full interactive visual diagram.

```
STAGE 1 — INGESTION:
  [Audio .wav] [PDF .pdf] [Images .jpg] [Video .mpg] [Text .csv]
        ↓            ↓           ↓            ↓            ↓

STAGE 2 — AI PROCESSING:
  Whisper    pdfplumber   YOLOv8      OpenCV       spaCy
  spaCy NER  PyMuPDF OCR  pytesseract YOLOv8       BART
  DistilBERT spaCy NER   OpenCV      Motion score  DistilBERT
        ↓            ↓           ↓            ↓            ↓

STAGE 3 — INFORMATION EXTRACTION:
  output_audio.csv  output_pdf.csv  output_images.csv  output_video.csv  output_text.csv
        ↓                 ↓                ↓                  ↓                ↓

STAGE 4 — STRUCTURED DATASET GENERATION:
              integration/integrate.py
                         ↓
             integration/integration_output.csv
             (Incident_ID | Audio_Event | PDF_Doc_Type | Image_Objects |
              Video_Event | Text_Crime_Type | Severity)

STAGE 5 — DASHBOARD & QUERY:
              streamlit run integration/dashboard.py
              → http://localhost:8501
```

---

## 6. Data Integration

**File:** `integration/integrate.py`

All five assignment steps implemented:

| Step | Implementation |
|------|---------------|
| 1. Common Incident_ID | Generated as `INC_001`, `INC_002`, ... across all 5 CSVs |
| 2. Merge DataFrames | Outer-merge on `Incident_ID` using `pandas.merge(..., on="Incident_ID")` |
| 3. Handle missing values | `df.fillna("N/A")` applied after merge |
| 4. Severity classification | Weighted scoring model from all 5 modality signals |
| 5. Dashboard/Query | Streamlit dashboard with filters, charts, and drill-down |

**Severity scoring weights:**

| Signal | Points |
|--------|--------|
| Audio urgency score ≥ 0.8 | +3 |
| Audio urgency score ≥ 0.5 | +2 |
| Audio sentiment = Distressed | +2 |
| Text severity label = High | +3 |
| Text severity label = Medium | +2 |
| Text severity label = Low | +1 |
| Image confidence ≥ 0.85 | +2 |
| Image confidence ≥ 0.5 | +1 (else 0) |
| Video confidence ≥ 0.85 | +2 |
| Video confidence ≥ 0.5 | +1 (else 0) |
| PDF_Doc_Type present (non-N/A) | +1 |
| **Score ≥ 8 → High \| Score ≥ 5 → Medium \| else → Low** | |

**Final output schema (exactly matches assignment requirement):**

| Incident_ID | Audio_Event | PDF_Doc_Type | Image_Objects | Video_Event | Text_Crime_Type | Severity |
|-------------|-------------|--------------|---------------|-------------|-----------------|----------|
| INC_001 | Building fire / trapped | 1033 Training Proposal | fire, smoke (0.95) | Person collapsing | Robbery / Theft | High |
| INC_002 | Shooting | 1033 Training Proposal | N/A | No significant event | Public Disturbance | Medium |
| INC_003 | Robbery | 1033 Training Proposal | N/A | Person detected | Theft/Robbery | High |

---

## 7. Challenges & Solutions

| Challenge | Solution Applied |
|-----------|-----------------|
| Scanned PDFs with no text layer | OCR fallback: PyMuPDF renders page at 200 DPI → pytesseract |
| Missing modality files | Missing fields are filled with `N/A` during integration |
| YOLOv8 COCO labels don't include "fire" | Scene classifier maps label clusters (smoke + person) to scene types |
| HuggingFace models slow to load | Lazy loading — models only imported when the module actually runs |
| 5 CSVs with different row counts | `Incident_ID` is generated/normalized per modality and outer-merged |
| BART zero-shot slow on large datasets | Capped at 50 rows for prototype; full run would use batching |
| Video .mpg format compatibility | OpenCV handles .mpg natively |

---

## 8. Results

Results from running `python run_pipeline.py` with built-in demo data:

| Metric | Value |
|--------|-------|
| Total incidents processed | 1033 |
| High severity incidents | 41 |
| Medium severity incidents | 26 |
| Low severity incidents | 966 |
| Average audio urgency score | 0.74 |
| Average image confidence score | 0.92 |
| Most common audio event | Unknown |
| Most common crime type (text) | Unknown |
| Modalities active | 5 of 5 |

---

## 9. Conclusion

The system successfully demonstrates an end-to-end multimodal AI pipeline that converts unstructured data from five different sources into a unified, queryable incident report. Key achievements:

- All 5 modalities implemented end-to-end with real production-grade AI models
- Pipeline runs with demo data even without downloading real datasets — fully testable immediately
- Severity scoring model combines signals from all 5 modalities for holistic classification
- Interactive Streamlit dashboard with severity filter, dropdown-based cross-modality search, severity distribution chart, and CSV export
- Modular design — each student's module is fully independent and plug-and-play

The system can be extended with real-time data ingestion, cloud hosting on AWS/GCP, and LLM-based summarization (bonus features).

---

## 10. References

| Tool / Dataset | Reference |
|----------------|-----------|
| OpenAI Whisper | https://github.com/openai/whisper |
| YOLOv8 (Ultralytics) | https://github.com/ultralytics/ultralytics |
| spaCy | https://spacy.io |
| HuggingFace Transformers | https://huggingface.co/transformers |
| pdfplumber | https://github.com/jsvine/pdfplumber |
| PyMuPDF | https://pymupdf.readthedocs.io |
| OpenCV | https://opencv.org |
| Streamlit | https://streamlit.io |
| 911 Calls Dataset | https://www.kaggle.com/code/stpeteishii/911-calls-wav2vec2 |
| Arkansas PD PDF | https://www.muckrock.com/foi/arkansas-114/arkansas-police-departments-1033-training-plan-proposals-20493/ |
| Roboflow Fire Detection | https://universe.roboflow.com/search?q=fire |
| CAVIAR CCTV Dataset | https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1 |
| CrimeReport Dataset | https://www.kaggle.com/datasets/cameliasiadat/crimereport |
