## Demonstration Script (What to Run / What to Show)

This is a simple, grader-friendly flow to demonstrate that your project:
1) runs the five modality analyzers,
2) merges them into one incident dataset (`integration_output.csv`),
3) computes `Severity`, and
4) provides a working Step 5 dashboard to filter/search incidents.

---

### Prerequisites

From the repo root (`Multimodal-crime-incident-analyzer-1`):

```bat
pip install -r requirements.txt
```

---

### Demo Steps

#### 1) Run the full pipeline (creates all modality CSVs + integration CSV)

```bat
python run_pipeline.py
```

What to show (open these CSVs):
- `audio/output_audio.csv`
- `pdf/output_pdf.csv`
- `images/output_images.csv`
- `video/output_video.csv`
- `text/output_text.csv`

Then show the integrated output:
- `integration/integration_output.csv`

Quick verification (optional):

```bat
python -c "import pandas as pd; df=pd.read_csv('integration/integration_output.csv'); print(len(df)); print(df['Severity'].value_counts().to_dict())"
```

---

#### 2) Open the Step 5 dashboard (display + filter + dropdown search)

```bat
streamlit run integration/dashboard.py
```

What to do in the dashboard:
1. Use the **Severity** multiselect to filter (e.g., select `High`).
2. Use **Search → Field + Value**:
   - Pick a `Field` (for example: **PDF doc type** or **Audio event**).
   - Pick a `Value` from the dropdown (the dropdown is populated from your integrated CSV).
3. Confirm the table row count updates (**Showing X**).
4. Download the filtered results via **Download CSV**.

---

### Optional: Text query interface (if you want a non-Streamlit demo)

```bat
python integration/integrate.py --cli-query
```

Then choose:
- severity filter, or
- keyword search (matches across modality summary fields)

