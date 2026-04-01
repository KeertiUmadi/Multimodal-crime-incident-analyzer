## Demonstration Script (What to Run / What to Show)

This is a simple, grader-friendly flow to demonstrate that your project:
1) runs the five modality analyzers,
2) merges them into one incident dataset (`integration_output.csv`),
3) computes `Severity`, and
4) provides a working Step 5 dashboard to filter/search incidents.

---

## Deliverable 5 — “Raw → structured incident report” (video / live demo)

**Goal:** Show that **each unstructured source type** is turned into **rows in a CSV**, then **one merged table** is the final structured incident report.

### One-sentence pitch

“We take raw audio, PDFs, images, video, and text; each modality has its own AI pipeline that writes a structured CSV; then integration merges them on `Incident_ID` and assigns `Severity`; the dashboard is the interactive view of that final report.”

### Suggested screen order (record this or follow live)

| Step | What you show on screen | What you say (short) |
|------|-------------------------|----------------------|
| 1 | Repo tree: `audio/`, `pdf/`, `images/`, `video/`, `text/`, `integration/` | Each folder is one modality plus shared integration. |
| 2 | Open **one raw file** per type (e.g. a `.wav`, a `.pdf`, a `.jpg`, a `.mpg` clip, `text/data/CrimeReport.txt`) | This is unstructured input — not usable for analytics as-is. |
| 3 | Run `python run_pipeline.py` (or show terminal output from a recent run) | This runs all five analyzers, then merge. |
| 4 | Open each `output_*.csv` side by side or one after another | Here the same evidence is now **structured columns** (IDs, events, types, scores). |
| 5 | Open `integration/integration_output.csv` | This is the **final incident report**: one row per incident, all modalities + `Severity`. |
| 6 | Run `streamlit run integration/dashboard.py` (or `run_dashboard.bat`) | This is how users **query and export** that report. |

### Printable storyline + sample rows (terminal)

From the repo root:

```bat
python scripts\demo_story.py
```

That script prints the same flow and the **first row** of each modality CSV plus the merged file (good cheat sheet while recording).

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

