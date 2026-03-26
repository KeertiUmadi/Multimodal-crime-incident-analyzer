# 🔗 Integration — Full Team

**Final Merging, Dataset Generation & Dashboard**

This module combines all five individual modality outputs into a single unified incident dataset, then visualizes it through an interactive Streamlit dashboard.

---

## 🎯 Responsibilities

This is the most important deliverable of the project. The full team collaborates to:

1. Define a common `Incident_ID` key across all five output CSVs
2. Merge all five DataFrames using `pandas` merge/join on `Incident_ID`
3. Handle missing values where a modality has no data for a given incident
4. Generate a final severity classification (Low / Medium / High) based on combined signals
5. Build a dashboard or query interface to display and filter incident summaries

---

## 📤 Final Integrated Output Schema

| Incident_ID | Audio_Event | PDF_Doc_Type | Image_Objects | Video_Event | Text_Crime_Type | Severity |
|-------------|-------------|--------------|---------------|-------------|-----------------|----------|
| INC_001 | Building fire / trapped | 1033 Training Proposal | fire, smoke (0.94) | Person collapsing | Robbery / Theft | High |

---

## 📁 Files

| File | Description |
|------|-------------|
| `integrate.py` | Merges all 5 modality CSVs into `integrated_incidents.csv` |
| `dashboard.py` | Streamlit dashboard for visualizing and filtering incidents |
| `requirements.txt` | Dependencies for this module |

---

## 🛠️ Tools & Libraries

| Library | Purpose | Install |
|---------|---------|---------|
| `pandas` | DataFrame merging, join operations, missing value handling | `pip install pandas` |
| `streamlit` | Interactive web dashboard | `pip install streamlit` |
| `plotly` | Charts and visualizations in the dashboard | `pip install plotly` |

---

## 🚀 Running the Module

```bash
pip install -r requirements.txt

# Run integration only
python integrate.py

# Launch the dashboard
streamlit run dashboard.py
```

The merged output is saved to `integrated_incidents.csv`.

---

## 📊 Dashboard Features

- Filter incidents by severity (Low / Medium / High)
- Filter by event type or source modality
- View incident timeline and location breakdown
- Query individual incident details across all 5 data sources
