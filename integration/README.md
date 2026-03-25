# Integration — Full Team 🔗

## Task
Merge all 5 modality CSV outputs into one unified incident dataset with severity scoring and a Streamlit dashboard.

## Assignment Steps Implemented
1. **Incident_ID** — `INC_001`, `INC_002`, ... as common key across all CSVs
2. **Merge** — all 5 DataFrames row-aligned into one unified dataset
3. **Missing values** — filled with `N/A` via `df.fillna()`
4. **Severity scoring** — weighted model from all 5 modality signals
5. **Dashboard** — Streamlit app with charts, filters, and per-incident detail view

## Run

### Full pipeline (recommended)
```bash
python run_pipeline.py
```

### Integration only
```bash
python integration/integrate.py
```

### Dashboard only
```bash
streamlit run integration/dashboard.py
# Opens at http://localhost:8501
```

## Severity Scoring Logic
| Signal | Points |
|--------|--------|
| Audio urgency ≥ 0.8 | +3 |
| Audio urgency ≥ 0.5 | +2 |
| Audio sentiment = Distressed | +2 |
| Text severity = High | +3 |
| Text severity = Medium | +2 |
| Image confidence ≥ 0.85 | +2 |
| Video confidence ≥ 0.85 | +2 |
| **Score ≥ 8 → High \| Score ≥ 5 → Medium \| else → Low** | |

## Final Output Schema (`integrated_incidents.csv`)
| Incident_ID | Source | Event | Location | Time | Severity |
|-------------|--------|-------|----------|------|----------|
| INC_001 | Audio + PDF + Image + Video + Text | Fire | Downtown Ave | 00:00:12 | High |
| INC_002 | Audio + PDF | Accident | Main Street | 00:00:24 | Medium |
