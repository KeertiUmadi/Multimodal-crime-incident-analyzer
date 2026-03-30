# 📝 Text Analyst — Student 5

**Modality: Social Media Posts / News Articles**

This module processes written text from social media and news sources, performing NLP analysis to extract structured incident information from unstructured text data.

---

## 🎯 Responsibilities

- Load **CrimeReport** from `text/data/crimereport.txt` or `crimereport.csv` (Kaggle download)
- Clean and preprocess text (normalize, tokenize, NLTK stopwords)
- **spaCy NER** for **location** entities (GPE / LOC / FAC) used in `Location_Entity`
- **HuggingFace** sentiment (3-class) and **zero-shot** topic labels
- Write **`text/output_text.csv`** with the schema below

---

## 📤 Output Schema

| Text_ID | Crime_Type | Location_Entity | Sentiment | Topic | Severity_Label |
|---------|------------|-----------------|-----------|-------|----------------|
| TXT_112 | Robbery | Oak Street, Chicago | Negative | Theft/Robbery | High |

**CSV header (exact order):** `Text_ID,Crime_Type,Location_Entity,Sentiment,Topic,Severity_Label`

---

## 🛠️ Tools & Libraries

| Library | Purpose | Install |
|---------|---------|---------|
| `spaCy` | NER and text preprocessing | `pip install spacy` |
| `transformers` (HuggingFace) | Sentiment analysis and zero-shot topic classification | `pip install transformers` |
| `NLTK` | Tokenization, stopword removal, stemming | `pip install nltk` |
| `pandas` | Structured output generation | `pip install pandas` |

---

## 📦 Dataset — what goes in `text/data/`?

[Kaggle CrimeReport](https://www.kaggle.com/datasets/cameliasiadat/crimereport) is often distributed as **`.txt`**. Put **one** file in `text/data/`:

| File | When to use |
|------|-------------|
| **`crimereport.txt`** | **Default path.** Download from Kaggle, then copy or rename the `.txt` to **`text/data/crimereport.txt`**. Format: **one crime report per line** (or CSV-like lines are parsed). |
| **`crimereport.csv`** | Optional — if your download or a notebook export is CSV, save it here; the script uses it **only if** `crimereport.txt` is **not** present. |

**Tip:** No scraping — use the Kaggle file as-is (after renaming to `crimereport.txt` if the name differs). Then run spaCy NER, HuggingFace sentiment, and topic classification.

### Automatic download (`kagglehub`)

If **`text/data/crimereport.txt`** and **`crimereport.csv`** are both missing, the script can download the dataset the same way Kaggle documents:

```python
import kagglehub
path = kagglehub.dataset_download("cameliasiadat/crimereport")
```

1. `pip install kagglehub` (included in root `requirements.txt`).
2. Add your Kaggle API credentials (from **Kaggle → Account → API → Create New Token**): place **`kaggle.json`** in `~/.kaggle/` (or set the env vars Kaggle documents).
3. Run `python text/text_analyzer.py` — it will download, pick `.csv` or `.txt` inside the bundle, then build **`text/output_text.csv`**.

---

## 🚀 Running the Module

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python text_analyzer.py
```

Output is written to **`text/output_text.csv`** (run from repo root, or `output_text.csv` when cwd is `text/`).
