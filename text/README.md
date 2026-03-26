# 📝 Text Analyst — Student 5

**Modality: Social Media Posts / News Articles**

This module processes written text from social media and news sources, performing NLP analysis to extract structured incident information from unstructured text data.

---

## 🎯 Responsibilities

- Clean and preprocess raw text: remove noise, normalize, tokenize
- Run Named Entity Recognition (NER) to extract: people, locations, organizations, dates
- Perform sentiment analysis and topic classification (accident / fire / theft / disturbance)
- Output a structured CSV with NLP analysis results

---

## 📤 Output Schema

| Text_ID | Crime_Type | Location_Entity | Sentiment | Topic | Severity_Label |
|---------|------------|-----------------|-----------|-------|----------------|
| TXT_112 | Robbery | Oak Street, Chicago | Negative | Theft / Robbery | High |

---

## 🛠️ Tools & Libraries

| Library | Purpose | Install |
|---------|---------|---------|
| `spaCy` | NER and text preprocessing | `pip install spacy` |
| `transformers` (HuggingFace) | Sentiment analysis and zero-shot topic classification | `pip install transformers` |
| `NLTK` | Tokenization, stopword removal, stemming | `pip install nltk` |
| `pandas` | Structured output generation | `pip install pandas` |

---

## 📦 Dataset

**CrimeReport** — Kaggle dataset containing real crime text reports with crime type, location, and details. Ready for NLP analysis.

- **Link:** [kaggle.com/datasets/cameliasiadat/crimereport](https://www.kaggle.com/datasets/cameliasiadat/crimereport)
- **Access:** Sign into Kaggle → open the link → click **Download**. Load the CSV directly with:

```python
df = pd.read_csv('crimereport.csv')
```

> **Tip:** The dataset is already in CSV format — skip scraping entirely. Focus on building NER with spaCy, sentiment analysis with HuggingFace, and topic classification.

---

## 🚀 Running the Module

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python text_analyzer.py
```

Output will be saved to `text_output.csv`.
