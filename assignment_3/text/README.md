# Student 5 — Text Analyst 📝

## Task
Process social media posts and news articles with NLP — NER, sentiment, and crime-type classification.

## Tools
- `spaCy` — Named Entity Recognition (persons, locations, organisations)
- `HuggingFace DistilBERT` — sentiment analysis
- `HuggingFace BART` — zero-shot crime-type classification (9 labels)
- `NLTK` — tokenization and stopword removal

## Dataset
**CrimeReport** — [Kaggle Link](https://www.kaggle.com/datasets/cameliasiadat/crimereport)
1. Sign into Kaggle → open link → Download
2. Place at `text/data/crimereport.csv`
3. OR use built-in `sample_data/sample_text.csv` (no download needed)

## Run
```bash
pip install -r text/requirements.txt
python -m spacy download en_core_web_sm
python text/text_analyzer.py
```

## Output: `text/output_text.csv`
| Text_ID | Source | Sentiment | Location_Entity | Crime_Type | Topic | Severity_Label |
|---------|--------|-----------|-----------------|------------|-------|----------------|
| TXT_112 | Twitter | Negative | Oak Street, Chicago | Robbery | Theft/Robbery | High |
