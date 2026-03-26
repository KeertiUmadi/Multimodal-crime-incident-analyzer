# 🎙️ Audio Analyst — Student 1

**Modality: Emergency Audio Calls / Witness Voice Statements**

This module processes audio recordings of emergency calls and witness statements, converting spoken content into structured, analyzable incident information.

---

## 🎯 Responsibilities

- Convert audio files to text using a speech-to-text model (OpenAI Whisper)
- Extract keywords: incident type, location mentions, names, urgency phrases
- Perform sentiment/urgency analysis on transcribed text (calm vs. distressed)
- Output a structured CSV with extracted incident fields

---

## 📤 Output Schema

| Call_ID | Transcript (excerpt) | Extracted_Event | Location | Sentiment | Urgency_Score |
|---------|----------------------|-----------------|----------|-----------|---------------|
| C001 | There is a fire, people are trapped on second floor | Building fire / trapped persons | Downtown Ave | Distressed | 0.91 |

---

## 🛠️ Tools & Libraries

| Library | Purpose | Install |
|---------|---------|---------|
| `openai-whisper` | Speech-to-text transcription | `pip install openai-whisper` |
| `spaCy` / `NLTK` | Keyword and entity extraction from transcript | `pip install spacy nltk` |
| `transformers` (HuggingFace) | Sentiment analysis using pre-trained models | `pip install transformers` |

---

## 📦 Dataset

**911 Calls + Wav2Vec2** — Real 911 emergency audio calls with a ready-made Wav2Vec2 speech-to-text notebook.

- **Link:** [kaggle.com/code/stpeteishii/911-calls-wav2vec2](https://www.kaggle.com/code/stpeteishii/911-calls-wav2vec2)
- **Access:** Sign in to Kaggle → open the link → click **Copy and Edit** to fork → run with free Kaggle GPU. No downloads needed.

> **Tip:** The notebook already includes working transcription code. Run it first, then add keyword extraction and sentiment analysis on top of the transcribed text output.

---

## 🚀 Running the Module

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python audio_analyzer.py
```

Output will be saved to `audio_output.csv`.
