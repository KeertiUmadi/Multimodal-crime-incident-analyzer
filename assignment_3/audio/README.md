# Student 1 — Audio Analyst 🎙️

## Task
Convert emergency audio calls to text, extract incident info, and score urgency.

## Tools
- `openai-whisper` — speech-to-text transcription
- `spaCy` — named entity recognition (locations)
- `HuggingFace transformers (DistilBERT)` — sentiment classification

## Dataset
**911 Calls + Wav2Vec2** — [Kaggle Link](https://www.kaggle.com/code/stpeteishii/911-calls-wav2vec2)
1. Sign in to Kaggle → open link → click **Copy & Edit**
2. Run with free Kaggle GPU (no download needed)
3. To run locally: download .wav files → place in `audio/data/`

## Run
```bash
pip install -r audio/requirements.txt
python audio/audio_analyzer.py
```
> No files in `audio/data/`? Demo data runs automatically.

## Output: `audio/output_audio.csv`
| Call_ID | Transcript | Extracted_Event | Location | Sentiment | Urgency_Score |
|---------|-----------|-----------------|----------|-----------|---------------|
| C001 | There is a fire... | Fire | Downtown Ave | Distressed | 0.91 |
| C002 | Car crash on Main... | Accident | Main Street | Distressed | 0.75 |
