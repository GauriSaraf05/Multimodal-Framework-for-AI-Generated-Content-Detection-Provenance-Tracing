# AI-Generated Content Detection & Explainable AI

A research-oriented system for detecting AI-like and human-like writing patterns using deep learning, with interpretable model outputs.

## Features

- AI-like, Human-like, and Uncertain text classification
- DistilRoBERTa-based text detection
- Word-level token occlusion explanations
- Highlighting of influential words
- Interactive React dashboard
- FastAPI backend
- Raw model scores and input statistics

## Technology Stack

**Frontend:** React, Vite, CSS

**Backend:** Python, FastAPI

**Machine Learning:** PyTorch, Hugging Face Transformers, DistilRoBERTa

## Running the Project

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install the Python dependencies.
4. Place the trained model files in `models/text/distilroberta_v2/`.
5. Start the FastAPI backend using `python -m uvicorn backend.main:app --reload`.
6. Run `npm install` and `npm run dev` inside `frontend/`.

## Important Limitations

The detector is experimental. Its raw scores are not calibrated probabilities of authorship, and its generalization to independent real-world content remains unverified. The current model uses a maximum input length of 128 tokens.

## Future Scope

- Image-based AI-generated content detection
- Audio detection
- Provenance tracing
- Improved model evaluation and explainability
- Unified analysis dashboard

## Project Status

Text detection MVP under development.