
from pathlib import Path
import json

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

ROOT = Path(__file__).resolve().parents[3]

SELECTION_PATH = (
    ROOT / "modules/text_detection/evaluation/model_selection.json"
)

MODEL_PATHS = {
    "DistilRoBERTa_V1": ROOT / "models/text/distilroberta_detector",
    "DistilRoBERTa_V2": ROOT / "models/text/distilroberta_v2"
}

MIN_WORDS = 50

# Illustrative abstention thresholds only.
# They are NOT calibrated probabilities or
# validated decision thresholds.
LOW_THRESHOLD = 0.20
HIGH_THRESHOLD = 0.80

def load_candidate():
    if not SELECTION_PATH.exists():
        raise FileNotFoundError(
            "Run select_best_model.py first."
        )

    with open(SELECTION_PATH, encoding="utf-8") as file:
        selection = json.load(file)

    name = selection.get("selected_model")

    if selection.get("status") != "development_candidate":
        return None, None, None

    if name not in MODEL_PATHS:
        raise ValueError(f"Unknown model: {name}")

    path = MODEL_PATHS[name]

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)

    model.eval()
    return name, tokenizer, model


def predict(text, name, tokenizer, model):
    word_count = len(text.split())

    if word_count < MIN_WORDS:
        return {
            "prediction": "INSUFFICIENT_TEXT",
            "ai_score": None,
            "reason": "At least 50 words are required for this demo."
        }

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.inference_mode():
        logits = model(**inputs).logits
        ai_score = torch.softmax(logits, dim=-1)[0, 1].item()

    if ai_score <= LOW_THRESHOLD:
        prediction = "HUMAN_LIKE"
    elif ai_score >= HIGH_THRESHOLD:
        prediction = "AI_LIKE"
    else:
        prediction = "UNCERTAIN"

    return {
        "model": name,
        "prediction": prediction,
        "ai_score": round(ai_score, 4),
        "reason": (
            "Experimental classification only. "
            "AI score is not a calibrated authorship probability."
        )
    }


if __name__ == "__main__":
    name, tokenizer, model = load_candidate()

    if model is None:
        print(
            "No model passed development screening. "
            "Inference is disabled pending further evaluation."
        )
    else:
        text = input("Enter text to analyze: ").strip()

        result = predict(text, name, tokenizer, model)

        print("\nRESULT")
        print(json.dumps(result, indent=4))
