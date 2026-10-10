
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = ROOT / "datasets/text/processed/external_benchmark.csv"
OUTPUT_DIR = ROOT / "modules/text_detection/evaluation"

MODELS = {
    "DistilRoBERTa_V1": ROOT / "models/text/distilroberta_detector",
    "DistilRoBERTa_V2": ROOT / "models/text/distilroberta_v2"
}

MAX_LENGTH = 128
BATCH_SIZE = 8

# Check files before loading models
if not DATA_PATH.is_file():
    raise FileNotFoundError(f"Benchmark not found: {DATA_PATH}")

for name, path in MODELS.items():
    if not (path / "config.json").is_file():
        raise FileNotFoundError(
            f"{name} not found at {path}. Complete its training first."
        )

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["text", "label"]).copy()

df["text"] = df["text"].astype(str).str.strip()
df["label"] = df["label"].astype(int)

if not df["label"].isin([0, 1]).all():
    raise ValueError("Labels must be 0 or 1")

if df["label"].nunique() != 2:
    raise ValueError("Both HUMAN and AI examples are required")

texts = df["text"].tolist()
labels = df["label"].to_numpy()

results = []
predictions_df = df[["sample_id", "text", "label"]].copy()

for name, model_path in MODELS.items():
    print(f"\nLoading {name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )

    model.to("cpu")
    model.eval()

    scores = []

    with torch.inference_mode():
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start:start + BATCH_SIZE]

            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )

            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)

            scores.extend(
                probabilities[:, 1].cpu().tolist()
            )

    scores = np.array(scores)
    predictions = (scores >= 0.5).astype(int)

    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    result = {
        "model": name,
        "accuracy": accuracy_score(labels, predictions),
        "precision_ai": precision_score(
            labels, predictions, zero_division=0
        ),
        "recall_ai": recall_score(
            labels, predictions, zero_division=0
        ),
        "f1_ai": f1_score(
            labels, predictions, zero_division=0
        ),
        "human_false_positive_rate": fp / (fp + tn)
        if (fp + tn) else 0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }

    results.append(result)

    predictions_df[f"{name}_prediction"] = predictions
    predictions_df[f"{name}_ai_score"] = scores

    print(f"\n{name} RESULTS")
    print("=" * 45)

    for key, value in result.items():
        print(f"{key}: {value}")

    del model

summary = pd.DataFrame(results)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

summary_path = OUTPUT_DIR / "transformer_comparison.csv"
predictions_path = OUTPUT_DIR / "transformer_predictions.csv"

summary.to_csv(summary_path, index=False)
predictions_df.to_csv(predictions_path, index=False)

print("\nFINAL MODEL COMPARISON")
print("=" * 75)

print(
    summary[
        [
            "model",
            "accuracy",
            "precision_ai",
            "recall_ai",
            "f1_ai",
            "human_false_positive_rate"
        ]
    ].round(4).to_string(index=False)
)

print("\nSaved:", summary_path)
print("Saved:", predictions_path)
