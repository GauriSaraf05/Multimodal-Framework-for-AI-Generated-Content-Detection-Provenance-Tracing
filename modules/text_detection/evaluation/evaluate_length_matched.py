
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = (
    ROOT / "datasets/text/processed/length_matched_development.csv"
)

OUTPUT_DIR = ROOT / "modules/text_detection/evaluation"

MODELS = {
    "DistilRoBERTa_V1": ROOT / "models/text/distilroberta_detector",
    "DistilRoBERTa_V2": ROOT / "models/text/distilroberta_v2"
}

MAX_LENGTH = 128
BATCH_SIZE = 8

df = pd.read_csv(DATA_PATH)

required = {"text", "label", "question_id"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

if df[["text", "label"]].isna().any().any():
    raise ValueError("Missing text or label")

texts = df["text"].astype(str).tolist()
labels = df["label"].astype(int).to_numpy()

if not set(labels).issubset({0, 1}):
    raise ValueError("Labels must be 0 or 1")

predictions_df = df.copy()
metrics = []

for model_name, model_path in MODELS.items():
    print(f"\nEvaluating {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )

    model.eval()
    model.to("cpu")

    ai_scores = []

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

            ai_scores.extend(
                probabilities[:, 1].cpu().tolist()
            )

    scores = np.asarray(ai_scores)
    predictions = (scores >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()

    metrics.append({
        "model": model_name,
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
        "roc_auc": roc_auc_score(labels, scores),
        "human_false_positive_rate": fp / (fp + tn),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    })

    predictions_df[f"{model_name}_prediction"] = predictions
    predictions_df[f"{model_name}_ai_score"] = scores

    del model

metrics_df = pd.DataFrame(metrics)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

metrics_path = OUTPUT_DIR / "length_matched_metrics.csv"
predictions_path = OUTPUT_DIR / "length_matched_predictions.csv"

metrics_df.to_csv(metrics_path, index=False)
predictions_df.to_csv(predictions_path, index=False)

print("\nLENGTH-MATCHED MODEL COMPARISON")
print("=" * 90)

print(metrics_df.round(4).to_string(index=False))

print("\nSaved:", metrics_path)
print("Saved:", predictions_path)
