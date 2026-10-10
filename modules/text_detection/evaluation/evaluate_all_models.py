
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = ROOT / "datasets/text/processed/external_benchmark.csv"
OUTPUT_DIR = ROOT / "modules/text_detection/evaluation"

MODEL_PATHS = {
    "Original_TFIDF": ROOT / "models/text/text_detector.pkl",
    "Grouped_TFIDF": ROOT / "models/text/text_detector_grouped.pkl",
}

TRANSFORMER_PATH = ROOT / "models/text/distilroberta_detector"

# -------------------------------
# Load and validate external data
# -------------------------------

df = pd.read_csv(DATA_PATH)

required = {"sample_id", "text", "label"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=["text", "label"]).copy()
df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() > 0]

if not df["label"].isin([0, 1]).all():
    raise ValueError("Labels must be 0 (HUMAN) or 1 (AI).")

if df["sample_id"].duplicated().any():
    raise ValueError("Duplicate sample IDs found.")

if df["text"].duplicated().any():
    raise ValueError("Duplicate texts found.")

print("External samples:", len(df))
print("Human samples:", (df["label"] == 0).sum())
print("AI samples:", (df["label"] == 1).sum())

if df["label"].nunique() < 2:
    raise ValueError(
        "External evaluation requires both HUMAN and AI samples."
    )

if len(df) < 20:
    print(
        "WARNING: Very small dataset. "
        "Metrics will not be reliable."
    )

y_true = df["label"].to_numpy()

# -------------------------------
# Evaluation helper
# -------------------------------

def evaluate(name, predictions, ai_scores):
    print(f"\n{'=' * 55}")
    print(name)
    print("=" * 55)

    print(
        classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            target_names=["HUMAN", "AI"],
            zero_division=0
        )
    )

    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    print("Confusion matrix:")
    print(cm)

    df[f"{name}_prediction"] = predictions
    df[f"{name}_ai_score"] = ai_scores

    return {
        "model": name,
        "accuracy": accuracy_score(y_true, predictions),
        "precision_ai": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall_ai": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1_ai": f1_score(
            y_true, predictions, zero_division=0
        ),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


results = []

# -------------------------------
# Evaluate TF-IDF models
# -------------------------------

for name, path in MODEL_PATHS.items():
    print(f"\nLoading {name}...")
    model = joblib.load(path)

    probabilities = model.predict_proba(df["text"].tolist())
    classes = list(model.classes_)
    ai_index = classes.index(1)

    ai_scores = probabilities[:, ai_index]
    predictions = (ai_scores >= 0.5).astype(int)

    results.append(
        evaluate(name, predictions, ai_scores)
    )

# -------------------------------
# Evaluate DistilRoBERTa
# -------------------------------

print("\nLoading DistilRoBERTa...")

tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_PATH)
transformer = AutoModelForSequenceClassification.from_pretrained(
    TRANSFORMER_PATH
)

transformer.eval()
transformer.to("cpu")

texts = df["text"].tolist()
all_scores = []

batch_size = 8

with torch.inference_mode():
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        outputs = transformer(**inputs)
        probabilities = torch.softmax(
            outputs.logits, dim=-1
        )

        all_scores.extend(
            probabilities[:, 1].cpu().tolist()
        )

ai_scores = np.array(all_scores)
predictions = (ai_scores >= 0.5).astype(int)

results.append(
    evaluate("DistilRoBERTa", predictions, ai_scores)
)

# -------------------------------
# Save evaluation results
# -------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

summary = pd.DataFrame(results)

summary_path = OUTPUT_DIR / "all_models_metrics.csv"
predictions_path = OUTPUT_DIR / "all_models_predictions.csv"

summary.to_csv(summary_path, index=False)
df.to_csv(predictions_path, index=False)

print("\nMODEL COMPARISON SUMMARY")
print(summary.to_string(index=False))

print("\nSaved metrics:", summary_path)
print("Saved predictions:", predictions_path)
