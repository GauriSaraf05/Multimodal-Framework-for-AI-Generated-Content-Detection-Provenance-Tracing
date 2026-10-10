
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = ROOT / "modules/text_detection/evaluation"

INPUT = EVAL_DIR / "transformer_predictions.csv"
OUTPUT = EVAL_DIR / "threshold_analysis.csv"

df = pd.read_csv(INPUT)

labels = df["label"].astype(int).to_numpy()

models = [
    "DistilRoBERTa_V1",
    "DistilRoBERTa_V2"
]

thresholds = np.arange(0.50, 1.00, 0.05)

results = []

for model in models:
    scores = df[f"{model}_ai_score"].to_numpy()

    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            labels,
            predictions,
            labels=[0, 1]
        ).ravel()

        results.append({
            "model": model,
            "threshold": round(float(threshold), 2),
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
            "human_false_positive_rate": fp / (fp + tn),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        })

results_df = pd.DataFrame(results)

results_df.to_csv(OUTPUT, index=False)

print("\nDISTILROBERTA V2 THRESHOLD ANALYSIS")
print("=" * 85)

v2 = results_df[
    results_df["model"] == "DistilRoBERTa_V2"
]

print(
    v2[
        [
            "threshold",
            "accuracy",
            "precision_ai",
            "recall_ai",
            "human_false_positive_rate",
            "fp",
            "fn"
        ]
    ].round(3).to_string(index=False)
)

print("\nSaved:", OUTPUT)
