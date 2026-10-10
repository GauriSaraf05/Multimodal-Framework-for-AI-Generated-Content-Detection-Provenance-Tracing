
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score
)

ROOT = Path(__file__).resolve().parents[3]

INPUT = ROOT / "datasets/text/processed/external_benchmark.csv"
OUTPUT = ROOT / "modules/text_detection/evaluation/length_shortcut_analysis.csv"

df = pd.read_csv(INPUT)
df = df.dropna(subset=["text", "label"]).copy()

df["label"] = df["label"].astype(int)
df["word_count"] = df["text"].astype(str).str.split().str.len()

if not df["label"].isin([0, 1]).all():
    raise ValueError("Expected labels 0=human and 1=AI")

# Exploratory diagnostic: evaluate how strongly
# shorter text is associated with AI labels.
#
# This is NOT a trained classifier and the
# resulting AUC is not an unbiased test score.

labels = df["label"].to_numpy()
lengths = df["word_count"].to_numpy()

auc = roc_auc_score(labels, -lengths)

print("\nLENGTH SHORTCUT DIAGNOSTIC")
print("=" * 50)
print("AUC using shorter text as AI signal:", round(auc, 4))

print("\nCLASS LENGTH QUANTILES")
print(
    df.groupby("label")["word_count"]
    .quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    .unstack()
    .round(1)
    .to_string()
)

# Fixed illustrative thresholds, not optimized
# against the external benchmark.
thresholds = [50, 100, 150, 200, 300]

results = []

for threshold in thresholds:
    predictions = (lengths <= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    ).ravel()

    results.append({
        "max_words_for_ai": threshold,
        "accuracy": accuracy_score(labels, predictions),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    })

result_df = pd.DataFrame(results)

print("\nFIXED LENGTH THRESHOLD RESULTS")
print(result_df.to_string(index=False))

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result_df.to_csv(OUTPUT, index=False)

print("\nSaved:", OUTPUT)
