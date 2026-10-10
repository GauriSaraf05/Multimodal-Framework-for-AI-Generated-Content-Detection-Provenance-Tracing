
from pathlib import Path
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = ROOT / "datasets/text/raw/external_dataset.csv"
OUTPUT_PATH = ROOT / "modules/text_detection/evaluation/external_predictions.csv"

MODELS = {
    "Original": ROOT / "models/text/text_detector.pkl",
    "Grouped": ROOT / "models/text/text_detector_grouped.pkl"
}

# Load external dataset
print("Loading external dataset...")

df = pd.read_csv(DATA_PATH)

if df.empty:
    raise ValueError(
        "External dataset is empty. Add samples before evaluation."
    )

df = df.dropna(subset=["text", "label"]).copy()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]
df["label"] = df["label"].astype(int)

if not df["label"].isin([0, 1]).all():
    raise ValueError("Labels must be 0 or 1.")

print("\nTotal samples:", len(df))
print("\nClass distribution:")
print(df["label"].value_counts())

results = df.copy()

# Evaluate both models
for name, path in MODELS.items():

    if not path.exists():
        print(f"\nSkipping {name}: model not found.")
        continue

    model = joblib.load(path)

    predictions = model.predict(df["text"])
    probabilities = model.predict_proba(df["text"])

    ai_index = list(model.classes_).index(1)

    results[f"{name}_prediction"] = predictions
    results[f"{name}_ai_probability"] = (
        probabilities[:, ai_index] * 100
    ).round(2)

    results[f"{name}_correct"] = (
        predictions == df["label"].to_numpy()
    )

    print(f"\n========== {name.upper()} MODEL ==========")

    print(
        "Accuracy:",
        round(accuracy_score(df["label"], predictions) * 100, 2),
        "%"
    )

    print("\nClassification Report:")
    print(classification_report(
        df["label"],
        predictions,
        labels=[0, 1],
        target_names=["Human", "AI"],
        zero_division=0
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(
        df["label"],
        predictions,
        labels=[0, 1]
    ))

# Save comparison results
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT_PATH, index=False)

print("\nEvaluation completed!")
print("Results saved to:", OUTPUT_PATH)

print("\nPrediction comparison:")
display_columns = [
    "sample_id",
    "label",
    "Original_prediction",
    "Original_ai_probability",
    "Grouped_prediction",
    "Grouped_ai_probability"
]

available_columns = [
    col for col in display_columns
    if col in results.columns
]

print(results[available_columns].to_string(index=False))
