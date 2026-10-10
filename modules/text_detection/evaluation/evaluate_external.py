
from pathlib import Path
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = ROOT / "datasets/text/processed/external_test.csv"

MODELS = {
    "Original Baseline": ROOT / "models/text/text_detector.pkl",
    "Grouped Baseline": ROOT / "models/text/text_detector_grouped.pkl"
}

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["text", "label"])

if df.empty:
    raise ValueError("External evaluation dataset is empty.")

df["label"] = df["label"].astype(int)

if not df["label"].isin([0, 1]).all():
    raise ValueError("Labels must be 0 or 1.")

print("External samples:", len(df))
print("\nClass distribution:")
print(df["label"].value_counts())

for name, model_path in MODELS.items():
    if not model_path.exists():
        print(f"\nSkipping {name}: model file not found.")
        continue

    model = joblib.load(model_path)

    predictions = model.predict(df["text"])
    probabilities = model.predict_proba(df["text"])

    ai_index = list(model.classes_).index(1)

    print(f"\n========== {name} ==========")

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

    print("\nIndividual predictions:")

    for i, (prediction, probability) in enumerate(
        zip(predictions, probabilities[:, ai_index]),
        start=1
    ):
        print(
            f"Sample {i}: "
            f"Actual={df.iloc[i-1]['label']}, "
            f"Predicted={prediction}, "
            f"AI Probability={probability * 100:.2f}%"
        )
