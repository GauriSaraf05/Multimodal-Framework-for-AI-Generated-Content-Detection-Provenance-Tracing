
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = ROOT / "datasets/text/processed/cleaned_dataset.csv"
MODEL_PATH = ROOT / "models/text/text_detector.pkl"

print("Loading dataset and model...")

df = pd.read_csv(DATA_PATH).dropna(subset=["text", "label"])
model = joblib.load(MODEL_PATH)

# Recreate the exact test split used in train_baseline.py
_, X_test, _, y_test = train_test_split(
    df["text"],
    df["label"].astype(int),
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

print("\nEvaluating baseline model...")
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Human", "AI"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save incorrectly classified examples
errors = pd.DataFrame({
    "text": X_test.to_numpy(),
    "actual_label": y_test.to_numpy(),
    "predicted_label": predictions
})

errors = errors[
    errors["actual_label"] != errors["predicted_label"]
]

OUTPUT = ROOT / "modules/text_detection/evaluation/misclassified.csv"

errors.to_csv(OUTPUT, index=False)

print("\nTotal misclassified samples:", len(errors))
print("Saved error analysis to:", OUTPUT)
