
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Project root
ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = ROOT / "datasets/text/processed/cleaned_dataset.csv"
MODEL_PATH = ROOT / "models/text/text_detector.pkl"

# Load dataset
print("Loading cleaned dataset...")
df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["text", "label"])
df["label"] = df["label"].astype(int)

print("Total samples:", len(df))
print("\nClass distribution:")
print(df["label"].value_counts())

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Build text classification pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        lowercase=True,
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

# Train the model
print("\nTraining started...")
model.fit(X_train, y_train)
print("Training completed!")

# Evaluate the model
predictions = model.predict(X_test)

print("\n========== MODEL RESULTS ==========")

print(
    "Accuracy:",
    round(accuracy_score(y_test, predictions) * 100, 2),
    "%"
)

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Human", "AI"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

# Save model
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully!")
print("Location:", MODEL_PATH)
