
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = ROOT / "datasets/text/processed/grouped_dataset.csv"
MODEL_PATH = ROOT / "models/text/text_detector_grouped.pkl"
RESULT_PATH = ROOT / "modules/text_detection/evaluation/grouped_results.csv"

# 1. Load dataset
print("Loading grouped dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Grouped dataset not found. Run prepare_grouped_dataset.py first."
    )

df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["text", "label", "question_id"]).copy()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]
df["label"] = df["label"].astype(int)

print("Total samples:", len(df))
print("Unique questions:", df["question_id"].nunique())

# 2. Group-aware splitting
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(
        df["text"],
        df["label"],
        groups=df["question_id"]
    )
)

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

# 3. Verify question groups do not overlap
train_groups = set(train_df["question_id"])
test_groups = set(test_df["question_id"])

assert train_groups.isdisjoint(test_groups), (
    "Data leakage: question groups overlap!"
)

# 4. Remove exact text duplicates crossing train/test
train_texts = set(train_df["text"])
test_df = test_df[
    ~test_df["text"].isin(train_texts)
].copy()

# Also check for duplicate question wording across groups
if "question" in df.columns:
    train_questions = set(
        train_df["question"].dropna().astype(str).str.strip()
    )
    test_df = test_df[
        ~test_df["question"].fillna("").astype(str).str.strip().isin(
            train_questions
        )
    ].copy()

if test_df.empty:
    raise ValueError("Test set is empty after leakage checks.")

print("\nTraining samples:", len(train_df))
print("Testing samples:", len(test_df))

print("\nTraining distribution:")
print(train_df["label"].value_counts())

print("\nTesting distribution:")
print(test_df["label"].value_counts())

# 5. Create classification pipeline
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

# 6. Train model
print("\nTraining grouped model...")
model.fit(train_df["text"], train_df["label"])

print("Training completed!")

# 7. Evaluate model
predictions = model.predict(test_df["text"])

accuracy = accuracy_score(test_df["label"], predictions)

print("\n========== GROUPED MODEL RESULTS ==========")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    test_df["label"],
    predictions,
    labels=[0, 1],
    target_names=["Human", "AI"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(
    test_df["label"],
    predictions,
    labels=[0, 1]
))

# 8. Save test predictions
results = test_df[
    ["question_id", "text", "label"]
].copy()

results["predicted_label"] = predictions
results["ai_probability"] = model.predict_proba(
    test_df["text"]
)[:, list(model.classes_).index(1)]

RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(RESULT_PATH, index=False)

# 9. Save model
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("\nGrouped model saved:", MODEL_PATH)
print("Evaluation results saved:", RESULT_PATH)
