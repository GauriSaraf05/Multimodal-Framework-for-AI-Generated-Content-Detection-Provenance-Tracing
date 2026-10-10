
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

path = ROOT / "modules/text_detection/evaluation/all_models_predictions.csv"
df = pd.read_csv(path)

models = [
    "Original_TFIDF",
    "Grouped_TFIDF",
    "DistilRoBERTa"
]

df["word_count"] = df["text"].astype(str).str.split().str.len()

print("\nDATASET LENGTH DISTRIBUTION")
print(df.groupby("label")["word_count"].describe().round(1))

print("\nPREDICTION DISTRIBUTION")

for model in models:
    col = f"{model}_prediction"
    score = f"{model}_ai_score"

    print(f"\n{model}")

    print("Predicted labels:")
    print(pd.crosstab(
        df["label"],
        df[col],
        rownames=["Actual"],
        colnames=["Predicted"]
    ))

    print("\nAI score distribution:")
    print(df.groupby("label")[score].describe().round(3))

print("\nEXAMPLE FALSE POSITIVES")

false_positives = df[
    (df["label"] == 0) &
    (df["DistilRoBERTa_prediction"] == 1)
]

for _, row in false_positives.head(5).iterrows():
    print("\nSample:", row["sample_id"])
    print("Words:", row["word_count"])
    print("Text:", row["text"][:500])
    print("AI score:", row["DistilRoBERTa_ai_score"])
