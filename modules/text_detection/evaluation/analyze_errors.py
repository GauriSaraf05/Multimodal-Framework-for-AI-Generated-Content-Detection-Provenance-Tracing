
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

INPUT = ROOT / "modules/text_detection/evaluation/all_models_predictions.csv"
OUTPUT = ROOT / "modules/text_detection/evaluation/error_analysis.csv"

df = pd.read_csv(INPUT)

models = [
    "Original_TFIDF",
    "Grouped_TFIDF",
    "DistilRoBERTa"
]

required = {"sample_id", "text", "label"}
required.update(f"{model}_prediction" for model in models)

missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

print("\nMODEL ERROR ANALYSIS")
print("=" * 60)

for model in models:
    pred_col = f"{model}_prediction"

    false_positives = df[
        (df["label"] == 0) &
        (df[pred_col] == 1)
    ]

    false_negatives = df[
        (df["label"] == 1) &
        (df[pred_col] == 0)
    ]

    print(f"\n{model}")
    print("False positives:", len(false_positives))
    print("False negatives:", len(false_negatives))

    df[f"{model}_error"] = (
        df["label"] != df[pred_col]
    )

# Find cases where all three models are incorrect
error_columns = [
    f"{model}_error" for model in models
]

df["all_models_wrong"] = df[error_columns].all(axis=1)

# Find cases where predictions disagree
prediction_columns = [
    f"{model}_prediction" for model in models
]

df["models_disagree"] = (
    df[prediction_columns].nunique(axis=1) > 1
)

# Approximate text length
df["word_count"] = df["text"].astype(str).str.split().str.len()

df["length_category"] = pd.cut(
    df["word_count"],
    bins=[0, 50, 150, 300, float("inf")],
    labels=["Short", "Medium", "Long", "Very Long"],
    include_lowest=True
)

print("\nAll models wrong:", df["all_models_wrong"].sum())
print("Models disagree:", df["models_disagree"].sum())

print("\nError rates by text length:")

for model in models:
    print(f"\n{model}")

    summary = df.groupby(
        "length_category",
        observed=True
    )[f"{model}_error"].agg(["count", "mean"])

    summary["mean"] = (summary["mean"] * 100).round(2)

    print(summary.rename(columns={"mean": "error_rate_%"}))

df.to_csv(OUTPUT, index=False)

print("\nSaved error analysis to:")
print(OUTPUT)
