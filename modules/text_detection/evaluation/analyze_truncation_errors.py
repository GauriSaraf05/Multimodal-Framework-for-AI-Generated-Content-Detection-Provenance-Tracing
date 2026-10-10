
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

TRUNCATION_PATH = ROOT / "modules/text_detection/evaluation/truncation_analysis.csv"
PREDICTIONS_PATH = ROOT / "modules/text_detection/evaluation/all_models_predictions.csv"

OUTPUT_PATH = ROOT / "modules/text_detection/evaluation/truncation_error_analysis.csv"

for path in [TRUNCATION_PATH, PREDICTIONS_PATH]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            "Complete Steps 23 and 26 first."
        )

truncation = pd.read_csv(TRUNCATION_PATH)
predictions = pd.read_csv(PREDICTIONS_PATH)

required = [
    "sample_id",
    "label",
    "DistilRoBERTa_prediction"
]

for column in required:
    if column not in predictions.columns:
        raise ValueError(f"Missing column: {column}")

# Keep only necessary prediction columns
prediction_data = predictions[
    ["sample_id", "DistilRoBERTa_prediction"]
]

df = truncation.merge(
    prediction_data,
    on="sample_id",
    how="inner",
    validate="one_to_one"
)

if len(df) != len(truncation):
    raise ValueError(
        "Some truncation records have no matching prediction. "
        "Check that both CSV files came from the same benchmark."
    )

df["incorrect"] = (
    df["label"] != df["DistilRoBERTa_prediction"]
)

df["error_type"] = "Correct"

df.loc[
    (df["label"] == 0) &
    (df["DistilRoBERTa_prediction"] == 1),
    "error_type"
] = "False Positive"

df.loc[
    (df["label"] == 1) &
    (df["DistilRoBERTa_prediction"] == 0),
    "error_type"
] = "False Negative"

print("\nTRUNCATION VS MODEL ERRORS")
print("=" * 55)

summary = df.groupby(
    ["label", "truncated"]
).agg(
    samples=("incorrect", "size"),
    errors=("incorrect", "sum"),
    average_tokens=("token_count", "mean")
)

summary["error_rate_%"] = (
    summary["errors"] / summary["samples"] * 100
).round(2)

print(summary.round(2).to_string())

print("\nERROR TYPES")
print(pd.crosstab(
    df["truncated"],
    df["error_type"]
).to_string())

print("\nAVERAGE TOKEN LENGTH BY RESULT")
print(
    df.groupby("error_type")["token_count"]
    .mean()
    .round(2)
    .to_string()
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("\nSaved:", OUTPUT_PATH)
