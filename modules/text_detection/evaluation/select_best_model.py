
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

EVAL_DIR = ROOT / "modules/text_detection/evaluation"
COMPARISON_PATH = EVAL_DIR / "transformer_comparison.csv"
OUTPUT_PATH = EVAL_DIR / "model_selection.json"

if not COMPARISON_PATH.exists():
    raise FileNotFoundError(
        "Run compare_transformer_versions.py first."
    )

df = pd.read_csv(COMPARISON_PATH)

required = {
    "model",
    "accuracy",
    "precision_ai",
    "recall_ai",
    "f1_ai",
    "human_false_positive_rate"
}

missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

# Provisional screening criteria, not validated
# production acceptance thresholds.
MAX_FALSE_POSITIVE_RATE = 0.10
MIN_ACCURACY = 0.70
MIN_PRECISION = 0.70

eligible = df[
    (df["human_false_positive_rate"] <= MAX_FALSE_POSITIVE_RATE)
    & (df["accuracy"] >= MIN_ACCURACY)
    & (df["precision_ai"] >= MIN_PRECISION)
].copy()

result = {
    "evaluation_type": "external_development_screening",
    "thresholds": {
        "max_human_false_positive_rate": MAX_FALSE_POSITIVE_RATE,
        "min_accuracy": MIN_ACCURACY,
        "min_precision_ai": MIN_PRECISION
    },
    "selected_model": None,
    "status": "not_ready",
    "reason": ""
}

if eligible.empty:
    result["reason"] = (
        "No model passed the provisional screening criteria. "
        "More evaluation and improvement are required."
    )
else:
    eligible = eligible.sort_values(
        by=[
            "human_false_positive_rate",
            "f1_ai",
            "accuracy"
        ],
        ascending=[True, False, False]
    )

    best = eligible.iloc[0]

    result["selected_model"] = str(best["model"])
    result["status"] = "development_candidate"
    result["reason"] = (
        "Model passed provisional development screening. "
        "Independent testing and calibration are still required."
    )

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(result, file, indent=4)

print("\nMODEL SELECTION RESULT")
print("=" * 50)
print("Status:", result["status"])
print("Selected model:", result["selected_model"])
print("Reason:", result["reason"])
print("\nSaved:", OUTPUT_PATH)
