
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = ROOT / "modules/text_detection/evaluation"

INPUT = EVAL_DIR / "transformer_predictions.csv"
OUTPUT = EVAL_DIR / "score_distribution_analysis.csv"

df = pd.read_csv(INPUT)

required = [
    "label",
    "DistilRoBERTa_V1_ai_score",
    "DistilRoBERTa_V2_ai_score"
]

for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

print("\nAI SCORE DISTRIBUTION BY ACTUAL CLASS")
print("=" * 65)

for model in ["DistilRoBERTa_V1", "DistilRoBERTa_V2"]:
    score_col = f"{model}_ai_score"

    print(f"\n{model}")

    print(
        df.groupby("label")[score_col]
        .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        .round(4)
        .to_string()
    )

    print("\nScore ranges:")
    bins = [-0.001, 0.1, 0.5, 0.9, 0.99, 1.001]
    categories = pd.cut(
        df[score_col],
        bins=bins,
        labels=[
            "0.00-0.10",
            "0.10-0.50",
            "0.50-0.90",
            "0.90-0.99",
            "0.99-1.00"
        ]
    )

    print(pd.crosstab(df["label"], categories).to_string())

df.to_csv(OUTPUT, index=False)

print("\nSaved:", OUTPUT)
