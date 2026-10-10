
from pathlib import Path
import hashlib
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT / "datasets/text/processed"
EVAL_DIR = ROOT / "modules/text_detection/evaluation"

BENCHMARK = DATA_DIR / "length_matched_development.csv"
TRAIN = DATA_DIR / "improved_training/train.csv"
VALIDATION = DATA_DIR / "improved_training/validation.csv"

OUTPUT = EVAL_DIR / "benchmark_integrity_report.json"

def normalized_hash(text):
    normalized = " ".join(str(text).lower().split())
    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()

benchmark = pd.read_csv(BENCHMARK)
train = pd.read_csv(TRAIN)
validation = pd.read_csv(VALIDATION)

for name, df in [
    ("benchmark", benchmark),
    ("train", train),
    ("validation", validation)
]:
    if not {"text", "question_id"}.issubset(df.columns):
        raise ValueError(f"Missing columns in {name}")

    if df[["text", "question_id"]].isna().any().any():
        raise ValueError(f"Missing text or question ID in {name}")

benchmark_hashes = set(
    benchmark["text"].map(normalized_hash)
)

training_hashes = set(
    train["text"].map(normalized_hash)
)

validation_hashes = set(
    validation["text"].map(normalized_hash)
)

benchmark_questions = set(
    benchmark["question_id"].astype(str)
)

training_questions = set(
    train["question_id"].astype(str)
)

validation_questions = set(
    validation["question_id"].astype(str)
)

report = {
    "benchmark_samples": len(benchmark),
    "train_samples": len(train),
    "validation_samples": len(validation),
    "normalized_text_overlap_train": len(
        benchmark_hashes & training_hashes
    ),
    "normalized_text_overlap_validation": len(
        benchmark_hashes & validation_hashes
    ),
    "question_id_overlap_train": len(
        benchmark_questions & training_questions
    ),
    "question_id_overlap_validation": len(
        benchmark_questions & validation_questions
    ),
    "limitations": [
        "Benchmark and training data originate from HC3.",
        "V1 training overlap has not been independently verified.",
        "Near-duplicate and paraphrase overlap are not tested.",
        "External domain generalization remains unverified."
    ]
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4)

print("\nBENCHMARK INTEGRITY REPORT")
print("=" * 55)

for key, value in report.items():
    if key != "limitations":
        print(f"{key}: {value}")

print("\nLIMITATIONS")
for item in report["limitations"]:
    print("-", item)

print("\nSaved:", OUTPUT)
