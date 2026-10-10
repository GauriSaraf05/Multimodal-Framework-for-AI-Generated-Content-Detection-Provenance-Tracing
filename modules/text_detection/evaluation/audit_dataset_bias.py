
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

TRAIN_PATH = (
    ROOT / "datasets/text/processed/improved_training/train.csv"
)

VAL_PATH = (
    ROOT / "datasets/text/processed/improved_training/validation.csv"
)

EXTERNAL_PATH = (
    ROOT / "datasets/text/processed/external_benchmark.csv"
)

datasets = {
    "Training": TRAIN_PATH,
    "Validation": VAL_PATH,
    "External": EXTERNAL_PATH
}

def audit_dataset(name, path):
    print(f"\n{'=' * 65}")
    print(name.upper())
    print("=" * 65)

    df = pd.read_csv(path)

    if not {"text", "label"}.issubset(df.columns):
        raise ValueError(f"Missing text/label columns in {path}")

    df = df.dropna(subset=["text", "label"]).copy()

    df["word_count"] = (
        df["text"].astype(str).str.split().str.len()
    )

    df["character_count"] = (
        df["text"].astype(str).str.len()
    )

    print("\nLabel distribution:")
    print(df["label"].value_counts().sort_index())

    print("\nWord-count statistics:")
    print(
        df.groupby("label")["word_count"]
        .describe()
        .round(2)
        .to_string()
    )

    print("\nCharacter-count statistics:")
    print(
        df.groupby("label")["character_count"]
        .describe()
        .round(2)
        .to_string()
    )

    if "source" in df.columns:
        print("\nSource distribution:")
        print(
            pd.crosstab(
                df["source"],
                df["label"]
            ).to_string()
        )

    print("\nDuplicate texts:", df["text"].duplicated().sum())

    return df

loaded = {}

for name, path in datasets.items():
    if not path.exists():
        raise FileNotFoundError(path)

    loaded[name] = audit_dataset(name, path)

print("\n" + "=" * 65)
print("CROSS-DATASET EXACT TEXT OVERLAP")
print("=" * 65)

train_texts = set(loaded["Training"]["text"].astype(str))
val_texts = set(loaded["Validation"]["text"].astype(str))
external_texts = set(loaded["External"]["text"].astype(str))

print("Train / Validation:", len(train_texts & val_texts))
print("Train / External:", len(train_texts & external_texts))
print("Validation / External:", len(val_texts & external_texts))

print("\nDataset audit completed.")
