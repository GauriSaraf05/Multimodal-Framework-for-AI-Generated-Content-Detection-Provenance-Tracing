
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[3]

SOURCE = ROOT / "datasets/text/processed/grouped_dataset.csv"
TRAIN = ROOT / "datasets/text/processed/improved_training/train.csv"
VALIDATION = ROOT / "datasets/text/processed/improved_training/validation.csv"

OUTPUT = (
    ROOT / "datasets/text/processed/"
    "length_matched_development.csv"
)

SEED = 42
MIN_WORDS = 50
MAX_WORDS = 250
BIN_WIDTH = 25
TARGET_PER_CLASS = 100

df = pd.read_csv(SOURCE)
train_df = pd.read_csv(TRAIN)
val_df = pd.read_csv(VALIDATION)

required = {"text", "label", "question_id"}

for name, data in [
    ("source", df),
    ("train", train_df),
    ("validation", val_df)
]:
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")

# Exclude all question IDs used in V2 training/validation.
used_ids = set(train_df["question_id"].astype(str))
used_ids.update(val_df["question_id"].astype(str))

df = df.dropna(subset=["text", "label", "question_id"]).copy()

df["question_id"] = df["question_id"].astype(str)
df = df[~df["question_id"].isin(used_ids)].copy()

# Remove exact text overlap as an additional safeguard.
used_texts = set(train_df["text"].astype(str))
used_texts.update(val_df["text"].astype(str))

df = df[~df["text"].astype(str).isin(used_texts)].copy()

df["word_count"] = df["text"].astype(str).str.split().str.len()

df = df[
    (df["word_count"] >= MIN_WORDS)
    & (df["word_count"] <= MAX_WORDS)
    & (df["label"].isin([0, 1]))
].copy()

df["length_bin"] = (
    (df["word_count"] // BIN_WIDTH) * BIN_WIDTH
)

# Match human and AI counts inside each length bin.
# Sampling is deterministic for reproducibility.
rng = np.random.default_rng(SEED)
parts = []

for length_bin, group in df.groupby("length_bin"):
    human = group[group["label"] == 0]
    ai = group[group["label"] == 1]

    n = min(len(human), len(ai))

    if n == 0:
        continue

    human = human.sample(
        n=n,
        random_state=int(rng.integers(1, 1_000_000))
    )

    ai = ai.sample(
        n=n,
        random_state=int(rng.integers(1, 1_000_000))
    )

    parts.extend([human, ai])

if not parts:
    raise ValueError("No overlapping length bins were found.")

matched = pd.concat(parts, ignore_index=True)

# Select the same number from each class.
# A single random sample is used after bin matching.
# Final length distributions are checked below.
human = matched[matched["label"] == 0]
ai = matched[matched["label"] == 1]

n = min(TARGET_PER_CLASS, len(human), len(ai))

human = human.sample(n=n, random_state=SEED)
ai = ai.sample(n=n, random_state=SEED)

result = pd.concat([human, ai], ignore_index=True)
result = result.sample(frac=1, random_state=SEED)
result = result.reset_index(drop=True)

# Verify no question IDs from the V2 split are present.
assert not (
    set(result["question_id"].astype(str)) & used_ids
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT, index=False)

print("\nLENGTH-MATCHED DEVELOPMENT DATASET")
print("=" * 55)
print("Total samples:", len(result))

print("\nClass distribution:")
print(result["label"].value_counts().sort_index())

print("\nWord-count statistics:")
print(
    result.groupby("label")["word_count"]
    .describe()
    .round(2)
    .to_string()
)

print("\nLength-bin distribution:")
print(
    pd.crosstab(
        result["length_bin"],
        result["label"]
    ).to_string()
)

print("\nSaved:", OUTPUT)

if n < TARGET_PER_CLASS:
    print(
        "\nNote: Fewer than 100 examples per class "
        "were available after matching."
    )
