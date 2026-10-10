
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[3]

INPUT = ROOT / "datasets/text/processed/grouped_dataset.csv"
OUTPUT_DIR = ROOT / "datasets/text/processed/improved_training"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

required = {"text", "label", "question_id"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.dropna(subset=["text", "label", "question_id"]).copy()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() >= 30]
df = df[df["label"].isin([0, 1])]

# Remove repeated text and conflicting labels
df = df.drop_duplicates(subset=["text"])

# Group split prevents the same HC3 question ID
# from appearing in both partitions
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(df, groups=df["question_id"])
)

train_df = df.iloc[train_idx].copy()
val_df = df.iloc[val_idx].copy()

# Remove identical texts across partitions
train_texts = set(train_df["text"])
val_df = val_df[~val_df["text"].isin(train_texts)]

# Also exclude identical question wording
if "question" in df.columns:
    train_questions = set(
        train_df["question"].fillna("").astype(str).str.strip()
    )
    val_df = val_df[
        ~val_df["question"].fillna("").astype(str).str.strip().isin(
            train_questions
        )
    ]

def sample_balanced(data, per_class, seed):
    counts = data["label"].value_counts()

    for label in [0, 1]:
        if counts.get(label, 0) < per_class:
            raise ValueError(
                f"Not enough label {label} samples. "
                f"Available: {counts.get(label, 0)}"
            )

    return pd.concat([
        data[data["label"] == 0].sample(
            n=per_class, random_state=seed
        ),
        data[data["label"] == 1].sample(
            n=per_class, random_state=seed
        )
    ]).sample(frac=1, random_state=seed).reset_index(drop=True)

# 2,000 training samples; 400 validation samples
train_sample = sample_balanced(train_df, 1000, 42)
val_sample = sample_balanced(val_df, 200, 43)

train_path = OUTPUT_DIR / "train.csv"
val_path = OUTPUT_DIR / "validation.csv"

train_sample.to_csv(train_path, index=False)
val_sample.to_csv(val_path, index=False)

print("\nIMPROVED TRAINING DATA READY")
print("=" * 45)
print("Training samples:", len(train_sample))
print("Validation samples:", len(val_sample))

print("\nTraining labels:")
print(train_sample["label"].value_counts())

print("\nValidation labels:")
print(val_sample["label"].value_counts())

print("\nSaved:")
print(train_path)
print(val_path)
