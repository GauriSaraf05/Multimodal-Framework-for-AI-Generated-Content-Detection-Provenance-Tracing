
from pathlib import Path
import pandas as pd

# Project root
ROOT = Path(__file__).resolve().parents[3]

# Dataset paths
RAW_PATH = ROOT / "datasets" / "text" / "raw" / "text_dataset.csv"
PROCESSED_PATH = ROOT / "datasets" / "text" / "processed" / "cleaned_dataset.csv"

def prepare_dataset():
    print("Loading dataset...")

    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)

    print("\nOriginal dataset shape:", df.shape)
    print("\nColumns:", df.columns.tolist())

    # Check required columns
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("Dataset must contain text and label columns")

    # Remove missing values
    df = df.dropna(subset=["text", "label"]).copy()

    # Clean whitespace without removing meaningful punctuation
    df["text"] = df["text"].astype(str).str.strip()
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True)

    # Remove empty texts
    df = df[df["text"] != ""]

    # Validate labels
    df["label"] = pd.to_numeric(df["label"], errors="raise")

    if not df["label"].isin([0, 1]).all():
        raise ValueError("Labels must be 0 (Human) or 1 (AI)")

    df["label"] = df["label"].astype(int)

    # Remove duplicates
    df = df.drop_duplicates(subset=["text"])

    # Remove contradictory labels if present
    df = df.groupby("text").filter(
        lambda group: group["label"].nunique() == 1
    )

    # Save cleaned dataset
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print("\nPreprocessing completed!")
    print("Total cleaned samples:", len(df))

    print("\nClass distribution:")
    print(df["label"].value_counts())

    print("\nSaved at:", PROCESSED_PATH)


if __name__ == "__main__":
    prepare_dataset()
