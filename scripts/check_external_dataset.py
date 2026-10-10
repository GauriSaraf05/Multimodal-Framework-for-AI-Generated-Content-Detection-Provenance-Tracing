
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "datasets/text/raw/external_dataset.csv"

df = pd.read_csv(path)

print("\nEXTERNAL DATASET REPORT")
print("=" * 40)

print("Total samples:", len(df))
print("Human samples:", (df["label"] == 0).sum())
print("AI samples:", (df["label"] == 1).sum())

print("\nMissing text:", df["text"].isna().sum())
print("Duplicate text:", df["text"].duplicated().sum())
print("Duplicate IDs:", df["sample_id"].duplicated().sum())

if df["label"].nunique() == 2:
    print("\nBoth classes are present.")
else:
    print("\nBoth classes are NOT present.")
    print("Add verified samples from the missing class.")
