
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

OUTPUT = (
    ROOT / "datasets" / "text" / "raw"
    / "external_dataset.csv"
)

columns = [
    "sample_id",
    "text",
    "label",
    "source",
    "category",
    "model_name",
    "prompt_id"
]

# Preserve existing work if the file already exists
if OUTPUT.exists():
    print("Dataset already exists:", OUTPUT)
    print("No changes made.")
else:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(columns=columns)
    df.to_csv(OUTPUT, index=False, encoding="utf-8")

    print("Dataset template created!")
    print("Saved to:", OUTPUT)
    print("Columns:", columns)
