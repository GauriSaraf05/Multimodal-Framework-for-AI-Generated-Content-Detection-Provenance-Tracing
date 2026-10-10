
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "datasets/text/raw/external_dataset.csv"

df = pd.read_csv(FILE)

samples = [
    {
        "sample_id": "AI_001",
        "text": (
            "Artificial Intelligence is transforming industries "
            "by enabling machines to perform tasks that "
            "traditionally require human intelligence. "
            "Machine learning algorithms identify patterns "
            "in large datasets and support decision-making "
            "across healthcare, education, and finance. "
            "However, responsible AI development requires "
            "attention to fairness, transparency, privacy, "
            "and accountability."
        ),
        "label": 1,
        "source": "ChatGPT",
        "category": "Technology",
        "model_name": "ChatGPT",
        "prompt_id": "P001"
    },
    {
        "sample_id": "AI_002",
        "text": (
            "Climate change is a major global challenge "
            "that affects ecosystems and communities. "
            "Rising temperatures, extreme weather events, "
            "and changes in precipitation patterns create "
            "risks for agriculture, biodiversity, and "
            "infrastructure. Governments and organizations "
            "must collaborate to reduce emissions and "
            "promote sustainable development."
        ),
        "label": 1,
        "source": "ChatGPT",
        "category": "Environment",
        "model_name": "ChatGPT",
        "prompt_id": "P002"
    }
]

new_df = pd.DataFrame(samples)

# Avoid inserting duplicate sample IDs
new_df = new_df[
    ~new_df["sample_id"].isin(df["sample_id"])
]

df = pd.concat([df, new_df], ignore_index=True)

df.to_csv(FILE, index=False, encoding="utf-8")

print("Samples added successfully!")
print("Total samples:", len(df))
print("\nClass distribution:")
print(df["label"].value_counts())
