
from pathlib import Path
import json
import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]

OUTPUT = (
    ROOT / "datasets" / "text" / "processed"
    / "grouped_dataset.csv"
)

print("Locating HC3 dataset...")

file_path = hf_hub_download(
    repo_id="Hello-SimpleAI/HC3",
    filename="all.jsonl",
    repo_type="dataset"
)

print("Preparing question-grouped dataset...")

records = []

with open(file_path, "r", encoding="utf-8") as f:
    for question_id, line in enumerate(f):
        item = json.loads(line)

        question = item.get("question", "")

        for answer in item.get("human_answers", []):
            if isinstance(answer, str) and answer.strip():
                records.append({
                    "question_id": question_id,
                    "question": question,
                    "text": answer.strip(),
                    "label": 0
                })

        for answer in item.get("chatgpt_answers", []):
            if isinstance(answer, str) and answer.strip():
                records.append({
                    "question_id": question_id,
                    "question": question,
                    "text": answer.strip(),
                    "label": 1
                })

df = pd.DataFrame(records)

if df.empty:
    raise ValueError("No valid samples found.")

# Basic cleaning
df["text"] = (
    df["text"]
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

df = df[df["text"] != ""]

# Remove exact duplicate texts
df = df.drop_duplicates(subset=["text"])

# Remove texts associated with conflicting labels
conflicting = (
    df.groupby("text")["label"]
    .nunique()
)

conflicting_texts = conflicting[conflicting > 1].index
df = df[~df["text"].isin(conflicting_texts)]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("\nGrouped dataset created!")
print("Total samples:", len(df))
print("Unique question IDs:", df["question_id"].nunique())

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nSaved to:", OUTPUT)
