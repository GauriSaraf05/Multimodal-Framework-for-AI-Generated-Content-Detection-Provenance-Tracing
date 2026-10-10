from pathlib import Path
import json
import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "datasets" / "text" / "raw" / "text_dataset.csv"

print("Downloading HC3 JSONL dataset...")

file_path = hf_hub_download(
    repo_id="Hello-SimpleAI/HC3",
    filename="all.jsonl",
    repo_type="dataset"
)

print("Download completed. Preparing CSV...")

records = []

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)

        for answer in item.get("human_answers", []):
            if isinstance(answer, str) and answer.strip():
                records.append({
                    "text": answer.strip(),
                    "label": 0
                })

        for answer in item.get("chatgpt_answers", []):
            if isinstance(answer, str) and answer.strip():
                records.append({
                    "text": answer.strip(),
                    "label": 1
                })

df = pd.DataFrame(records)

if df.empty:
    raise ValueError("No valid text samples found.")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False, encoding="utf-8")

print("\nDataset saved successfully!")
print("Total samples:", len(df))
print("\nClass distribution:")
print(df["label"].value_counts())
print("\nSaved at:", OUTPUT)