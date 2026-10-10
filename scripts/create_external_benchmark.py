
from pathlib import Path
from collections import Counter

import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]

OUTPUT = ROOT / "datasets/text/processed/external_benchmark.csv"
TRAINING_DATA = ROOT / "datasets/text/processed/cleaned_dataset.csv"

DATASET_NAME = "artem9k/ai-text-detection-pile"
TARGET_PER_CLASS = 100

print("Loading external dataset...")

dataset = load_dataset(
    DATASET_NAME,
    split="train",
    streaming=True
)

# Avoid exact matches with HC3 training data
training_texts = set()

if TRAINING_DATA.exists():
    train_df = pd.read_csv(
        TRAINING_DATA,
        usecols=["text"]
    )

    training_texts = set(
        train_df["text"].dropna().astype(str).str.strip()
    )

    print("Training texts loaded:", len(training_texts))

samples = []
counts = Counter()
seen = set()

for record in dataset:
    source = str(record.get("source", "")).strip().lower()
    text = str(record.get("text") or "").strip()

    if source == "human":
        label = 0
    elif source in ("ai", "machine", "generated", "chatgpt"):
        label = 1
    else:
        continue

    if counts[label] >= TARGET_PER_CLASS:
        continue

    if len(text) < 100:
        continue

    if text in seen or text in training_texts:
        continue

    seen.add(text)
    counts[label] += 1

    samples.append({
        "sample_id": f"EXT_{label}_{counts[label]:03d}",
        "text": text,
        "label": label,
        "source": DATASET_NAME,
        "category": source,
        "model_name": "",
        "prompt_id": ""
    })

    if all(
        counts[label] >= TARGET_PER_CLASS
        for label in [0, 1]
    ):
        break

print("\nCollected samples:")
print("Human:", counts[0])
print("AI:", counts[1])

if counts[0] == 0 or counts[1] == 0:
    raise RuntimeError(
        "Could not collect both classes. "
        "Inspect dataset source values before continuing."
    )

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame(samples)
df.to_csv(OUTPUT, index=False)

print("\nBenchmark saved to:")
print(OUTPUT)
