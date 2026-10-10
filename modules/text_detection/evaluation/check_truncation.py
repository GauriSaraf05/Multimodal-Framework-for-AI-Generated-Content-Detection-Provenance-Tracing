
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = ROOT / "datasets/text/processed/external_benchmark.csv"
MODEL_PATH = ROOT / "models/text/distilroberta_detector"
OUTPUT_PATH = ROOT / "modules/text_detection/evaluation/truncation_analysis.csv"

MAX_LENGTH = 128

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["text", "label"]).copy()

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Counting tokens...")

df["token_count"] = [
    len(tokenizer.encode(text, truncation=False))
    for text in df["text"].astype(str)
]

df["truncated"] = df["token_count"] > MAX_LENGTH

df["tokens_removed"] = (
    df["token_count"] - MAX_LENGTH
).clip(lower=0)

df["percent_removed"] = (
    df["tokens_removed"] / df["token_count"] * 100
).round(2)

print("\nOVERALL TRUNCATION ANALYSIS")
print("=" * 50)

print("Total samples:", len(df))
print("Truncated samples:", df["truncated"].sum())
print("Truncation rate:", round(df["truncated"].mean() * 100, 2), "%")

print("\nTRUNCATION BY CLASS")

summary = df.groupby("label").agg(
    samples=("text", "count"),
    average_tokens=("token_count", "mean"),
    truncated_samples=("truncated", "sum"),
    average_percent_removed=("percent_removed", "mean")
)

summary["truncation_rate_%"] = (
    summary["truncated_samples"] / summary["samples"] * 100
).round(2)

print(summary.round(2).to_string())

print("\nTOKEN LENGTH STATISTICS")

print(
    df.groupby("label")["token_count"]
    .describe()
    .round(2)
    .to_string()
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("\nSaved:", OUTPUT_PATH)
