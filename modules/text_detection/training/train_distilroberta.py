
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = ROOT / "datasets/text/processed/grouped_dataset.csv"
MODEL_DIR = ROOT / "models/text/distilroberta_detector"

MODEL_NAME = "distilbert/distilroberta-base"

# CPU-friendly experiment settings
TRAIN_SAMPLES_PER_CLASS = 200
VAL_SAMPLES_PER_CLASS = 50
MAX_LENGTH = 128

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["text", "label", "question_id"]).copy()
df["label"] = df["label"].astype(int)

# Split by question before sampling
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(df, df["label"], groups=df["question_id"])
)

train_df = df.iloc[train_idx].copy()
val_df = df.iloc[val_idx].copy()

# Exclude duplicate text across splits
train_texts = set(train_df["text"])
val_df = val_df[~val_df["text"].isin(train_texts)]

# Exclude identical question wording across splits
if "question" in df.columns:
    train_questions = set(
        train_df["question"].fillna("").astype(str).str.strip()
    )
    val_df = val_df[
        ~val_df["question"].fillna("").astype(str).str.strip().isin(
            train_questions
        )
    ]

# Select balanced subsets
def balanced_sample(data, count):
    return pd.concat([
        data[data["label"] == 0].sample(
            n=count, random_state=42
        ),
        data[data["label"] == 1].sample(
            n=count, random_state=42
        )
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

train_df = balanced_sample(train_df, TRAIN_SAMPLES_PER_CLASS)
val_df = balanced_sample(val_df, VAL_SAMPLES_PER_CLASS)

print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(value[idx])
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(
            self.labels[idx], dtype=torch.long
        )
        return item

train_dataset = TextDataset(
    train_df["text"].tolist(),
    train_df["label"].tolist()
)

val_dataset = TextDataset(
    val_df["text"].tolist(),
    val_df["label"].tolist()
)

print("Loading DistilRoBERTa classifier...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "HUMAN", 1: "AI"},
    label2id={"HUMAN": 0, "AI": 1}
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, zero_division=0)
    }

training_args = TrainingArguments(
    output_dir=str(MODEL_DIR / "checkpoints"),
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=20,
    report_to="none",
    use_cpu=True,
    dataloader_num_workers=0
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

print("\nStarting CPU fine-tuning...")
trainer.train()

print("\nEvaluating model...")
metrics = trainer.evaluate()
print(metrics)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

trainer.save_model(str(MODEL_DIR))
tokenizer.save_pretrained(str(MODEL_DIR))

print("\nModel saved successfully!")
print("Location:", MODEL_DIR)
