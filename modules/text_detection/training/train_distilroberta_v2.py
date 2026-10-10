
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# -----------------------------------
# Configuration
# -----------------------------------

ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT / "datasets/text/processed/improved_training"

TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "validation.csv"

MODEL_NAME = "distilbert/distilroberta-base"
OUTPUT_DIR = ROOT / "models/text/distilroberta_v2"

MAX_LENGTH = 128
BATCH_SIZE = 4
EPOCHS = 2
LEARNING_RATE = 2e-5
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------------
# Load datasets
# -----------------------------------

print("Loading training data...")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

for name, df in [("train", train_df), ("validation", val_df)]:
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError(f"{name} dataset must contain text and label")

    if df[["text", "label"]].isna().any().any():
        raise ValueError(f"{name} dataset contains missing values")

    if not df["label"].isin([0, 1]).all():
        raise ValueError(f"{name} dataset has invalid labels")

if set(train_df["text"]) & set(val_df["text"]):
    raise ValueError("Training and validation contain overlapping text")

if (
    "question_id" in train_df.columns
    and "question_id" in val_df.columns
):
    if set(train_df["question_id"]) & set(val_df["question_id"]):
        raise ValueError(
            "Training and validation contain overlapping question IDs"
        )

print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))

# -----------------------------------
# Tokenization
# -----------------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class TextDataset(Dataset):
    def __init__(self, dataframe):
        self.texts = dataframe["text"].astype(str).tolist()
        self.labels = dataframe["label"].astype(int).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoded = tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(
                self.labels[idx],
                dtype=torch.long
            )
        }

train_dataset = TextDataset(train_df)
val_dataset = TextDataset(val_df)

# -----------------------------------
# Load pretrained model
# -----------------------------------

print("Loading DistilRoBERTa...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "HUMAN", 1: "AI"},
    label2id={"HUMAN": 0, "AI": 1}
)

# -----------------------------------
# Evaluation metrics
# -----------------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0
    )

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision_ai": precision,
        "recall_ai": recall,
        "f1_ai": f1
    }

# -----------------------------------
# Training configuration
# -----------------------------------

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR / "checkpoints"),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=50,
    report_to="none",
    use_cpu=True,
    dataloader_num_workers=0,
    seed=SEED
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# -----------------------------------
# Train
# -----------------------------------

print("\nStarting DistilRoBERTa V2 training...")
print("Device: CPU")

trainer.train()

# -----------------------------------
# Evaluate
# -----------------------------------

print("\nEvaluating V2 model...")

metrics = trainer.evaluate()

print("\nVALIDATION RESULTS")
print("=" * 50)

for key, value in metrics.items():
    print(f"{key}: {value}")

# -----------------------------------
# Save model and results
# -----------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

with open(
    OUTPUT_DIR / "validation_metrics.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metrics, file, indent=4)

predictions_output = trainer.predict(val_dataset)
predicted_labels = np.argmax(
    predictions_output.predictions,
    axis=-1
)

cm = confusion_matrix(
    val_df["label"],
    predicted_labels,
    labels=[0, 1]
)

print("\nCONFUSION MATRIX")
print(cm)

print("\nModel saved to:", OUTPUT_DIR)
print("Training completed successfully!")
