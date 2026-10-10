
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "distilbert/distilroberta-base"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading pretrained transformer...")
model = AutoModel.from_pretrained(MODEL_NAME)

text = "Artificial intelligence is changing the world."

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=128
)

model.eval()

with torch.no_grad():
    outputs = model(**inputs)

print("\nTransformer loaded successfully!")
print("Tokenized input shape:", inputs["input_ids"].shape)
print("Last hidden state shape:", outputs.last_hidden_state.shape)
