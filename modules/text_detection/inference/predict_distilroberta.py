
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "models/text/distilroberta_detector"

if not (MODEL_DIR / "config.json").exists():
    raise FileNotFoundError(
        "Fine-tuned model not found. Complete Step 16 first."
    )

print("Loading fine-tuned DistilRoBERTa...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

model.eval()

def predict_text(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    human_probability = probabilities[0].item() * 100
    ai_probability = probabilities[1].item() * 100

    prediction = "AI-Generated" if ai_probability >= 50 else "Human-Written"

    return {
        "prediction": prediction,
        "ai_probability": round(ai_probability, 2),
        "human_probability": round(human_probability, 2)
    }


if __name__ == "__main__":
    print("\nAI-Generated Text Detection")
    print("Type EXIT to quit.\n")

    while True:
        text = input("Enter text: ").strip()

        if text.lower() == "exit":
            break

        if not text:
            print("Please enter some text.\n")
            continue

        result = predict_text(text)

        print("\nPrediction:", result["prediction"])
        print("AI Probability:", result["ai_probability"], "%")
        print("Human Probability:", result["human_probability"], "%")
        print("-" * 40)
