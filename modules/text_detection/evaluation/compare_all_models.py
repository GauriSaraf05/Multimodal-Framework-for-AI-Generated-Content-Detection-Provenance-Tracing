
from pathlib import Path
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[3]

ORIGINAL_MODEL = ROOT / "models/text/text_detector.pkl"
GROUPED_MODEL = ROOT / "models/text/text_detector_grouped.pkl"
TRANSFORMER_DIR = ROOT / "models/text/distilroberta_detector"

# Check saved models exist
for path in [ORIGINAL_MODEL, GROUPED_MODEL]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing model: {path}")

if not (TRANSFORMER_DIR / "config.json").is_file():
    raise FileNotFoundError(
        "Fine-tuned DistilRoBERTa not found. Complete Step 16."
    )

print("Loading all three models...")

original_model = joblib.load(ORIGINAL_MODEL)
grouped_model = joblib.load(GROUPED_MODEL)

tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_DIR)
transformer = AutoModelForSequenceClassification.from_pretrained(
    TRANSFORMER_DIR
)
transformer.eval()


def predict_baseline(model, text):
    probabilities = model.predict_proba([text])[0]
    classes = list(model.classes_)

    ai_probability = float(probabilities[classes.index(1)])

    return {
        "prediction": "AI" if ai_probability >= 0.5 else "HUMAN",
        "ai_probability": round(ai_probability * 100, 2)
    }


def predict_transformer(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = transformer(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    ai_probability = probabilities[1].item()

    return {
        "prediction": "AI" if ai_probability >= 0.5 else "HUMAN",
        "ai_probability": round(ai_probability * 100, 2)
    }


def compare(text):
    results = {
        "Original TF-IDF": predict_baseline(original_model, text),
        "Grouped TF-IDF": predict_baseline(grouped_model, text),
        "DistilRoBERTa": predict_transformer(text)
    }

    print("\n" + "=" * 65)
    print("MODEL COMPARISON")
    print("=" * 65)

    print(f"{'Model':<22} {'Prediction':<15} {'AI Score':>12}")
    print("-" * 65)

    for name, result in results.items():
        print(
            f"{name:<22} "
            f"{result['prediction']:<15} "
            f"{result['ai_probability']:>11.2f}%"
        )

    predictions = [r["prediction"] for r in results.values()]

    if len(set(predictions)) == 1:
        print("\nAll three models agree.")
    else:
        print("\nModels disagree. Further evaluation is needed.")

    print("\nNote: AI scores are not calibrated probabilities")
    print("or proof of authorship.")


if __name__ == "__main__":
    print("\nAI Content Detection - Three Model Comparison")
    print("Enter EXIT to quit.")

    while True:
        text = input("\nEnter text: ").strip()

        if text.lower() == "exit":
            break

        if not text:
            print("Please enter text.")
            continue

        compare(text)
