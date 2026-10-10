
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[3]

ORIGINAL_MODEL = ROOT / "models/text/text_detector.pkl"
GROUPED_MODEL = ROOT / "models/text/text_detector_grouped.pkl"

def load_model(path):
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)

print("Loading trained models...")

original = load_model(ORIGINAL_MODEL)
grouped = load_model(GROUPED_MODEL)

# AI-generated paragraph for testing
sample_text = """
Artificial Intelligence (AI) is rapidly transforming the modern
world by enabling machines to perform tasks that traditionally
require human intelligence. From healthcare and education to
finance and transportation, AI-powered technologies are improving
efficiency, accuracy, and decision-making. Machine learning, a
subset of artificial intelligence, allows computer systems to
learn patterns from data and make predictions without being
explicitly programmed for every task. As AI continues to evolve,
it is essential to address challenges such as data privacy,
algorithmic bias, and ethical considerations to ensure that
these technologies benefit society responsibly.
"""

def evaluate(model, text):
    probabilities = model.predict_proba([text])[0]
    classes = list(model.classes_)

    ai_probability = float(probabilities[classes.index(1)])
    human_probability = float(probabilities[classes.index(0)])

    return {
        "prediction": (
            "AI-Generated"
            if ai_probability >= 0.5
            else "Human-Written"
        ),
        "ai_probability": round(ai_probability * 100, 2),
        "human_probability": round(human_probability * 100, 2)
    }

print("\n========== ORIGINAL MODEL ==========")
original_result = evaluate(original, sample_text)

for key, value in original_result.items():
    print(f"{key}: {value}")

print("\n========== GROUPED MODEL ==========")
grouped_result = evaluate(grouped, sample_text)

for key, value in grouped_result.items():
    print(f"{key}: {value}")

print("\n========== COMPARISON ==========")

if grouped_result["ai_probability"] > original_result["ai_probability"]:
    print("Grouped model assigns higher AI probability.")
elif grouped_result["ai_probability"] < original_result["ai_probability"]:
    print("Original model assigns higher AI probability.")
else:
    print("Both models assign the same AI probability.")

print("\nNote: A higher AI probability on one example does not")
print("prove that a model is more accurate overall.")
