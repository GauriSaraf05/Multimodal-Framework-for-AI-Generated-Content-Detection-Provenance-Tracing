
from pathlib import Path
import joblib

# Project root
ROOT = Path(__file__).resolve().parents[3]

# Trained model location
MODEL_PATH = ROOT / "models" / "text" / "text_detector.pkl"

# Load trained model
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Trained model not found. Run train_baseline.py first."
    )

model = joblib.load(MODEL_PATH)


def detect_text(text):
    # Validate input
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please enter valid text.")

    # Predict probabilities
    probabilities = model.predict_proba([text])[0]

    # Match probabilities to correct class labels
    classes = list(model.classes_)

    ai_probability = float(
        probabilities[classes.index(1)]
    )

    human_probability = float(
        probabilities[classes.index(0)]
    )

    # Determine predicted class
    prediction = (
        "AI-Generated"
        if ai_probability >= 0.5
        else "Human-Written"
    )

    return {
        "prediction": prediction,
        "ai_probability": round(ai_probability * 100, 2),
        "human_probability": round(human_probability * 100, 2)
    }


if __name__ == "__main__":
    print("=" * 45)
    print("AI-GENERATED TEXT DETECTION SYSTEM")
    print("=" * 45)

    while True:
        print("\nEnter text for detection.")
        print("Type 'exit' to stop.\n")

        text = input("Text: ")

        if text.lower().strip() == "exit":
            print("Exiting...")
            break

        try:
            result = detect_text(text)

            print("\n========== DETECTION RESULT ==========")
            print("Prediction:", result["prediction"])
            print("AI Probability:", result["ai_probability"], "%")
            print("Human Probability:", result["human_probability"], "%")
            print("======================================")

        except ValueError as error:
            print("Error:", error)
