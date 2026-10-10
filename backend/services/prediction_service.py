
from pathlib import Path
from functools import lru_cache
import re

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "text" / "distilroberta_v2"

MAX_LENGTH = 128
MIN_WORDS = 50
MAX_EXPLAIN_WORDS = 80
EXPLANATION_BATCH_SIZE = 8


@lru_cache(maxsize=1)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )
    model.eval()

    return tokenizer, model


def get_ai_score(model, input_ids, attention_mask):
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        probabilities = torch.softmax(
            outputs.logits, dim=-1
        )

    # Training labels: 0 = Human, 1 = AI
    return probabilities[:, 1]


def explain_prediction(text, tokenizer, model, inputs, base_score):
    """
    Explain model sensitivity using word-level token occlusion.

    A positive impact means masking the word decreased the
    model's AI score, suggesting the word supported AI_LIKE.

    A negative impact means masking the word increased the
    model's AI score, suggesting the word supported HUMAN_LIKE.
    """

    offsets = inputs["offset_mapping"][0].tolist()
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    if tokenizer.mask_token_id is None:
        return []

    word_matches = list(re.finditer(r"\b[\w'-]+\b", text))
    candidates = []

    for match in word_matches:
        start, end = match.span()

        token_positions = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_end > token_start
            and token_start < end
            and token_end > start
        ]

        if not token_positions:
            continue

        candidates.append({
            "word": match.group(),
            "start": start,
            "end": end,
            "positions": token_positions,
        })

        if len(candidates) >= MAX_EXPLAIN_WORDS:
            break

    explanations = []

    for batch_start in range(
        0, len(candidates), EXPLANATION_BATCH_SIZE
    ):
        batch = candidates[
            batch_start:batch_start + EXPLANATION_BATCH_SIZE
        ]

        batch_ids = input_ids.repeat(len(batch), 1)
        batch_masks = attention_mask.repeat(len(batch), 1)

        for row, candidate in enumerate(batch):
            for position in candidate["positions"]:
                batch_ids[row, position] = tokenizer.mask_token_id

        scores = get_ai_score(
            model,
            batch_ids,
            batch_masks,
        ).tolist()

        for candidate, masked_score in zip(batch, scores):
            impact = base_score - masked_score

            explanations.append({
                "word": candidate["word"],
                "start": candidate["start"],
                "end": candidate["end"],
                "impact": round(impact, 5),
                "direction": (
                    "AI"
                    if impact > 0
                    else "HUMAN"
                    if impact < 0
                    else "NEUTRAL"
                ),
            })

    explanations.sort(
        key=lambda item: abs(item["impact"]),
        reverse=True,
    )

    return explanations


def analyze_text(text: str):
    text = text.strip()
    word_count = len(text.split())

    if word_count < MIN_WORDS:
        return {
            "prediction": "INSUFFICIENT_TEXT",
            "ai_score": None,
            "word_count": word_count,
            "token_count": None,
            "truncated": False,
            "model": "DistilRoBERTa_V2",
            "explanation": {
                "summary": "Please enter at least 50 words.",
                "highlights": [],
            },
            "warning": "Insufficient text for analysis.",
        }

    tokenizer, model = load_model()

    full_token_count = len(
        tokenizer.encode(
            text,
            truncation=False,
        )
    )

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
    )

    offsets = encoded.pop("offset_mapping")

    score = get_ai_score(
        model,
        encoded["input_ids"],
        encoded["attention_mask"],
    )[0].item()

    if score >= 0.80:
        prediction = "AI_LIKE"
        summary = (
            "The model assigned a high score to the AI class. "
            "The highlighted words show where masking the text "
            "changed this score. This is not proof of AI authorship."
        )
    elif score <= 0.20:
        prediction = "HUMAN_LIKE"
        summary = (
            "The model assigned a low score to the AI class. "
            "The highlighted words show which parts influenced "
            "its assessment. This is not proof of human authorship."
        )
    else:
        prediction = "UNCERTAIN"
        summary = (
            "The model's output falls within the uncertain range. "
            "The highlighted words indicate model sensitivity, "
            "but the result should not be treated as conclusive."
        )

    explanation_inputs = {
        **encoded,
        "offset_mapping": offsets,
    }

    highlights = explain_prediction(
        text,
        tokenizer,
        model,
        explanation_inputs,
        score,
    )

    return {
        "prediction": prediction,
        "ai_score": round(score, 4),
        "word_count": word_count,
        "token_count": full_token_count,
        "truncated": full_token_count > MAX_LENGTH,
        "model": "DistilRoBERTa_V2",
        "explanation": {
            "method": "word_level_token_occlusion",
            "summary": summary,
            "highlights": highlights[:12],
            "evaluated_words": len(highlights),
        },
        "warning": (
            "Experimental model. Raw scores are not calibrated "
            "probabilities. Explanations show model sensitivity, "
            "not verified authorship. Independent real-world "
            "generalization remains unverified."
        ),
    }
