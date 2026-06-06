from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise RuntimeError("HF_TOKEN is missing. Please define it in your .env file.")

print("HF_TOKEN loaded:", bool(hf_token))

MODEL_NAME = "ProsusAI/finbert"
CONFIDENCE_THRESHOLD = 0.92

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

LABELS = {
    0: "positive",
    1: "negative",
    2: "neutral",
}


def analyze_financial_sentiment(text: str) -> dict:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][predicted_class].item()

    return {
        "sentiment": LABELS[predicted_class],
        "confidence": confidence
    }


def decide_trade_action(sentiment: str, confidence: float) -> str:
    if sentiment == "positive" and confidence > CONFIDENCE_THRESHOLD:
        return "BUY"

    if sentiment == "negative" and confidence > CONFIDENCE_THRESHOLD:
        return "SELL"

    return "HOLD"


def dummy_buy_api(sentence: str, confidence: float) -> dict:
    return {
        "api": "BUY",
        "status": "success",
        "message": "Dummy BUY order was created.",
        "confidence": confidence,
        "sentence": sentence
    }


def dummy_sell_api(sentence: str, confidence: float) -> dict:
    return {
        "api": "SELL",
        "status": "success",
        "message": "Dummy SELL order was created.",
        "confidence": confidence,
        "sentence": sentence
    }


def trade_from_sentence(sentence: str) -> dict:
    analysis = analyze_financial_sentiment(sentence)

    sentiment = analysis["sentiment"]
    confidence = analysis["confidence"]

    action = decide_trade_action(sentiment, confidence)

    if action == "BUY":
        api_result = dummy_buy_api(sentence, confidence)
    elif action == "SELL":
        api_result = dummy_sell_api(sentence, confidence)
    else:
        api_result = {
            "api": "HOLD",
            "status": "skipped",
            "message": "No dummy trade was created.",
            "confidence": confidence,
            "sentence": sentence
        }

    return {
        "sentence": sentence,
        "sentiment": sentiment,
        "confidence": confidence,
        "action": action,
        "api_result": api_result
    }


def run_examples() -> None:
    examples = [
        "The company reported record quarterly earnings and raised its full-year guidance.",
        "Shares plunged after the CEO resigned amid an accounting investigation.",
        "The board approved a dividend unchanged from the prior quarter.",
    ]

    for sentence in examples:
        result = trade_from_sentence(sentence)

        print("Sentence:")
        print(result["sentence"])

        print("Sentiment:", result["sentiment"])
        print(f"Confidence: {result['confidence']:.2%}")
        print("Action:", result["action"])
        print("API Result:", result["api_result"]["message"])
        print("-" * 80)


if __name__ == "__main__":
    run_examples()