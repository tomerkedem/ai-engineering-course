from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise RuntimeError("HF_TOKEN is missing. Please define it in your .env file.")

print("HF_TOKEN loaded:", bool(hf_token))


# Load model and tokenizer
model_name = "tabularisai/robust-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Function to predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text.lower(), return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][predicted_class].item()

    
    sentiment_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
    
    return {
        "label": sentiment_map[predicted_class],
        "confidence": confidence
    }


# Example usage
texts = [
    "I absolutely loved this movie! The acting was superb and the plot was engaging.",
    "The service at this restaurant was terrible. I'll never go back.",
    "The product works as expected. Nothing special, but it gets the job done.",
    "I'm somewhat disappointed with my purchase. It's not as good as I hoped.",
    "This book changed my life! I couldn't put it down and learned so much."
]

for text in texts:
    result = predict_sentiment(text)

    print(f"Text: {text}")
    print(f"Sentiment: {result['label']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print()
