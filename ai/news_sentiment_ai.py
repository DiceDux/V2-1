# news_sentiment_ai.py

from transformers import pipeline
import re

# مدل HuggingFace برای تحلیل احساس
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def clean_text(text):
    # حذف کاراکترهای اضافی
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", "", text)
    return text.strip()

def analyze_sentiment(text):
    cleaned = clean_text(text)
    result = sentiment_pipeline(cleaned[:512])[0]
    label = result["label"]
    score = result["score"]

    if label == "POSITIVE":
        return round(score, 2)
    elif label == "NEGATIVE":
        return round(-score, 2)
    else:
        return 0.0
