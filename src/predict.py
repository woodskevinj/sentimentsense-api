# This module loads the saved model and provides a simple function for prediction.
# will be imported by app.py

import joblib
from src.preprocess import clean_text

# Load model and vectorizer once at startup
vectorizer, model = joblib.load("models/sentiment_model.pkl")

def predict_sentiment(text:str) -> str:
    """
    Predicts sentiment (positive, negative, neutral) for a given text input.
    Ensures return value is a standard Python string.
    Includes a simple rule-based override to improve small-dataset results.
    """
    cleaned_text = clean_text(text)

    # --- Rule-based sentiment hints ---
    positive_words = ["love", "great", "fantastic", "amazing", "happy", "best", "wonderful", "awesome", "thrilled"]
    negative_words = ["hate", "terrible", "awful", "worst", "disappointed", "horrible", "bad", "useless"]

    if any(word in cleaned_text for word in positive_words):
        return "positive"
    if any(word in cleaned_text for word in negative_words):
        return "negative"
    
    # --- Fallback to ML model prediction ---
    vectorized = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized)[0]
    return str(prediction)
