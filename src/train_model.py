# This script trains your sentiment model (TF-IDF + Logistic Regression) and saves it as sentiment_model.pkl.

# Will use a small sample dataset (you can replace it later with IMDb or Kaggle data).

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from preprocess import clean_text
import os

# Optional: sample data
SAMPLE_DATA = {
    "text": [
        "I love this product, it works great!",
        "This is terrible. I hate it.",
        "Amazing experience, will buy again.",
        "Worst purchase ever, waste of money.",
        "It's okay, nothing special.",
        "I am very happy with the service.",
        "Really disappointed by the quality.",
        "Absolutely fantastic job, I'm thrilled!",
        "I am not happy with this at all.",
        "This is the best thing I've ever used.",
        "Horrible experience, will not return.",
        "Mediocre results, it's fine I guess.",
        "Everything about this was perfect!",
        "Awful, completely useless.",
        "Not bad, could be better though."
    ],
    "label": [
        "positive", "negative", "positive", "negative", "neutral", "positive", "negative",
        "positive", "negative", "positive", "negative", "neutral", "positive", "negative", "neutral"
    ]
}

def train_and_save_model():
    print("📘 Loading and cleaning data...")
    df = pd.DataFrame(SAMPLE_DATA)
    df["clean_text"] = df["text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42
    )

    print("🔠 Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("🤖 Training model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump((vectorizer, model), "models/sentiment_model.pkl")

    print("\n✅ Model training complete. Saved to 'models/sentiment_model.pkl'")

if __name__ == "__main__":
    train_and_save_model()