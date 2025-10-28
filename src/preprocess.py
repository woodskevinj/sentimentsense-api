# This module handles text cleaning and preprocessing before vectorization.
# Will use NLTK for tokenization, stopword removal, and lemmatization.

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure necessary resources are downloaded
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

#Initialize
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - Lowercsing
    - Removing punctuation, numbers ad extra spaces
    - Revomving stopwords
    - Lemmatization
    """
    # Lowercase and remove non-alphabetic chars
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())

    # Tokenize
    tokens = text.split()

    # Remove stopwords and lemmatize
    cleaned_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(cleaned_tokens)

