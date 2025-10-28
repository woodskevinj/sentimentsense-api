from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="SentimentSense API")

class TextInput(BaseModel):
    test: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the SentimentSense API!"}

@app.post("/predict")
def predict_sentiment(input: TextInput):
    text = input.text
    # placeholder for prediction logic
    sentiment = "positive" if "good" in text.lower() else "negative"
    return {"sentiment": sentiment}

