from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict_sentiment

app = FastAPI(title="SentimentSense API")

class TextInput(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the SentimentSense API!"}

@app.post("/predict")
def predict(input: TextInput):
    text = input.text
    # placeholder for prediction logic
    sentiment = predict_sentiment(input.text)
    return {"sentiment": sentiment}

