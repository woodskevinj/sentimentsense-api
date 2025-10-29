import os
import logging
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.predict import predict_sentiment


app = FastAPI(title="SentimentSense API")

# ======================================================
# Logging Setup (add near the top of api/app.py)
# ======================================================

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "sentiments.log")

# Rotate when file reaches 1MB, keep 3 backups
handler = RotatingFileHandler(
    LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8")

formatter = logging.Formatter("%(asctime)s | %(message)s")
handler.setFormatter(formatter)

logger = logging.getLogger("sentiment_logger")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False # prevents duplicate logs if Uvicorn also logs

logger.info("🚀 SentimentSense API started (rotating logs active)")

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

    # 🔹 Log each prediction
    logger.info(f"Text:  {text} | Sentiment: {sentiment}")

    return {"sentiment": sentiment}

# ======================================================
# Logs Endpoint — View Recent Predictions
# ======================================================
@app.get("/logs")
async def get_logs(limit: int = 10):
    """
    Return the last 'limit' lines from the prediction log file.
    Example: GET /logs?limit=5
    """
    try:
        if not os.path.exists(LOG_FILE):
            return {"message": "No logs found yet."}
        
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()

        # Return the last N lines (most recent predictions)
        recent_logs = [line.strip() for line in lines[-limit:]]
        return {"recent_sentiments": recent_logs}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading logs: {e}")