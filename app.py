import os
import logging
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from src.predict import predict_sentiment, vectorizer, model


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
    
# ======================================================
# Health Check Endpoint — for ECS / Monitoring
# ======================================================
@app.get("/health")
async def health_check():
    """
    Verifies that the API, model and vectorizer are load properly.
    Useful for ECS or uptime monitoring
    """
    try:
        # Basic in-memory test
        test_test = "system test"
        X_test = vectorizer.transform([test_test])
        _=model.predict(X_test)

        model_path = os.path.exists("models/sentiment_model.pkl")

        return {
            "status": "healthy",
            "model_loaded": True,
            "model_file_exists": model_path,
            "vectorizer_features": getattr(vectorizer, "max_features", None),
            "message": "API and model are ready for inference."
        }
    
    except Exception as e:
        return {
            "status": "unhealth",
            "model_loaded": False,
            "error": str(e)
        }
    
# ======================================================
# Model Info Endpoint — Metadata for Debugging & Versioning
# ======================================================
@app.get("/info")
async def model_info():
    """
    Returns key metadata about the currently loaded sentiment model.
    Useful for debugging, version tracking and documentation
    """
    try:
        model_path = "models/sentiment_model.pkl"

        # Model file stats
        if os.path.exists(model_path):
            modified_time = datetime.fromtimestamp(os.path.getmtime(model_path))
            file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        else:
            modified_time = None
            file_size_mb = None
        
        # Vectorizer stats
        vocab_size = len(vectorizer.vocabulary_) if hasattr(vectorizer, "vocabulary_") else None

        return {
            "model_name": "Logistic Regression (TF-IDF)",
            "framework": "scikit-learn",
            "vectorizer_type": type(vectorizer).__name__,
            "vocabulary_size": vocab_size,
            "model_size": os.path.basename(model_path),
            "model_size_mb": round(file_size_mb, 2) if file_size_mb else None,
            "last_modified": str(modified_time) if modified_time else "unknown",
            "status": "ready"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model info: {e}")