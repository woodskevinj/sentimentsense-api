# 💬 SentimentSense API – Real-Time Sentiment Analysis

A lightweight **FastAPI-based sentiment analysis microservice** that classifies text as Positive, Negative, or Neutral using a hybrid NLP approach — combining a rule-based sentiment layer with a trained Logistic Regression model.

---

## 🧩 Project Overview

SentimentSense API enables developers and businesses to analyze customer feedback, reviews, or social media text instantly through a RESTful endpoint.

The system includes:

- Text preprocessing (tokenization, stopword removal, lemmatization)
- Hybrid sentiment logic (keyword rules + trained model)
- Real-time inference via FastAPI
- Optional Docker and AWS ECS deployment

---

## 🧠 Tech Stack

| Component     | Technology                  |
| ------------- | --------------------------- |
| Language      | Python 3.9+                 |
| Framework     | FastAPI                     |
| NLP           | scikit-learn + NLTK         |
| Model Serving | joblib + FastAPI            |
| Deployment    | Docker / AWS ECS (optional) |

---

## ⚙️ Quick Start

```bash
# 1️⃣ Create and activate venv
python -m venv venv
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt

# 2️⃣ Train model
python src/train_model.py

# 3️⃣ Run API
uvicorn app:app --reload
```

## 🧪 Example Predictions

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "I absolutely love this project!"}'
```

Expected response:

```json
{ "sentiment": "positive" }
```

---

## 🔗 API Endpoints

### **1️⃣ `/predict` — Sentiment Inference**

**Method:** `POST`
**Description:** Classifies input text as Positive, Negative, or Neutral using the hybrid sentiment model.

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "I absolutely love this project!"}'
```

- Response:

```json
{ "sentiment": "positive" }
```

### **2️⃣ `/logs` — View Recent Predictions**

**Method:** `GET`
**Query Param:** limit (optional, default=10)
**Description:** Returns the most recent logged predictions from logs/sentiments.log.

```bash
curl "http://127.0.0.1:8000/logs?limit=5"
```

- Response:

```json
{
  "recent_sentiments": [
    "2025-10-28 22:10:43 | Text: I love this project! | Sentiment: positive",
    "2025-10-28 22:12:17 | Text: This is awful | Sentiment: negative"
  ]
}
```

### **3️⃣ `/health` — API + Model Readiness**

**Method:** `GET`
**Description:** Confirms that the API, model, and vectorizer are loaded correctly — useful for ECS or uptime monitoring.

```bash
curl http://127.0.0.1:8000/health
```

- Response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_file_exists": true,
  "vectorizer_features": 5000,
  "message": "API and model are ready for inference."
}
```

### **4️⃣ `/info` — Model Metadata & Version Info**

**Method:** `GET`  
**Description:** Returns key metadata about the currently loaded model, including file size, modification date, and vectorizer stats.

```bash
curl http://127.0.0.1:8000/info
```

---

## 📂 Project Structure

```css
sentimentsense-api/
├── app.py
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── predict.py
│   └── train_model.py
├── models/
│   └── sentiment_model.pkl
├── data/
│   └── sample_texts.csv
├── requirements.txt
├── README.md
├── Dockerfile
├── .dockerignore
└── .gitignore

```

---

## 📜 Logging and Monitoring

Every prediction request is automatically logged to rotating files under `/logs/`.

Each log entry records:

```bash
YYYY-MM-DD HH:MM:SS | Text: <input> | Sentiment: <predicted_class>
```

Log rotation details:

- File: `logs/sentiments.log`
- Max file size: 1 MB
- Up to 3 backups are maintained (`.1`, `.2`, `.3`)

You can also view the latest predictions via API:

```bash
curl "http://127.0.0.1:8000/logs?limit=5"
```

---

## 🚀 Current Progress

✅ Completed:

- Model training and hybrid prediction logic

- Working FastAPI endpoint /predict

- Successful local testing via curl

🛠️ Next Steps:

- Add /health route with metadata

- Train on larger dataset (IMDb or Sentiment140)

- Dockerize and deploy via AWS ECS

---

## 📅 Roadmap

- [x] Project setup

- [x] Model training

- [x] API endpoint for predictions

- [x] Add logging, /logs, /health, and /info endpoints

- [ ] Containerize with Docker

- [ ] Deploy to AWS ECS

---

👨‍💻 Author

- Kevin Woods
- Applied ML Engineer | AWS Certified AI Practitioner | AWS Machine Learning Certified Engineer – Associate
- 🔗 GitHub: woodskevinj
