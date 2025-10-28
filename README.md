# 💬 SentimentSense API – Real-Time Sentiment Analysis

A lightweight **FastAPI-based sentiment analysis microservice** that classifies text as Positive, Negative, or Neutral using a trained NLP model.  
This project demonstrates practical NLP deployment workflows for applied machine learning engineers.

---

## 🧩 Project Overview

SentimentSense API enables developers and businesses to analyze customer feedback, reviews, or social media text instantly through a RESTful endpoint.

The system includes:

- Text preprocessing (tokenization, stopword removal, etc.)
- Model training (Scikit-learn or Hugging Face transformers)
- API endpoint for real-time sentiment prediction
- Optional Docker containerization and AWS deployment

---

## 🧠 Tech Stack

| Component     | Technology                  |
| ------------- | --------------------------- |
| Language      | Python 3.9+                 |
| Framework     | FastAPI                     |
| NLP           | Scikit-learn / Transformers |
| Model Serving | Pickle + FastAPI            |
| Deployment    | Docker / AWS ECS (optional) |

---

## ⚙️ Quick Start

```bash
# 1️⃣ Create and activate venv
python -m venv venv
source venv/bin/activate  # (Mac)
pip install -r requirements.txt

# 2️⃣ Run locally
uvicorn app:app --reload

# 3️⃣ Test endpoint
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"text": "I love this API!"}'

```

---

Expected response:

```json
{ "sentiment": "positive" }
```

## 📂 Project Structure

```css
sentimentsense-api/
├── app.py
├── src/
│   ├── preprocess.py
│   ├── predict.py
│   └── train_model.py
├── models/
│   └── sentiment_model.pkl
├── data/
│   └── sample_texts.csv
├── requirements.txt
├── README.md
└── .gitignore

```

---

## 📅 Roadmap

- [x] Project setup

- [ ] Model training

- [ ] API endpoint for predictions

- [ ] Add logging, /logs, /health, and /info endpoints

- [ ] Containerize with Docker

- [ ] Deploy to AWS ECS

---

👨‍💻 Author

- Kevin Woods
- Applied ML Engineer | AWS Certified AI Practitioner | Machine Learning Certified Engineer – Associate
- 🔗 GitHub: woodskevinj
