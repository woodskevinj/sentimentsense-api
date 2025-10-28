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
└── .gitignore

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

- [ ] Add logging, /logs, /health, and /info endpoints

- [ ] Containerize with Docker

- [ ] Deploy to AWS ECS

---

👨‍💻 Author

- Kevin Woods
- Applied ML Engineer | AWS Certified AI Practitioner | AWS Machine Learning Certified Engineer – Associate
- 🔗 GitHub: woodskevinj
