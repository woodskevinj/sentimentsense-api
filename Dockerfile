# ======================================================
# 🐳 SentimentSense API - Dockerfile
# ------------------------------------------------------
# Description : Builds the containerized FastAPI microservice
# Maintainer  : Kevin Woods (Applied ML Engineer)
# Base Image  : python:3.10-slim
# Build Date  : 2025-10-28
# Version     : 1.0.0
# ------------------------------------------------------
# Usage:
#   docker build -t sentimentsense-api .
#   docker run -p 8000:8000 sentimentsense-api
# ======================================================


# ======================================================
# Stage 1: Build Environment
# ======================================================
FROM python:3.10-slim AS builder

# Prevents Python from writing .pyc files and enables unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# Install system dependencies (for nltk, scikit-learn, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
 && rm -rf /var/lib/apt/lists/*

# Copy dependency file first for caching
COPY requirements.txt .

# Install dependencies in builder layer
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt


# ======================================================
# Stage 2: Runtime Environment
# ======================================================
FROM python:3.10-slim

WORKDIR /app

# Copy only necessary files from builder image
COPY --from=builder /usr/local /usr/local

# Copy application code
COPY . .

# Create logs directory (so FastAPI can write to it)
RUN mkdir -p logs

# Expose FastAPI port
EXPOSE 8000

# Default command to run API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
