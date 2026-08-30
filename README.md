# Fake News Detection System

A lightweight, interpretable Natural Language Processing (NLP) pipeline and web application to analyze news articles and detect linguistic misinformation patterns.

---

## Overview

This project implements a text-classification pipeline trained on a balanced benchmark dataset of 6,305 verified and fake news articles. It compares **Multinomial Naive Bayes**, **Logistic Regression**, and **Linear SVM**, optimizes hyperparameters via 3-Fold Stratified Cross-Validation, and serves predictions with calibrated confidence scores and explainable linguistic indicators through a Flask web interface.

---

## Model Benchmark Results

Evaluated on an untouched 20% test partition (1,261 articles) with a 25,000 unigram/bigram TF-IDF feature space:

| Model | Hyperparameters | Accuracy | Precision | Recall | F1-Score | Status |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Linear SVM** | `C=5.0, Calibrated (cv=3)` | **93.66%** | **93.66%** | **93.66%** | **0.9366** | **Production Winner** |
| **Logistic Regression** | `C=50.0, lbfgs` | 93.34% | 93.34% | 93.34% | 0.9334 | Strong Baseline |
| **Multinomial Naive Bayes** | `alpha=0.01` | 90.33% | 90.37% | 90.33% | 0.9032 | Fast Baseline |
| *DistilBERT (Optional)* | `distilbert-base-uncased` | *95.20%* | *95.25%* | *95.20%* | *0.9521* | Transformer Reference |

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/jachinsamuel/AI_Fake_News_Detector.git
cd AI_Fake_News_Detector
pip install -r requirements.txt
```

### 2. Download Data & Train Models

```bash
# 1. Download & clean the dataset
python data/download_or_prepare.py

# 2. Train and optimize all models
python src/train.py
```

### 3. Launch Web Application

```bash
python app.py
```

Open **`http://127.0.0.1:5000`** in your browser.

### 4. Run Test Suite

```bash
python -m unittest tests/test_pipeline.py
```

---

## Pipeline Architecture

1. **Preprocessing (`src/preprocessing.py`):**
   * Lowercasing, URL removal, HTML stripping, punctuation filtering.
   * Word tokenization, stopword removal (NLTK), and WordNet lemmatization.

2. **Feature Extraction:**
   * TF-IDF vectorizer (25,000 features, unigram + bigrams, sublinear term-frequency scaling).
   * **Zero Data Leakage:** Fitted strictly on the 80% training split.

3. **Classification & Calibration (`src/train.py`):**
   * Multi-model cross-validation with `GridSearchCV`.
   * Platt probability scaling (`CalibratedClassifierCV`) on Linear SVM to provide true confidence percentages.

4. **Explainable AI (`src/explain.py`):**
   * Extracts model hyperplane weights $\text{Score}(w) = \theta_w \cdot \text{TF-IDF}(w)$ to highlight top influential words that drove the classification.

---

## REST API Reference

### Predict Text Authenticity
* **Endpoint:** `POST /predict`
* **Headers:** `Content-Type: application/json`
* **Request:**
```json
{
  "text": "WASHINGTON (Reuters) - The Senate approved bipartisan infrastructure legislation on Thursday."
}
```
* **Response (200 OK):**
```json
{
  "prediction": "REAL",
  "confidence": 95.48,
  "model": "Linear SVM (Calibrated)",
  "important_features": ["republican", "leader", "relief", "senate", "thursday"],
  "explanation": "The article exhibits formal reporting syntax and structured journalistic terminology.",
  "processing_time_ms": 11.4
}
```

### Additional Endpoints
* `GET /api/metrics` — Returns cross-validation and test set metrics for all models.
* `GET /api/examples` — Returns pre-configured sample news articles for testing.
* `GET /api/health` — Service readiness status.

---

## Project Structure

```
FakeNews/
├── data/
│   ├── download_or_prepare.py    # Dataset acquisition & EDA
│   └── news.csv                  # 6,305 labeled articles
├── models/
│   ├── best_model.pkl            # Serialized winning model
│   ├── vectorizer.pkl            # Fitted TF-IDF vectorizer
│   └── model_metadata.json       # Training metadata & scores
├── notebooks/
│   └── experiments.ipynb         # Interactive Jupyter research notebook
├── results/
│   ├── model_comparison.csv      # CSV metrics table
│   └── model_comparison.png      # Evaluation bar chart
├── src/
│   ├── preprocessing.py          # NLP text cleaning pipeline
│   ├── train.py                  # Training & GridSearchCV optimization
│   ├── evaluate.py               # Metric calculation & visualization
│   ├── explain.py                # XAI feature importance extraction
│   ├── predict.py                # Standalone inference class
│   └── distilbert_train.py       # Optional Transformer fine-tuning
├── templates/
│   └── index.html                # Clean web UI
├── static/
│   ├── style.css                 # Editorial CSS stylesheet
│   └── script.js                 # Client-side controller
├── tests/
│   └── test_pipeline.py          # Unit & integration tests
├── app.py                        # Flask server & REST API
└── requirements.txt              # Dependencies
```

---

## Limitations

* **Linguistic Classifier:** This system classifies stylistic and vocabulary patterns learned from training data. It does not verify real-world facts against live external news databases.
* **Adversarial Text:** Heavily edited or nuanced satire may occasionally bypass purely lexical filters.
