# AI Fake News Detection & Live Fact-Checking System

An end-to-end Machine Learning and Natural Language Processing (NLP) system combined with a real-time **AI Live Web Verification Agent** to detect misinformation, classify journalistic style, and cross-reference claims against live news wires and independent fact-checkers.

---

## Key Features

* **Multi-Domain Dataset (12,736 Articles):** Consolidated and balanced across McIntire News, FakeNewsNet PolitiFact, and FakeNewsNet GossipCop datasets (covering Politics, Science, Health, Entertainment, and World News).
* **Multi-Model ML Pipeline:** Compares **Linear SVM (Platt Calibrated)**, **Logistic Regression**, and **Multinomial Naive Bayes** with 3-Fold Stratified `GridSearchCV` hyperparameter tuning.
* **Hybrid AI Live Web Verification:** Extracts claims in real-time and queries **NewsAPI**, **Google Fact Check Tools API**, **GNews API**, and **Google News RSS** to verify coverage and fact-checking records.
* **Explainable AI (XAI):** Identifies top influential unigrams/bigrams and outputs plain-English reasoning for each prediction.
* **Modern Web Interface:** Fast, responsive UI with live word/char counters, confidence gauge, detected vocabulary badges, and clickable live source citations.

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/jachinsamuel/AI_Fake_News_Detector.git
cd AI_Fake_News_Detector
pip install -r requirements.txt
```

### 2. Configure API Keys (Optional)

The system works out-of-the-box using the built-in zero-key live search fallback. To add your own keys, edit `.env`:

```ini
# .env
NEWS_API_KEY=your_newsapi_key
GOOGLE_FACTCHECK_API_KEY=your_google_api_key
GNEWS_API_KEY=your_gnews_key
```

### 3. Prepare Dataset & Train Models

```bash
# 1. Acquire and consolidate datasets (12,736 balanced records)
python data/download_or_prepare.py

# 2. Preprocess, train, and optimize ML models
python src/train.py
```

### 4. Launch Web Application

```bash
python app.py
```

Open **`http://127.0.0.1:5000`** in your browser.

### 5. Run Automated Tests

```bash
python -m unittest tests/test_pipeline.py
```

---

## Pipeline Architecture

```
User Input (Headline / Article)
  │
  ├──► [NLP Preprocessing] ──► [TF-IDF Vectorizer (25,000 features)] ──► [Classifier (Logistic Regression / SVM)]
  │                                                                            │
  └──► [Live Web Verifier Agent] ──► [Google Fact Check / NewsAPI / RSS] ──────┤
                                                                               ▼
                                                            [Hybrid Consensus Decision Engine]
                                                                               │
                                                                               ▼
                                                            Verdict + Confidence + Live Citations
```

1. **Text Preprocessing (`src/preprocessing.py`):**
   * HTML/URL stripping, punctuation cleaning, lowercasing.
   * NLTK tokenization, stopword removal, and WordNet lemmatization.

2. **Feature Extraction:**
   * 25,000 unigram/bigram features with sublinear term-frequency scaling.
   * Strictly fitted on the training split to prevent data leakage.

3. **Classification & Probability Calibration (`src/train.py`):**
   * Optimized with 3-Fold Stratified `GridSearchCV`.
   * Calibrated probability scoring for accurate confidence measurements.

4. **Live Web Verification (`src/web_verifier.py`):**
   * Automatically extracts search queries from input claims.
   * Queries Google Fact Check database (Snopes, PolitiFact, Reuters Fact Check) and live news feeds.
   * Synthesizes web evidence into the final authenticity confidence score.

---

## REST API Reference

### Predict Authenticity
* **Endpoint:** `POST /predict`
* **Headers:** `Content-Type: application/json`
* **Request:**
```json
{
  "text": "NASA James Webb Space Telescope discovers distant galaxy at cosmic dawn.",
  "check_web": true
}
```
* **Response (200 OK):**
```json
{
  "status": "success",
  "prediction": "REAL",
  "confidence": 94.0,
  "model": "Logistic Regression",
  "important_features": ["telescope", "galaxy", "nasa", "space", "distant"],
  "explanation": "Corroborating reporting found across 4 active web news articles.",
  "web_verification": {
    "web_verdict": "MATCHING_NEWS_FOUND",
    "sources_count": 4,
    "live_sources": [
      {
        "title": "NASA's Roman Space Telescope launches to map billions of galaxies",
        "source": "The Brighter Side of News",
        "url": "https://..."
      }
    ]
  },
  "processing_time_ms": 520.4
}
```

### Additional Endpoints
* `GET /api/config` — Returns active API services and server health.
* `GET /api/metrics` — Returns cross-validation and test set metrics.
* `GET /api/examples` — Returns pre-configured sample news articles.
* `GET /api/health` — System health check.

---

## Project Structure

```
FakeNews/
├── .env.example              # Example environment configuration
├── .gitignore                # Git exclusions (protects secrets & caches)
├── app.py                    # Flask server & REST API
├── requirements.txt          # Python dependencies
├── data/
│   ├── download_or_prepare.py# Multi-dataset consolidation & EDA
│   └── news.csv              # 12,736 labeled articles
├── models/
│   ├── best_model.pkl        # Serialized production model
│   ├── vectorizer.pkl        # Fitted TF-IDF vectorizer
│   ├── label_encoder.pkl     # Label mapping
│   └── model_metadata.json   # Training scores & parameters
├── results/
│   ├── model_comparison.csv  # Benchmark metrics table
│   ├── model_comparison.png  # Performance comparison chart
│   ├── confusion_matrices.png# Confusion matrices
│   └── eda_distribution.png  # Dataset exploratory analysis
├── src/
│   ├── config.py             # Environment & API key loader
│   ├── preprocessing.py      # NLP text cleaning pipeline
│   ├── train.py              # ML training & GridSearchCV
│   ├── evaluate.py           # Evaluation metrics & charts
│   ├── explain.py            # Feature importance & XAI
│   ├── predict.py            # Inference & Hybrid decision synthesis
│   ├── web_verifier.py       # AI Live Web Verification agent
│   └── distilbert_train.py   # Optional transformer fine-tuning
├── static/
│   ├── style.css             # Minimalist styling
│   └── script.js             # Client-side controller
├── templates/
│   └── index.html            # Web application interface
└── tests/
    └── test_pipeline.py      # 17 Automated unit & integration tests
```

---

## License

Open source under the [MIT License](LICENSE).
