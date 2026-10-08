# AI Fake News Detection & Live Fact-Checking System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask%203.0%2B-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Tests-38%20Passing-brightgreen.svg)](tests/test_pipeline.py)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning and Natural Language Processing (NLP) intelligence platform coupled with an active **AI Live Web Verification Agent**. The system classifies journalistic patterns, extracts text from screenshots or voice dictation, cross-references claims against global news wires and independent fact-checkers in real time, and synthesizes hybrid consensus verdicts with plain-English explanations.

---

## Architecture Overview

Traditional fake news classifiers rely solely on stylistic patterns in text and cannot evaluate whether an event actually occurred in the real world. This system implements a **5-Layer Hybrid Decision Architecture** that bridges static NLP feature classification with dynamic live web corroboration:

```
                            [ User Input ]
               (Direct Text | Article URL | Image OCR | Voice)
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│     Layer 1: NLP Preprocessing  │       │   Layer 2: Live Web Verifier    │
│  - HTML / URL Stripping         │       │  - Google Fact Check API        │
│  - Regex Normalization          │       │  - NewsAPI & GNews API          │
│  - Lemmatization & Stopwords    │       │  - Google News RSS Fallback     │
│  - TF-IDF (25,000 N-Grams)      │       │  - Wikipedia REST Grounding     │
└────────────────┬────────────────┘       └────────────────┬────────────────┘
                 │                                         │
                 ▼                                         ▼
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│  Layer 3: Ensemble ML Classifiers│      │ Layer 4: Hoax & Attack Filter   │
│  - Linear SVM (Platt Calibrated)│       │  - Predicate-Action Matching    │
│  - Logistic Regression          │       │  - Uncorroborated Warfare Check │
│  - Multinomial Naive Bayes      │       │  - 130+ Domain Trust Database   │
└────────────────┬────────────────┘       └────────────────┬────────────────┘
                 │                                         │
                 └──────────────────┬──────────────────────┘
                                    ▼
                 ┌─────────────────────────────────────┐
                 │  Layer 5: Hybrid Decision Engine    │
                 │  - Consensus Probability Synthesis  │
                 │  - Plain-English Explanation (XAI)  │
                 │  - Two-Tier LRU & Disk Cache Store  │
                 └──────────────────┬──────────────────┘
                                    ▼
       [ Final Verdict: REAL NEWS / FAKE NEWS + Verified Citations ]
```

---

## Key Features

### 1. Multimodal Input Modes
* **Text Analysis:** Paste any news headline, excerpt, or full article.
* **Live Article URL Scraper:** Direct extraction of web articles using automated metadata, headline, and paragraph parsing via `src/scraper.py`.
* **Client-Side Image OCR:** Upload screenshots or photos of social media posts; text is extracted locally using WebAssembly-powered Tesseract.js.
* **Microphone Voice Dictation:** Real-time speech-to-text input via the browser Web Speech API for hands-free analysis.

### 2. Multi-Model ML Ensemble
* **Soft-Voting Ensemble:** Combines **Linear SVM (Platt Calibrated)**, **Logistic Regression**, and **Multinomial Naive Bayes** to achieve lower variance and high generalization.
* **25,000 N-Gram Features:** Sublinear term-frequency scaling across unigrams and bigrams strictly fitted on training splits to prevent data leakage.
* **Calibrated Confidence:** Generates true posterior probabilities rather than arbitrary distance heuristics.

### 3. Real-Time Web Corroboration Agent
* **Multi-API Parallelism:** Uses `ThreadPoolExecutor` to query **Google Fact Check Tools API**, **NewsAPI**, **GNews**, and **Google News RSS** concurrently in under 1.5 seconds.
* **Zero-Key Fallback:** Works immediately out of the box with zero required API keys using Google News RSS and Wikipedia REST feeds.
* **Wikipedia Encyclopedic Grounding:** Automatically extracts recognized entities (political figures, organizations, nations) and pulls summaries from the official Wikipedia API.

### 4. Anti-False-Positive Warfare & Hoax Filter
* **Predicate-Action Validation:** Strictly matches claimed actions (e.g., attacks, bombings, airstrikes) against retrieved articles to prevent false correlations with unrelated diplomatic summits or peace talks.
* **Uncorroborated Critical Claim Detection:** High-stakes geopolitical breaking claims (e.g., assassinations, war declarations, military strikes) that have zero coverage across global news wires (Reuters, AP, BBC, Al Jazeera) are identified as fabricated.
* **Source Trust Scoring:** Automatically evaluates live sources against an internal catalog of 130+ ranked news domains.

### 5. Explainable AI (XAI) & Instant Caching
* **Linguistic Driver Tags:** Displays the exact vocabulary signals driving the model's classification.
* **Two-Tier Caching:** Sub-millisecond (`0.05ms`) response times for repeated queries via client-side storage and persistent disk cache (`data/cache_store.json`).
* **1-Click PDF Verification Export:** Generates an official, print-ready Fact-Check Verification Report complete with a unique Report ID, timestamp, and citation list.

---

## Benchmark & Performance Evaluation

The model was trained and evaluated on a multi-domain consolidated dataset of **12,736 balanced articles** compiled from the McIntire benchmark, FakeNewsNet PolitiFact, and FakeNewsNet GossipCop datasets (10,188 training samples, 2,548 hold-out test samples).

### Model Performance Metrics

| Model | Accuracy | Precision | Recall | F1 Score | F1 (Macro) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **82.03%** | **82.08%** | **82.03%** | **82.02%** | **82.02%** |
| **Ensemble Classifier** | **81.75%** | **81.87%** | **81.75%** | **81.73%** | **81.73%** |
| **Linear SVM (Calibrated)** | **81.24%** | **81.27%** | **81.24%** | **81.24%** | **81.24%** |
| **Multinomial Naive Bayes** | **79.79%** | **80.50%** | **79.79%** | **79.67%** | **79.67%** |

*All metrics are verified on unseen hold-out test sets. Benchmark data and plots are located in [`results/`](results/).*

---

## Quick Start

### 1. Prerequisites
* Python 3.10 or higher
* `pip` package manager

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/jachinsamuel/AI_Fake_News_Detector.git
cd AI_Fake_News_Detector
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment Configuration (Optional)

The application functions completely out of the box using built-in web search fallback engines. If you wish to use official API keys, copy `.env.example` to `.env`:

```ini
# .env
NEWS_API_KEY=your_newsapi_key_here
GOOGLE_FACTCHECK_API_KEY=your_google_factcheck_api_key_here
GNEWS_API_KEY=your_gnews_api_key_here
```

### 4. Dataset Preparation & Model Training

To download and prepare the 12,736-sample dataset and train the ensemble models:

```bash
# 1. Download and standardize datasets
python data/download_or_prepare.py

# 2. Train vectorizer, individual models, and ensemble classifier
python src/train.py
```

### 5. Launch the Application

```bash
python app.py
```

Open **`http://localhost:5000`** in your browser.

---

## Test Suite & Verification

The project includes an automated test suite containing **38 unit and integration tests** across 9 test suites:

```bash
python -m unittest tests/test_pipeline.py
```

### Test Coverage Highlights
* `TestPreprocessingPipeline`: URL removal, HTML stripping, punctuation cleaning, and tokenization.
* `TestModelPredictionAndExplainability`: Ensemble inference, probability ranges, and XAI feature outputs.
* `TestFlaskAPI`: Status codes, JSON structures, error handling, and latency endpoints.
* `TestWebVerifierAndKnowledgeGrounding`: Live web search, source extraction, and Wikipedia grounding.
* `TestCacheAndScraper`: LRU cache hits, cache resets, and URL article parsing.
* `TestForensicStylometry`: Linguistic metric computation and text length bounds.
* `TestWorldKnowledgeAndRefutation`: False-positive refutation and geopolitical breaking claim detection.
* `TestImageOcrEndpoint`: OCR input handling and base64 text parsing.
* `TestRealWorldProofsAndCredibility`: Domain trust calculation and publisher tier verification.

---

## REST API Reference

### `POST /predict`
Analyzes a news headline or text excerpt.

**Request:**
```json
{
  "text": "NASA James Webb Space Telescope discovers distant galaxy at cosmic dawn.",
  "check_web": true
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "prediction": "REAL",
  "confidence": 94.0,
  "confidence_decimal": 0.94,
  "model": "Soft-Voting Ensemble (SVM + LR + NB)",
  "important_features": ["telescope", "galaxy", "nasa", "space", "distant"],
  "explanation": "Corroborating reporting found across 4 active web news articles.",
  "web_verification": {
    "status": "SUCCESS",
    "web_verdict": "CORROBORATED_BY_LIVE_NEWS",
    "sources_count": 4,
    "live_sources": [
      {
        "title": "NASA Roman Space Telescope to map cosmic structures",
        "source": "SpaceNews",
        "url": "https://...",
        "credibility": {
          "trust_score": 92,
          "badge": "High Credibility"
        }
      }
    ],
    "wikipedia_grounding": {
      "is_grounded": true,
      "entity": "James Webb Space Telescope",
      "description": "Space telescope operated by NASA",
      "url": "https://en.wikipedia.org/wiki/James_Webb_Space_Telescope"
    }
  },
  "processing_time_ms": 480.2,
  "cached": false
}
```

### `POST /api/scrape-url`
Fetches and extracts headline and text from a news URL.

**Request:**
```json
{
  "url": "https://example.com/breaking-news-article"
}
```

### `POST /export-report`
Generates a printable HTML/PDF Fact-Check Verification Certificate from analysis data.

### Additional Endpoints
* `GET /api/config` — Returns active API services, search engines, and cache status.
* `GET /api/metrics` — Returns model benchmark scores and training parameters.
* `GET /api/examples` — Returns verified real and fake news sample claims.
* `GET /api/health` — System uptime and model readiness check.

---

## Project Structure

```
FakeNews/
├── .env.example              # Template environment variables
├── .gitignore                # Git exclusions (credentials, caches, virtualenvs)
├── app.py                    # Flask application server & REST endpoints
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── data/
│   ├── download_or_prepare.py# Multi-dataset acquisition & consolidation
│   ├── add_article.py        # Helper to append curated samples
│   ├── news.csv              # 12,736 labeled articles (McIntire + FakeNewsNet)
│   └── cache_store.json      # Persistent disk query cache
├── models/
│   ├── best_model.pkl        # Production model checkpoint
│   ├── ensemble_classifier.pkl # Soft-voting ensemble (SVM + LR + NB)
│   ├── linear_svm.pkl        # Platt-calibrated Linear Support Vector Machine
│   ├── logistic_regression.pkl# L2-regularized Logistic Regression
│   ├── naive_bayes.pkl       # Multinomial Naive Bayes classifier
│   ├── vectorizer.pkl        # Fitted TF-IDF feature extractor (25k terms)
│   ├── label_encoder.pkl     # Encoded target class mapping
│   └── model_metadata.json   # Training scores, metrics & hyperparameters
├── results/
│   ├── model_comparison.csv  # Precision, Recall, Accuracy, F1 benchmark table
│   ├── model_comparison.png  # Performance visual chart
│   ├── confusion_matrices.png# Model confusion matrices
│   └── eda_distribution.png  # Dataset class & length distribution plots
├── src/
│   ├── cache.py              # Two-tier LRU memory + disk cache manager
│   ├── config.py             # API keys and environment configuration
│   ├── distilbert_train.py   # Optional Hugging Face transformer fine-tuning
│   ├── evaluate.py           # Metric calculations & evaluation plots
│   ├── explain.py            # Feature contributions & plain-English XAI
│   ├── features.py           # TF-IDF vectorization & feature engineering
│   ├── predict.py            # Hybrid inference & decision synthesis engine
│   ├── preprocessing.py      # NLP text cleaning & normalization pipeline
│   ├── scraper.py            # Web article extraction & DOM parsing
│   ├── train.py              # Model training, GridSearchCV & calibration
│   └── web_verifier.py       # Live multi-threaded search & Wikipedia agent
├── static/
│   ├── script.js             # Client-side UI controller (OCR, Voice, API fetch)
│   └── style.css             # Responsive theme & layout styling
├── templates/
│   ├── index.html            # Main web application dashboard
│   └── report_template.html  # Fact-check PDF verification certificate template
└── tests/
    └── test_pipeline.py      # 38 Automated unit & integration tests
```

---

## License

This project is licensed under the [MIT License](LICENSE).
