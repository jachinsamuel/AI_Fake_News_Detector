"""
Flask REST API and Web Server for AI Fake News Detector.
Provides interactive UI rendering and RESTful prediction endpoints with Live Web Verification.
"""

import os
import sys
import json
import time
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.predict import get_predictor
from src.config import get_available_api_services

app = Flask(__name__)
CORS(app)

# Pre-load model predictor at startup
try:
    predictor = get_predictor()
except Exception as e:
    print(f"[WARNING] Could not load predictor at initialization: {e}")
    predictor = None

# Sample news articles for testing
SAMPLE_ARTICLES = [
    {
        "id": "real_science",
        "category": "Science & Space",
        "type": "REAL",
        "title": "James Webb Telescope Discovers Ancient Galaxy",
        "text": "WASHINGTON (Reuters) - NASA's James Webb Space Telescope has identified an extraordinarily distant and ancient galaxy that formed just 350 million years after the Big Bang, according to astronomers and peer-reviewed research published in the Astrophysical Journal. Researchers confirmed that spectroscopic data collected by infrared instruments provides strong evidence of early stellar formation during the cosmic dawn."
    },
    {
        "id": "fake_health",
        "category": "Health & Medicine",
        "type": "FAKE",
        "title": "Shocking Miracle Herb Cures All Aging and Diseases",
        "text": "BREAKING NEWS: Secret underground doctors have just leaked classified medical research revealing an ancient miracle herb that reverses all human aging and cures cancer in 48 hours! Big Pharma and corrupt government regulators are actively censoring this video and trying to ban the website to protect corporate profits. Share this before it gets deleted!"
    },
    {
        "id": "real_politics",
        "category": "Government & Policy",
        "type": "REAL",
        "title": "Senate Passes Bipartisan Infrastructure Funding Bill",
        "text": "WASHINGTON (Reuters) - The United States Senate on Thursday approved a bipartisan funding package aimed at modernizing federal transit systems and highway bridges. The 74-24 vote followed bipartisan negotiations between Democratic and Republican committee members. The Congressional Budget Office projected that the legislation will reduce freight transport bottlenecks over the next decade."
    },
    {
        "id": "fake_tech",
        "category": "Technology & AI",
        "type": "FAKE",
        "title": "Secret 5G Tower Transmitting Mind Control Signals",
        "text": "SHOCKING BOMBSHELL: Anonymous whistleblowers inside military intelligence reveal that newly installed 5G telecommunication towers are transmitting classified electromagnetic frequencies designed to control civilian minds and track citizen DNA. Major mainstream media outlets refuse to cover the leaked documents to keep the public obedient!"
    }
]


@app.route("/")
def index():
    """Render main web application interface."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint with Live Web Verification.
    Accepts JSON: {"text": "news text", "check_web": true}
    """
    global predictor
    if predictor is None:
        try:
            predictor = get_predictor()
        except Exception as e:
            return jsonify({
                "error": "Model initialization failure",
                "message": f"Could not load trained model: {str(e)}"
            }), 500

    # Parse input data
    input_text = ""
    check_web = True
    if request.is_json:
        data = request.get_json(silent=True) or {}
        input_text = data.get("text", "")
        check_web = data.get("check_web", True)
    elif request.form:
        input_text = request.form.get("text", "")
        check_web = request.form.get("check_web", "true").lower() in ("true", "1", "yes")
    else:
        raw_body = request.get_data(as_text=True)
        if raw_body:
            input_text = raw_body

    if not input_text or not input_text.strip():
        return jsonify({
            "error": "Validation Error",
            "message": "Input news article text cannot be empty. Please enter a headline or article body."
        }), 400

    cleaned_input = input_text.strip()
    words = cleaned_input.split()
    if len(words) < 3 and len(cleaned_input) < 15:
        return jsonify({
            "error": "Validation Error",
            "message": "Input text is too short. Please provide at least 3 words or a complete headline for reliable classification."
        }), 400

    try:
        result = predictor.predict(cleaned_input, check_web=check_web)
        return jsonify({
            "status": "success",
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "confidence_decimal": result["confidence_decimal"],
            "model": result["model_used"],
            "important_features": result["important_features"],
            "feature_details": result["feature_details"],
            "explanation": result["explanation"],
            "disclaimer": result["disclaimer"],
            "web_verification": result.get("web_verification"),
            "stats": result["stats"],
            "processing_time_ms": result["processing_time_ms"]
        }), 200

    except Exception as e:
        return jsonify({
            "error": "Inference Error",
            "message": f"An error occurred during prediction: {str(e)}"
        }), 500


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """Return model performance comparison metrics and training statistics."""
    metadata_path = os.path.join(CURRENT_DIR, "models", "model_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        return jsonify({"status": "success", "data": metadata}), 200
    else:
        return jsonify({
            "status": "error",
            "message": "Model metadata not found. Train models with 'python src/train.py' first."
        }), 404


@app.route("/api/examples", methods=["GET"])
def get_examples():
    """Return sample news articles for testing."""
    return jsonify({
        "status": "success",
        "examples": SAMPLE_ARTICLES
    }), 200


@app.route("/api/config", methods=["GET"])
def get_config():
    """Return active API services and configuration info."""
    return jsonify({
        "status": "online",
        "active_api_services": get_available_api_services(),
        "live_web_verification": True
    }), 200


@app.route("/api/health", methods=["GET"])
def health_check():
    """System health check endpoint."""
    return jsonify({
        "status": "online",
        "predictor_ready": predictor is not None,
        "service": "AI Fake News Detection System with Live Web Verification",
        "version": "2.0.0"
    }), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f" [STARTING] AI Fake News Detector with Live Web Verifier")
    print(f" Access URL: http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host="127.0.0.1", port=port, debug=False)
