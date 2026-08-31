"""
Comprehensive Test Suite for AI Fake News Detection System.
Tests preprocessing, feature vectorization, explainability, prediction, and Flask API endpoints.
"""

import os
import sys
import unittest
import json

# Ensure project root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.preprocessing import (
    preprocess_text,
    remove_urls,
    remove_html,
    remove_special_characters
)
from src.predict import FakeNewsPredictor, get_predictor
from app import app


class TestPreprocessingPipeline(unittest.TestCase):
    """Test text cleaning and NLP normalization."""

    def test_url_removal(self):
        raw = "Read more at https://example.com/article and www.fakestory.net/alert"
        cleaned = remove_urls(raw)
        self.assertNotIn("https://", cleaned)
        self.assertNotIn("www.fakestory.net", cleaned)

    def test_html_removal(self):
        raw = "<div><h3>Breaking News</h3><p>Scientists announce result.</p></div>"
        cleaned = remove_html(raw)
        self.assertNotIn("<div>", cleaned)
        self.assertNotIn("<p>", cleaned)
        self.assertIn("Breaking News", cleaned)

    def test_special_character_removal(self):
        raw = "Alert!!! 100% Miracle #1 ($500) @doctor."
        cleaned = remove_special_characters(raw)
        self.assertNotIn("!", cleaned)
        self.assertNotIn("%", cleaned)
        self.assertNotIn("#", cleaned)
        self.assertNotIn("$", cleaned)

    def test_end_to_end_preprocessing(self):
        raw = "BREAKING: Incredible discoveries were made at https://nasa.gov/news <p>Yesterday</p>!"
        cleaned = preprocess_text(raw)
        self.assertIsInstance(cleaned, str)
        self.assertTrue(len(cleaned) > 0)
        self.assertEqual(cleaned, cleaned.lower())
        self.assertNotIn("http", cleaned)
        self.assertNotIn("<p>", cleaned)


class TestModelPredictionAndExplainability(unittest.TestCase):
    """Test predictor and explainability functionality."""

    @classmethod
    def setUpClass(cls):
        cls.predictor = get_predictor()

    def test_predictor_loaded(self):
        self.assertIsNotNone(self.predictor.model)
        self.assertIsNotNone(self.predictor.vectorizer)
        self.assertIsNotNone(self.predictor.label_encoder)

    def test_prediction_output_structure(self):
        sample_article = (
            "WASHINGTON (Reuters) - The United States Senate on Thursday approved a bipartisan "
            "infrastructure funding bill following committee hearings and budget office analysis."
        )
        res = self.predictor.predict(sample_article)
        
        # Verify required keys
        required_keys = [
            "prediction", "confidence", "model_used", "important_features",
            "feature_details", "explanation", "disclaimer", "stats", "processing_time_ms"
        ]
        for key in required_keys:
            self.assertIn(key, res)

        self.assertIn(res["prediction"], ["REAL", "FAKE"])
        self.assertGreaterEqual(res["confidence"], 50.0)
        self.assertLessEqual(res["confidence"], 100.0)
        self.assertIsInstance(res["important_features"], list)
        self.assertGreater(len(res["important_features"]), 0)

    def test_empty_or_short_input_validation(self):
        with self.assertRaises(ValueError):
            self.predictor.predict("")

        with self.assertRaises(ValueError):
            self.predictor.predict("hi")


class TestFlaskAPI(unittest.TestCase):
    """Test Flask web routes and REST API endpoints."""

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_index_route(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Fake News Detector", response.data)

    def test_health_route(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "online")
        self.assertTrue(data["predictor_ready"])

    def test_examples_route(self):
        response = self.client.get("/api/examples")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "success")
        self.assertGreater(len(data["examples"]), 0)

    def test_metrics_route(self):
        response = self.client.get("/api/metrics")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "success")
        self.assertIn("best_model_name", data["data"])

    def test_predict_endpoint_valid(self):
        payload = {
            "text": "WASHINGTON (Reuters) - Federal officials released quarterly economic data today."
        }
        response = self.client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "success")
        self.assertIn(data["prediction"], ["REAL", "FAKE"])
        self.assertIn("confidence", data)
        self.assertIn("important_features", data)

    def test_predict_endpoint_empty_input(self):
        payload = {"text": "   "}
        response = self.client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn("error", data)

    def test_predict_endpoint_short_input(self):
        payload = {"text": "Hello"}
        response = self.client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)

    def test_config_route(self):
        response = self.client.get("/api/config")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "online")
        self.assertIn("active_api_services", data)


class TestWebVerifier(unittest.TestCase):
    """Test AI Live Web Verification agent."""

    def test_query_extraction(self):
        from src.web_verifier import extract_search_query
        sample = "WASHINGTON (Reuters) - NASA launches new Mars explorer mission to search for water."
        query = extract_search_query(sample)
        self.assertIsInstance(query, str)
        self.assertGreater(len(query), 5)
        self.assertNotIn("WASHINGTON", query)

    def test_verify_article_structure(self):
        from src.web_verifier import verify_article_on_web
        sample = "NASA James Webb Space Telescope discovers distant galaxies in deep space."
        res = verify_article_on_web(sample)
        self.assertIn("status", res)
        self.assertIn("web_verdict", res)
        self.assertIn("live_sources", res)
        self.assertIn("web_summary", res)


if __name__ == "__main__":
    unittest.main()
