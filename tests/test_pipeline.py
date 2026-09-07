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

    def test_scrape_url_route_invalid(self):
        response = self.client.post(
            "/api/scrape-url",
            data=json.dumps({"url": ""}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)

    def test_export_report_route(self):
        payload = {
            "prediction": "REAL",
            "confidence": 98.2,
            "model": "Soft-Voting Ensemble",
            "input_text": "Sample verified news",
            "explanation": "Verified on web",
            "feature_details": [{"word": "sample", "direction": "REAL", "impact": "High", "score": "+1.0"}]
        }
        response = self.client.post(
            "/export-report",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"FACT-CHECK VERIFICATION CERTIFICATE", response.data)


class TestWebVerifierAndKnowledgeGrounding(unittest.TestCase):
    """Test AI Live Web Verification agent and Wikipedia Grounding."""

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

    def test_wikipedia_grounding(self):
        from src.web_verifier import query_wikipedia_grounding
        res = query_wikipedia_grounding("Narendra_Modi", "Narendra Modi is the prime minister of india")
        self.assertIsNotNone(res)
        self.assertIn("entity", res)
        self.assertTrue(res["is_grounded"])


class TestCacheAndScraper(unittest.TestCase):
    """Test LRU QueryCache and URL Scraper."""

    def test_lru_cache_operations(self):
        from src.cache import QueryCache
        cache = QueryCache(maxsize=3, ttl_seconds=60)
        cache.set("test query", {"res": "ok"})
        hit = cache.get("test query")
        self.assertIsNotNone(hit)
        self.assertEqual(hit["res"], "ok")
        self.assertTrue(hit.get("cached"))
        miss = cache.get("nonexistent query")
        self.assertIsNone(miss)

    def test_scraper_validation(self):
        from src.scraper import scrape_article_from_url
        with self.assertRaises(ValueError):
            scrape_article_from_url("")


class TestForensicStylometry(unittest.TestCase):
    """Test psycholinguistic stylometric feature engineering and extraction."""

    def test_sensationalism_extraction(self):
        from src.features import analyze_text_stylometry
        clickbait = "SHOCKING BOMBSHELL: MIRACLE HERB CURES ALL CANCER THEY DONT WANT YOU TO KNOW!!!"
        res = analyze_text_stylometry(clickbait)
        self.assertIn(res["sensationalism_level"], ["Moderate", "High"])
        self.assertGreater(res["punctuation_dramatism"], 0.2)
        self.assertGreater(res["uppercase_ratio"], 0.3)
        self.assertEqual(res["style_verdict"], "Sensationalist / Clickbait")

    def test_journalistic_attribution_extraction(self):
        from src.features import analyze_text_stylometry
        formal = (
            "WASHINGTON (Reuters) - The Federal Reserve announced an interest rate adjustment "
            "following official committee hearings on Wednesday, according to a spokesperson."
        )
        res = analyze_text_stylometry(formal)
        self.assertGreater(res["attribution_score"], 0.1)
        self.assertEqual(res["sensationalism_level"], "Low")
        self.assertEqual(res["style_verdict"], "Formal Journalistic")

    def test_stylometric_transformer(self):
        from src.features import StylometricFeatureExtractor
        import numpy as np
        extractor = StylometricFeatureExtractor()
        data = ["Breaking news report with facts.", "SHOCKING BOMBSHELL HOAX!!!"]
        matrix = extractor.transform(data)
        self.assertIsInstance(matrix, np.ndarray)
        self.assertEqual(matrix.shape, (2, 10))


class TestWorldKnowledgeAndRefutation(unittest.TestCase):
    """Test corporate leadership, scientific consensus, and NLI refutation."""

    def test_tech_ceo_world_gk(self):
        from src.web_verifier import verify_world_gk_claim
        real_claim = verify_world_gk_claim("Tim Cook is the CEO of Apple")
        self.assertIsNotNone(real_claim)
        self.assertEqual(real_claim["verdict"], "REAL")
        self.assertGreaterEqual(real_claim["confidence"], 98.0)

        fake_claim = verify_world_gk_claim("Jachin Samuel is the CEO of Apple")
        self.assertIsNotNone(fake_claim)
        self.assertEqual(fake_claim["verdict"], "FAKE")
        self.assertIn("Tim Cook", fake_claim["explanation"])

    def test_scientific_consensus_debunks(self):
        from src.web_verifier import verify_world_gk_claim
        flat_earth = verify_world_gk_claim("The earth is flat")
        self.assertIsNotNone(flat_earth)
        self.assertEqual(flat_earth["verdict"], "FAKE")
        self.assertGreaterEqual(flat_earth["confidence"], 99.0)

        vaccine_myth = verify_world_gk_claim("Vaccines cause autism")
        self.assertIsNotNone(vaccine_myth)
        self.assertEqual(vaccine_myth["verdict"], "FAKE")

    def test_headline_refutation_detection(self):
        from src.web_verifier import check_headline_refutation
        query_words = ["garlic", "cures", "cancer"]
        debunk_title = "Fact Check: Garlic does not cure cancer, oncologists warn"
        self.assertTrue(check_headline_refutation(query_words, debunk_title))

        corroborate_title = "Clinical study examines new cancer therapy"
        self.assertFalse(check_headline_refutation(query_words, corroborate_title))

    def test_predict_includes_stylometry(self):
        from src.predict import get_predictor
        predictor = get_predictor()
        res = predictor.predict("NASA James Webb Space Telescope discovers distant galaxies.", check_web=False)
        self.assertIn("stylometry", res)
        self.assertIn("sensationalism_density", res["stylometry"])
        self.assertIn("style_verdict", res["stylometry"])


class TestImageOcrEndpoint(unittest.TestCase):
    """Test /api/ocr image input decoding and validation."""

    @classmethod
    def setUpClass(cls):
        from app import app
        cls.client = app.test_client()

    def test_ocr_missing_payload(self):
        resp = self.client.post("/api/ocr")
        self.assertEqual(resp.status_code, 400)
        data = json.loads(resp.data)
        self.assertIn("error", data)

    def test_ocr_invalid_base64(self):
        resp = self.client.post(
            "/api/ocr",
            data=json.dumps({"image": "not-valid-base64!!!"}),
            content_type="application/json"
        )
        self.assertEqual(resp.status_code, 400)

    def test_ocr_valid_base64_image(self):
        import io
        import base64
        from PIL import Image

        # Create a small 50x20 test image in memory
        img = Image.new("RGB", (50, 20), color=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")

        resp = self.client.post(
            "/api/ocr",
            data=json.dumps({"image": f"data:image/png;base64,{b64_data}"}),
            content_type="application/json"
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["image_size"], "50x20")

    def test_ocr_multipart_file_upload(self):
        import io
        from PIL import Image

        img = Image.new("RGB", (60, 30), color=(240, 240, 240))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        resp = self.client.post(
            "/api/ocr",
            data={"image": (buf, "headline_screenshot.jpg")},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["image_size"], "60x30")


if __name__ == "__main__":
    unittest.main()
