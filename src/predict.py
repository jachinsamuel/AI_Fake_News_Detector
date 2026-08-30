"""
Inference & Prediction Pipeline.
Loads saved model artifacts, executes preprocessing, generates predictions,
calculates calibrated confidence scores, and extracts explainable linguistic features.
"""

import os
import sys
import time
import joblib

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.preprocessing import preprocess_text
from src.explain import explain_prediction

MODELS_DIR = os.path.join(ROOT_DIR, "models")


class FakeNewsPredictor:
    def __init__(self, model_name: str = "best_model.pkl"):
        self.model_path = os.path.join(MODELS_DIR, model_name)
        self.vectorizer_path = os.path.join(MODELS_DIR, "vectorizer.pkl")
        self.encoder_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
        
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.model_title = "Linear SVM (Calibrated)"
        self.load_artifacts()

    def load_artifacts(self):
        """Load trained model, TF-IDF vectorizer, and label encoder."""
        if not os.path.exists(self.model_path) or not os.path.exists(self.vectorizer_path):
            raise FileNotFoundError(
                f"Model artifacts not found in {MODELS_DIR}. Please run 'python src/train.py' first."
            )
        
        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)
        self.label_encoder = joblib.load(self.encoder_path)
        
        # Determine model display name
        clf_type = type(self.model).__name__
        if "Calibrated" in clf_type:
            self.model_title = "Linear SVM (Calibrated)"
        elif "Logistic" in clf_type:
            self.model_title = "Logistic Regression"
        elif "MultinomialNB" in clf_type:
            self.model_title = "Multinomial Naive Bayes"
        else:
            self.model_title = clf_type

    def switch_model(self, model_key: str):
        """Switch active model between 'linear_svm', 'logistic_regression', 'naive_bayes', or 'best_model'."""
        filename = f"{model_key}.pkl" if not model_key.endswith(".pkl") else model_key
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            self.model = joblib.load(path)
            self.model_path = path
            self.load_artifacts()
            return True
        return False

    def predict(self, raw_text: str) -> dict:
        """
        Execute full inference pipeline for a given raw text.
        """
        t0 = time.time()
        
        if not raw_text or not isinstance(raw_text, str) or len(raw_text.strip()) == 0:
            raise ValueError("Input news text cannot be empty.")
            
        raw_text_stripped = raw_text.strip()
        word_count = len(raw_text_stripped.split())
        char_count = len(raw_text_stripped)
        
        if word_count < 3 and char_count < 15:
            raise ValueError("Input text is too short to analyze reliably (minimum 3 words required).")
            
        # 1. NLP Preprocessing
        clean_text = preprocess_text(raw_text_stripped)
        
        if not clean_text:
            clean_text = raw_text_stripped.lower()
            
        # 2. TF-IDF Transformation
        tfidf_features = self.vectorizer.transform([clean_text])
        
        # 3. Model Prediction
        pred_encoded = self.model.predict(tfidf_features)[0]
        predicted_label = self.label_encoder.inverse_transform([pred_encoded])[0]
        
        # 4. Confidence / Probability Calculation
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(tfidf_features)[0]
            # Prob for predicted class
            confidence = float(probabilities[pred_encoded])
        elif hasattr(self.model, "decision_function"):
            # Calibrated decision score approximation if predict_proba is not available
            score = self.model.decision_function(tfidf_features)[0]
            confidence = float(1.0 / (1.0 + np.exp(-abs(score))))
        else:
            confidence = 0.90
            
        # Ensure confidence is formatted as clean percentage (e.g., 94.2)
        confidence_pct = round(confidence * 100, 2)
        
        # 5. Explainable AI Feature Extraction
        xai_info = explain_prediction(
            model=self.model,
            vectorizer=self.vectorizer,
            raw_text=raw_text_stripped,
            preprocessed_text=clean_text,
            predicted_label=predicted_label,
            confidence=confidence_pct,
            label_encoder=self.label_encoder,
            top_n=8
        )
        
        elapsed_ms = round((time.time() - t0) * 1000, 2)
        
        return {
            "prediction": predicted_label,
            "confidence": confidence_pct,
            "confidence_decimal": round(confidence, 4),
            "model_used": self.model_title,
            "important_features": xai_info["important_features"],
            "feature_details": xai_info["feature_details"],
            "explanation": xai_info["explanation"],
            "disclaimer": xai_info["disclaimer"],
            "stats": {
                "word_count": word_count,
                "char_count": char_count,
                "cleaned_tokens_count": len(clean_text.split())
            },
            "processing_time_ms": elapsed_ms
        }


# Singleton instance for quick module-level imports
_predictor_instance = None


def get_predictor():
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = FakeNewsPredictor()
    return _predictor_instance


if __name__ == "__main__":
    predictor = get_predictor()
    
    test_articles = [
        "BREAKING: Secret cure for aging revealed by miracle doctor, government is trying to ban it!",
        "The Federal Reserve announced on Wednesday that benchmark interest rates will remain unchanged following economic data."
    ]
    
    for article in test_articles:
        res = predictor.predict(article)
        print("\n" + "-" * 50)
        print(f"Article: {article[:60]}...")
        print(f"Prediction: {res['prediction']} ({res['confidence']}%) | Model: {res['model_used']}")
        print(f"Important Terms: {res['important_features']}")
        print(f"Explanation: {res['explanation']}")
