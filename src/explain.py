"""
Explainable AI (XAI) Module for Fake News Detection.
Extracts top influential linguistic terms and feature contributions
for individual predictions and global model interpretation.
"""

import numpy as np


def get_model_coefficients(model):
    """
    Extract weights/coefficients vector from various sklearn classifiers.
    Handles LogisticRegression, LinearSVC, CalibratedClassifierCV, and MultinomialNB.
    """
    # 1. CalibratedClassifierCV wrapping a linear model
    if hasattr(model, "calibrated_classifiers_"):
        coefs = []
        for clf in model.calibrated_classifiers_:
            # CalibratedClassifierCV might wrap estimator
            base = getattr(clf, "estimator", None) or getattr(clf, "base_estimator", None)
            if base is not None and hasattr(base, "coef_"):
                coefs.append(base.coef_[0])
        if coefs:
            return np.mean(coefs, axis=0)
            
    # 2. Direct linear models (LogisticRegression, LinearSVC)
    if hasattr(model, "coef_"):
        return model.coef_[0]

    # 3. Multinomial Naive Bayes (difference in log probabilities)
    if hasattr(model, "feature_log_prob_"):
        # Index 1 (REAL) log prob minus Index 0 (FAKE) log prob
        return model.feature_log_prob_[1] - model.feature_log_prob_[0]

    return None


def explain_prediction(
    model,
    vectorizer,
    raw_text: str,
    preprocessed_text: str,
    predicted_label: str,
    confidence: float,
    label_encoder=None,
    top_n: int = 8
) -> dict:
    """
    Extract the most influential words in the query that influenced the prediction.
    
    Returns:
        dict containing:
            - predicted_label
            - confidence
            - top_features: list of dicts [{"word": str, "weight": float, "direction": "FAKE"|"REAL"}]
            - explanation_text: contextual summary
            - limitation_disclaimer: clear ML text-classification limitation notice
    """
    coefs = get_model_coefficients(model)
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    # Transform preprocessed text to TF-IDF vector
    tfidf_vec = vectorizer.transform([preprocessed_text])
    feature_indices = tfidf_vec.nonzero()[1]
    
    influential_words = []
    
    if coefs is not None and len(feature_indices) > 0:
        # Calculate individual contribution = coefficient * tfidf_value
        contributions = []
        for idx in feature_indices:
            word = feature_names[idx]
            tfidf_val = tfidf_vec[0, idx]
            weight = coefs[idx]
            # Contribution score: positive means REAL, negative means FAKE
            score = weight * tfidf_val
            contributions.append((word, score, weight, tfidf_val))
            
        # If predicted FAKE: prioritize strongest negative scores (leaning FAKE)
        # If predicted REAL: prioritize strongest positive scores (leaning REAL)
        if predicted_label.upper() == "FAKE":
            # Sort by most negative score first
            sorted_by_impact = sorted(contributions, key=lambda x: x[1])
        else:
            # Sort by most positive score first
            sorted_by_impact = sorted(contributions, key=lambda x: x[1], reverse=True)
            
        for word, score, weight, tfidf_val in sorted_by_impact[:top_n]:
            direction = "REAL" if score > 0 else "FAKE"
            influential_words.append({
                "word": word,
                "score": round(float(abs(score)), 4),
                "direction": direction,
                "impact": "High" if abs(score) > 0.3 else "Medium"
            })
            
    # Fallback if no specific TF-IDF terms matched or short query
    if not influential_words and preprocessed_text:
        words = preprocessed_text.split()[:top_n]
        for w in words:
            influential_words.append({
                "word": w,
                "score": 0.10,
                "direction": predicted_label,
                "impact": "Contextual"
            })

    # Prepare user-friendly explanation
    word_list_str = ", ".join([f"'{item['word']}'" for item in influential_words[:5]])
    if predicted_label.upper() == "FAKE":
        explanation_text = (
            f"The model detected stylistic patterns and linguistic markers typically associated with misinformation, "
            f"such as {word_list_str}."
        )
    else:
        explanation_text = (
            f"The article exhibits formal reporting syntax, structured journalistic terminology, and keywords like "
            f"{word_list_str} characteristic of verified news sources."
        )

    disclaimer = (
        "Academic & Technical Disclaimer: This system is a machine-learning-based natural language classifier. "
        "Predictions reflect stylistic and vocabulary correlations learned from benchmark training data; "
        "the model does NOT independently verify real-world facts against live external news databases."
    )

    return {
        "prediction": predicted_label,
        "confidence": confidence,
        "important_features": [item["word"] for item in influential_words],
        "feature_details": influential_words,
        "explanation": explanation_text,
        "disclaimer": disclaimer
    }
