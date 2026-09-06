"""
Forensic Stylometry & Psycholinguistic Feature Extraction Module.
Extracts 12 quantitative disinformation markers from news text:
1. Sensationalism & Hyperbole Density (clickbait trigger lexicon)
2. Uppercase Shouting Ratio (all-caps word ratio)
3. Punctuation Dramatism (excessive !, ?, !?, ...)
4. Journalistic Attribution Score (epistemic authority: 'according to', 'spokesperson', 'confirmed')
5. Conspiracy Framing Score ('they don't want you to know', 'big pharma', 'coverup')
6. Lexical Diversity (Type-Token Ratio / TTR)
7. Quotation Frequency & Scare Quotes
8. Average Sentence Length & Informational Density
9. Pronoun Subjectivity Ratio (1st/2nd person vs 3rd person objective reporting)
10. Numeric & Statistical Grounding (dates, percentages, statistics)

Implements scikit-learn BaseEstimator and TransformerMixin for seamless FeatureUnion integration.
"""

import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Compiled forensic lexicons
SENSATIONALISM_TERMS = {
    "shocking", "bombshell", "miracle", "unbelievable", "secret", "exposed", "banned",
    "censored", "coverup", "whistleblower", "mind control", "cure all", "cures all",
    "magic herb", "miracle herb", "leaked", "conspiracy", "hoax", "hidden truth",
    "forbidden", "they don't want you to know", "share before deleted", "watch before deleted",
    "wake up", "urgent warning", "classified leak", "unmasked", "undeniable proof",
    "smoking gun", "breakthrough they hid", "instant cure"
}

JOURNALISTIC_ATTRIBUTION_TERMS = {
    "according to", "spokesperson", "confirmed", "reported", "statement",
    "press briefing", "published in", "researchers found", "official said",
    "reuters", "associated press", "testified", "announced on", "formal briefing",
    "peer-reviewed", "investigation revealed", "spoke to reporters", "briefed reporters",
    "cited official", "government data", "bureau of", "department of"
}

CONSPIRACY_TERMS = {
    "deep state", "big pharma", "corrupt media", "mainstream media refuses",
    "globalist", "puppet masters", "secret underground", "cabal", "elites hide",
    "sheeple", "plandemic", "microchip", "5g towers", "chemtrails", "illuminati"
}

PRONOUNS_SUBJECTIVE = {"i", "me", "my", "we", "us", "our", "you", "your"}
PRONOUNS_OBJECTIVE = {"he", "him", "his", "she", "her", "they", "them", "their", "it", "its"}

RE_EXCLAMATION = re.compile(r"!+")
RE_QUESTION = re.compile(r"\?+")
RE_QUOTES = re.compile(r'["\'“”‘’]')
RE_NUMBERS = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
RE_WORDS = re.compile(r"\b[a-zA-Z]+\b")
RE_SENTENCES = re.compile(r"[.!?\n]+")


KNOWN_ACRONYMS = {
    "WASHINGTON", "LONDON", "PARIS", "BERLIN", "TOKYO", "BEIJING", "MOSCOW", "DELHI",
    "REUTERS", "AFP", "NASA", "NATO", "WHO", "CDC", "FDA", "USA", "FBI", "CIA", "UN",
    "GOP", "DNC", "PM", "CEO", "CFO", "CTO", "AI", "EU", "UK", "US"
}


def analyze_text_stylometry(text: str) -> dict:
    """
    Compute fine-grained forensic stylometric metrics for a single news article/headline.
    Returns a dictionary of raw and normalized indicators.
    """
    if not text or not isinstance(text, str):
        return {
            "sensationalism_density": 0.0,
            "uppercase_ratio": 0.0,
            "punctuation_dramatism": 0.0,
            "attribution_score": 0.0,
            "conspiracy_score": 0.0,
            "lexical_diversity": 0.0,
            "quotation_density": 0.0,
            "avg_sentence_len": 0.0,
            "subjectivity_ratio": 0.0,
            "numeric_grounding": 0.0,
            "sensationalism_level": "Low",
            "attribution_level": "Absent",
            "style_verdict": "Neutral"
        }

    raw = text.strip()
    words = RE_WORDS.findall(raw)
    total_words = len(words)
    total_chars = len(raw)

    if total_words == 0:
        total_words = 1  # prevent div by zero

    lower_text = raw.lower()
    lower_words = [w.lower() for w in words]

    # 1. Sensationalism Density
    sens_matches = sum(1 for term in SENSATIONALISM_TERMS if term in lower_text)
    sens_density = min(1.0, (sens_matches * 3.5) / (total_words / 10.0 + 1.0))

    # 2. Uppercase Shouting Ratio (excluding single-letter 'I' or standard datelines/acronyms)
    all_caps_words = sum(1 for w in words if len(w) > 2 and w.isupper() and w not in KNOWN_ACRONYMS)
    upper_ratio = min(1.0, (all_caps_words * 2.0) / (total_words + 1.0))

    # 3. Punctuation Dramatism
    excl_count = len(RE_EXCLAMATION.findall(raw))
    ques_count = len(RE_QUESTION.findall(raw))
    multi_punct = len(re.findall(r"[!?]{2,}", raw))
    punct_score = min(1.0, (excl_count * 1.5 + ques_count * 0.8 + multi_punct * 3.0) / (total_words / 15.0 + 1.0))

    # 4. Journalistic Attribution Score
    attr_matches = sum(1 for term in JOURNALISTIC_ATTRIBUTION_TERMS if term in lower_text)
    attr_score = min(1.0, (attr_matches * 2.5) / (total_words / 20.0 + 1.0))

    # 5. Conspiracy Framing Score
    consp_matches = sum(1 for term in CONSPIRACY_TERMS if term in lower_text)
    consp_score = min(1.0, (consp_matches * 4.0) / (total_words / 10.0 + 1.0))

    # 6. Lexical Diversity (Type-Token Ratio)
    unique_words = len(set(lower_words))
    lexical_ttr = unique_words / total_words

    # 7. Quotation Frequency
    quote_matches = len(RE_QUOTES.findall(raw))
    quote_density = min(1.0, quote_matches / (total_words / 10.0 + 1.0))

    # 8. Sentence Structure & Length
    sentences = [s.strip() for s in RE_SENTENCES.split(raw) if s.strip()]
    num_sentences = max(1, len(sentences))
    avg_sentence_len = total_words / num_sentences

    # 9. Pronoun Subjectivity Ratio
    subj_count = sum(1 for w in lower_words if w in PRONOUNS_SUBJECTIVE)
    obj_count = sum(1 for w in lower_words if w in PRONOUNS_OBJECTIVE)
    total_pronouns = subj_count + obj_count
    subjectivity_ratio = (subj_count / (total_pronouns + 1.0)) if total_pronouns > 0 else 0.0

    # 10. Numeric & Statistical Grounding
    numbers_count = len(RE_NUMBERS.findall(raw))
    numeric_grounding = min(1.0, numbers_count / (total_words / 10.0 + 1.0))

    # Qualitative categorizations
    sens_level = "High" if sens_density > 0.4 else ("Moderate" if sens_density > 0.15 else "Low")
    attr_level = "High" if attr_score > 0.4 else ("Moderate" if attr_score > 0.1 else "Absent")
    
    if sens_density > 0.35 or punct_score > 0.4 or consp_score > 0.3:
        style_verdict = "Sensationalist / Clickbait"
    elif attr_score >= 0.15 and upper_ratio < 0.15 and punct_score < 0.2:
        style_verdict = "Formal Journalistic"
    else:
        style_verdict = "Neutral / Conversational"

    return {
        "sensationalism_density": round(float(sens_density), 4),
        "uppercase_ratio": round(float(upper_ratio), 4),
        "punctuation_dramatism": round(float(punct_score), 4),
        "attribution_score": round(float(attr_score), 4),
        "conspiracy_score": round(float(consp_score), 4),
        "lexical_diversity": round(float(lexical_ttr), 4),
        "quotation_density": round(float(quote_density), 4),
        "avg_sentence_len": round(float(avg_sentence_len), 2),
        "subjectivity_ratio": round(float(subjectivity_ratio), 4),
        "numeric_grounding": round(float(numeric_grounding), 4),
        "sensationalism_level": sens_level,
        "attribution_level": attr_level,
        "style_verdict": style_verdict
    }


class StylometricFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that converts text samples
    into a 10-dimensional dense numpy matrix of forensic stylometric features.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            metrics = analyze_text_stylometry(text)
            features.append([
                metrics["sensationalism_density"],
                metrics["uppercase_ratio"],
                metrics["punctuation_dramatism"],
                metrics["attribution_score"],
                metrics["conspiracy_score"],
                metrics["lexical_diversity"],
                metrics["quotation_density"],
                metrics["avg_sentence_len"] / 50.0,  # normalized scale
                metrics["subjectivity_ratio"],
                metrics["numeric_grounding"]
            ])
        return np.array(features, dtype=np.float32)
