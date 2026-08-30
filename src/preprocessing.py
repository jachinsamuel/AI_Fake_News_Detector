"""
Natural Language Processing (NLP) Preprocessing Pipeline.
Performs text normalization, URL/HTML stripping, punctuation cleanup,
tokenization, stopword removal, and WordNet lemmatization.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Initialize stopwords and lemmatizer safely
try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))

# Custom domain-specific stopwords if any (we keep standard NLTK list)
# Note: Retain words that might convey strong sentiment or stylistic indicators
try:
    LEMMATIZER = WordNetLemmatizer()
    # Test lemmatizer
    _ = LEMMATIZER.lemmatize("running")
except LookupError:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    LEMMATIZER = WordNetLemmatizer()


def remove_urls(text: str) -> str:
    """Remove hyperlinks and URLs from raw text."""
    url_pattern = re.compile(r"https?://\S+|www\.\S+|bit\.ly/\S+")
    return url_pattern.sub(" ", text)


def remove_html(text: str) -> str:
    """Remove HTML tags from raw text."""
    html_pattern = re.compile(r"<.*?>")
    return html_pattern.sub(" ", text)


def remove_special_characters(text: str) -> str:
    """Remove special characters, numbers, and extra symbols, leaving alphabetic words."""
    # Replace newlines, tabs, and non-alphabetic chars with single spaces
    text = re.sub(r"[\r\n\t]+", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Keep only alphabetic words and standard spacing
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return text


def clean_tokenize_lemmatize(text: str, remove_stops: bool = True) -> str:
    """
    Tokenizes text, filters stopwords, lemmatizes words to root form,
    and returns a clean space-separated string.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    # 1. Lowercasing
    text = text.lower()

    # 2. Strip URLs & HTML
    text = remove_urls(text)
    text = remove_html(text)

    # 3. Strip special characters and punctuation
    text = remove_special_characters(text)

    # 4. Tokenization (using split or nltk tokenizer)
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()

    # 5. Stopword removal and Lemmatization
    cleaned_tokens = []
    for token in tokens:
        token = token.strip()
        if len(token) > 1:  # ignore single stray letters
            if remove_stops and token in STOP_WORDS:
                continue
            lemma = LEMMATIZER.lemmatize(token)
            cleaned_tokens.append(lemma)

    return " ".join(cleaned_tokens)


def preprocess_text(text: str) -> str:
    """
    Standard single-text preprocessor for both training and inference pipelines.
    """
    return clean_tokenize_lemmatize(text, remove_stops=True)


def preprocess_corpus(corpus, verbose: bool = False):
    """
    Batch preprocessing for a list/series of news texts.
    """
    results = []
    total = len(corpus)
    for idx, item in enumerate(corpus):
        results.append(preprocess_text(item))
        if verbose and (idx + 1) % 1000 == 0:
            print(f"[PREPROCESSING] Processed {idx + 1}/{total} texts...")
    return results


if __name__ == "__main__":
    sample_text = (
        "BREAKING NEWS: Scientists discover UNBELIEVABLE breakthrough at https://harvard.edu/news! "
        "<p>The official government report confirmed 100% success yesterday.</p>"
    )
    print("--- RAW INPUT ---")
    print(sample_text)
    print("\n--- PREPROCESSED OUTPUT ---")
    print(preprocess_text(sample_text))
