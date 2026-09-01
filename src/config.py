"""
Configuration Manager for Real-World News APIs and Project Settings.
Loads environment variables from .env file or system environment.
"""

import os
from dotenv import load_dotenv

# Load .env file from project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(ROOT_DIR, ".env")
load_dotenv(ENV_PATH)

# API Keys
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "").strip()
GOOGLE_FACTCHECK_API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY", "").strip()
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "").strip()

# Configuration flags
ENABLE_LIVE_WEB_CHECK = os.getenv("ENABLE_LIVE_WEB_CHECK", "true").lower() in ("true", "1", "yes")
WEB_SEARCH_TIMEOUT_SECONDS = int(os.getenv("WEB_SEARCH_TIMEOUT_SECONDS", "5"))


def get_available_api_services():
    """Return list of configured external API providers."""
    services = []
    if NEWS_API_KEY:
        services.append("NewsAPI")
    if GOOGLE_FACTCHECK_API_KEY:
        services.append("Google Fact Check API")
    if GNEWS_API_KEY:
        services.append("GNews API")
    # Live fallback engines
    services.append("Google News RSS (Live)")
    services.append("DuckDuckGo Web (Live)")
    return services
