"""
Live AI Web Verification & Fact-Checking Agent.
Queries real-world news APIs (NewsAPI, GNews, Google Fact Check Tools API)
and live public search engines (Google News RSS, DuckDuckGo) to verify claims,
corroborate reporting with credible news organizations, and detect debunked stories.
"""

import os
import sys
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import json
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.config import (
    NEWS_API_KEY,
    GOOGLE_FACTCHECK_API_KEY,
    GNEWS_API_KEY,
    THE_GUARDIAN_API_KEY,
    WEB_SEARCH_TIMEOUT_SECONDS
)

# Major reputable journalistic domains for trust scoring
CREDIBLE_DOMAINS = [
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "nytimes.com",
    "washingtonpost.com", "wsj.com", "bloomberg.com", "theguardian.com",
    "cnn.com", "nbcnews.com", "cbsnews.com", "abcnews.go.com", "npr.org",
    "politico.com", "snopes.com", "politifact.com", "factcheck.org", "nature.com",
    "sciencemag.org", "nasa.gov", "who.int", "cdc.gov"
]


def extract_search_query(text: str, max_words: int = 10) -> str:
    """
    Extract a concise, searchable query from headline or article text.
    Removes timestamps, city tags (e.g. 'WASHINGTON (Reuters) -'), and filler words.
    """
    # Clean leading wire datelines like 'WASHINGTON (Reuters) -' or 'NEW YORK ---'
    cleaned = re.sub(r"^[A-Z\s,]+\([A-Za-z\s]+\)\s*[-—–:]\s*", "", text.strip())
    cleaned = re.sub(r"^(BREAKING|ALERT|EXCLUSIVE|SHOCKING|WATCH|UPDATE)[:\s-]*", "", cleaned, flags=re.IGNORECASE)
    
    # Take first sentence
    sentences = re.split(r"[.!?\n]", cleaned)
    first_sentence = sentences[0].strip() if sentences else cleaned
    
    # Remove punctuation
    query = re.sub(r"[^\w\s]", " ", first_sentence)
    words = [w for w in query.split() if len(w) > 2][:max_words]
    return " ".join(words)


def query_google_fact_check(query: str) -> list:
    """
    Query Google Fact Check Tools API for existing fact checks (Snopes, PolitiFact, etc.).
    """
    if not GOOGLE_FACTCHECK_API_KEY or not query:
        return []
        
    try:
        encoded_q = urllib.parse.quote_plus(query)
        url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={encoded_q}&key={GOOGLE_FACTCHECK_API_KEY}&languageCode=en"
        
        req = urllib.request.Request(url, headers={"User-Agent": "FakeNewsDetector/2.0"})
        with urllib.request.urlopen(req, timeout=WEB_SEARCH_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            
        claims = data.get("claims", [])
        fact_checks = []
        for claim in claims[:3]:
            claim_text = claim.get("text", "")
            claimant = claim.get("claimant", "Unknown")
            reviews = claim.get("claimReview", [])
            for rev in reviews:
                publisher = rev.get("publisher", {}).get("name", "Fact Checker")
                rating = rev.get("textualRating", "Unrated")
                review_url = rev.get("url", "")
                fact_checks.append({
                    "claim": claim_text,
                    "claimant": claimant,
                    "publisher": publisher,
                    "rating": rating,
                    "url": review_url
                })
        return fact_checks
    except Exception as e:
        # Silently degrade if API is unreachable
        return []


def query_news_api(query: str) -> list:
    """Query NewsAPI.org for matching news articles."""
    if not NEWS_API_KEY or not query:
        return []
        
    try:
        encoded_q = urllib.parse.quote_plus(query)
        url = f"https://newsapi.org/v2/everything?q={encoded_q}&language=en&sortBy=relevancy&pageSize=4&apiKey={NEWS_API_KEY}"
        
        req = urllib.request.Request(url, headers={"User-Agent": "FakeNewsDetector/2.0"})
        with urllib.request.urlopen(req, timeout=WEB_SEARCH_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            
        articles = data.get("articles", [])
        results = []
        for a in articles:
            results.append({
                "title": a.get("title", ""),
                "source": a.get("source", {}).get("name", "News Outlet"),
                "url": a.get("url", ""),
                "description": a.get("description", "") or "",
                "published_at": a.get("publishedAt", "")[:10]
            })
        return results
    except Exception:
        return []


def query_gnews_api(query: str) -> list:
    """Query GNews API for matching live articles."""
    if not GNEWS_API_KEY or not query:
        return []
        
    try:
        encoded_q = urllib.parse.quote_plus(query)
        url = f"https://gnews.io/api/v4/search?q={encoded_q}&lang=en&max=4&apikey={GNEWS_API_KEY}"
        
        req = urllib.request.Request(url, headers={"User-Agent": "FakeNewsDetector/2.0"})
        with urllib.request.urlopen(req, timeout=WEB_SEARCH_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            
        articles = data.get("articles", [])
        results = []
        for a in articles:
            results.append({
                "title": a.get("title", ""),
                "source": a.get("source", {}).get("name", "News Outlet"),
                "url": a.get("url", ""),
                "description": a.get("description", "") or "",
                "published_at": a.get("publishedAt", "")[:10]
            })
        return results
    except Exception:
        return []


def query_google_news_rss(query: str) -> list:
    """
    Zero-key public Google News RSS query engine.
    Fetches real-time matching articles from verified international news outlets.
    """
    if not query:
        return []
        
    try:
        encoded_q = urllib.parse.quote_plus(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_q}&hl=en-US&gl=US&ceid=US:en"
        
        req = urllib.request.Request(
            rss_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=WEB_SEARCH_TIMEOUT_SECONDS) as resp:
            xml_data = resp.read()
            
        root = ET.fromstring(xml_data)
        items = root.findall(".//item")
        
        sources = []
        for item in items[:4]:
            title = item.find("title").text if item.find("title") is not None else ""
            link = item.find("link").text if item.find("link") is not None else ""
            pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""
            source_el = item.find("source")
            source_name = source_el.text if source_el is not None else "Verified News Outlet"
            
            # Clean headline title if it contains source trailing tag
            if " - " in title:
                title_clean = title.rsplit(" - ", 1)[0]
            else:
                title_clean = title
                
            sources.append({
                "title": title_clean,
                "source": source_name,
                "url": link,
                "published_at": pub_date[:16] if pub_date else ""
            })
        return sources
    except Exception as e:
        return []


def verify_article_on_web(text: str) -> dict:
    """
    Main web verification agent:
    1. Extracts search query.
    2. Queries Google Fact Check API, NewsAPI, GNews, or Google News RSS.
    3. Analyzes source credibility and consensus.
    """
    query = extract_search_query(text)
    if not query or len(query.strip()) < 4:
        return {
            "status": "SKIPPED",
            "query_used": "",
            "sources_found": 0,
            "fact_checks": [],
            "live_sources": [],
            "web_summary": "Text was too short to generate a search query for web verification."
        }

    # 1. Check Fact Check APIs
    fact_checks = query_google_fact_check(query)

    # 2. Check News APIs (fallback to zero-key Google News RSS)
    live_sources = []
    if NEWS_API_KEY:
        live_sources = query_news_api(query)
    elif GNEWS_API_KEY:
        live_sources = query_gnews_api(query)
        
    if not live_sources:
        # Live Google News RSS fallback (Free, reliable, no key needed)
        live_sources = query_google_news_rss(query)

    # 3. Analyze Web Consensus
    has_fact_checks = len(fact_checks) > 0
    has_live_news = len(live_sources) > 0
    
    # Check if fact check says false/debunked
    is_debunked = False
    for fc in fact_checks:
        rating_lower = fc.get("rating", "").lower()
        if any(w in rating_lower for w in ["false", "pants on fire", "fake", "hoax", "misleading", "incorrect", "unproven"]):
            is_debunked = True
            break
            
    # Check for credible domain matches
    credible_matches = 0
    for src in live_sources:
        src_text = (src.get("source", "") + " " + src.get("url", "")).lower()
        if any(dom in src_text for dom in CREDIBLE_DOMAINS):
            credible_matches += 1

    if is_debunked:
        verdict = "DEBUNKED_BY_FACT_CHECKERS"
        summary = f"Flagged by independent fact-checkers ({fact_checks[0]['publisher']}) with rating: '{fact_checks[0]['rating']}'."
    elif has_live_news and credible_matches > 0:
        verdict = "CORROBORATED_BY_LIVE_NEWS"
        summary = f"Corroborating coverage identified across {len(live_sources)} live news sources including {live_sources[0]['source']}."
    elif has_live_news:
        verdict = "MATCHING_NEWS_FOUND"
        summary = f"Found {len(live_sources)} related articles reporting on this topic on the live web."
    else:
        verdict = "NO_LIVE_COVERAGE"
        summary = "No direct matching reporting found on major live news feeds. May be an unverified claim, satire, or historical text."

    return {
        "status": "SUCCESS",
        "query_used": query,
        "web_verdict": verdict,
        "is_debunked": is_debunked,
        "sources_count": len(live_sources),
        "credible_sources_count": credible_matches,
        "fact_checks": fact_checks,
        "live_sources": live_sources,
        "web_summary": summary
    }


if __name__ == "__main__":
    sample = "WASHINGTON (Reuters) - NASA's James Webb Space Telescope has identified an ancient galaxy."
    print("Testing Web Verifier on:", sample)
    res = verify_article_on_web(sample)
    print("Result:", json.dumps(res, indent=2))
