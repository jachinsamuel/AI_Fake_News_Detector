"""
Live AI Web Verification & Fact-Checking Agent.
Queries real-world news APIs (NewsAPI, GNews, Google Fact Check Tools API)
and live public search engines (Google News RSS, DuckDuckGo) to verify claims,
corroborate reporting with credible news organizations, and detect debunked stories.
Includes Semantic Relevance Matching, Entity Syntactic Role Analysis, and Hoax Detection.
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
    WEB_SEARCH_TIMEOUT_SECONDS
)

# Major reputable journalistic domains for trust scoring
CREDIBLE_DOMAINS = [
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "nytimes.com",
    "washingtonpost.com", "wsj.com", "bloomberg.com", "theguardian.com",
    "cnn.com", "nbcnews.com", "cbsnews.com", "abcnews.go.com", "npr.org",
    "politico.com", "snopes.com", "politifact.com", "factcheck.org", "nature.com",
    "sciencemag.org", "nasa.gov", "who.int", "cdc.gov", "dw.com", "thehindu.com",
    "ndtv.com", "indianexpress.com"
]

# High-impact critical claims that require strict subject+predicate verification
DEATH_TERMS = {
    "dead", "died", "dies", "death", "killed", "assassinated", "murdered", "passed away", "succumbs"
}

ARREST_TERMS = {
    "arrested", "imprisoned", "jailed", "detained", "indicted", "convicted"
}

CURE_TERMS = {
    "miracle cure", "cures all", "cures cancer", "secret herb", "reverses aging", "miracle herb"
}

CRITICAL_CLAIM_TERMS = DEATH_TERMS.union(ARREST_TERMS).union(CURE_TERMS).union({
    "resigned", "resigns", "hoax", "alien", "mind control", "5g"
})

# Words indicating that a living person is merely reacting to or condoling someone else's death
CONDOLENCE_TERMS = {
    "condoles", "condoled", "condolence", "condolences", "mourns", "mourned",
    "mourning", "grief", "grieves", "tribute", "reacts to", "expresses sorrow",
    "prays for", "pays tribute"
}


def extract_search_query(text: str, max_words: int = 10) -> str:
    """
    Extract a concise, searchable query from headline or article text.
    Removes timestamps, city tags (e.g. 'WASHINGTON (Reuters) -'), and filler words.
    """
    cleaned = re.sub(r"^[A-Z\s,]+\([A-Za-z\s]+\)\s*[-—–:]\s*", "", text.strip())
    cleaned = re.sub(r"^(BREAKING|ALERT|EXCLUSIVE|SHOCKING|WATCH|UPDATE)[:\s-]*", "", cleaned, flags=re.IGNORECASE)
    
    sentences = re.split(r"[.!?\n]", cleaned)
    first_sentence = sentences[0].strip() if sentences else cleaned
    
    query = re.sub(r"[^\w\s]", " ", first_sentence)
    words = [w for w in query.split() if len(w) > 2][:max_words]
    return " ".join(words)


def is_headline_semantically_relevant(query_words: list, title: str) -> bool:
    """
    Ensure a returned news article actually matches the core claim of the query.
    Detects false positives like 'PM Modi condoles bus accident deaths' when query is 'Modi is dead'.
    """
    title_lower = title.lower()
    
    # 1. Death Claim Specific Check
    has_death_in_query = any(d in query_words for d in DEATH_TERMS)
    if has_death_in_query:
        # If the title mentions condolences / mourning, the subject is ALIVE and condoling
        if any(c in title_lower for c in CONDOLENCE_TERMS):
            return False
            
        subj_words = [w for w in query_words if w not in DEATH_TERMS and len(w) > 2]
        if subj_words:
            # Check if any death term and any subject term exist in the title in direct relation
            has_subj = any(w in title_lower for w in subj_words)
            has_death = any(d in title_lower for d in DEATH_TERMS)
            if not (has_subj and has_death):
                return False
            # Check syntax pattern (e.g. 'Modi dies', 'Death of Modi', 'Modi's death')
            pattern = r"\b(" + "|".join(re.escape(w) for w in subj_words) + r")\b.*?\b(" + "|".join(re.escape(d) for d in DEATH_TERMS) + r")\b"
            pattern_rev = r"\b(" + "|".join(re.escape(d) for d in DEATH_TERMS) + r")\b.*?\b(" + "|".join(re.escape(w) for w in subj_words) + r")\b"
            if not (re.search(pattern, title_lower) or re.search(pattern_rev, title_lower)):
                return False
        return True

    # 2. Arrest Claim Specific Check
    has_arrest_in_query = any(a in query_words for a in ARREST_TERMS)
    if has_arrest_in_query:
        subj_words = [w for w in query_words if w not in ARREST_TERMS and len(w) > 2]
        if subj_words:
            has_subj = any(w in title_lower for w in subj_words)
            has_arr = any(a in title_lower for a in ARREST_TERMS)
            return has_subj and has_arr

    # 3. General Semantic Headline Matching
    matching_count = sum(1 for w in query_words if w in title_lower and len(w) > 2)
    return matching_count >= max(1, len(query_words) // 3)


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
    except Exception:
        return []


def query_news_api(query: str) -> list:
    """Query NewsAPI.org for matching news articles."""
    if not NEWS_API_KEY or not query:
        return []
        
    try:
        encoded_q = urllib.parse.quote_plus(query)
        url = f"https://newsapi.org/v2/everything?q={encoded_q}&language=en&sortBy=relevancy&pageSize=6&apiKey={NEWS_API_KEY}"
        
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
        url = f"https://gnews.io/api/v4/search?q={encoded_q}&lang=en&max=6&apikey={GNEWS_API_KEY}"
        
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
        for item in items[:6]:
            title = item.find("title").text if item.find("title") is not None else ""
            link = item.find("link").text if item.find("link") is not None else ""
            pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""
            source_el = item.find("source")
            source_name = source_el.text if source_el is not None else "Verified News Outlet"
            
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
    except Exception:
        return []


def verify_article_on_web(text: str) -> dict:
    """
    Main web verification agent:
    1. Extracts search query & identifies critical claim terms (e.g. death/cure/arrest hoaxes).
    2. Queries Google Fact Check API, GNews, NewsAPI, and Google News RSS.
    3. Performs semantic relevance filtering to discard unaligned search noise.
    4. Evaluates whether the specific claim is verified, debunked, or an uncorroborated hoax.
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

    query_words = [w.lower() for w in query.split()]
    is_critical = any(w in CRITICAL_CLAIM_TERMS for w in query_words)

    # 1. Check Fact Check APIs
    fact_checks = query_google_fact_check(query)

    # 2. Check News APIs & Live Feeds
    raw_sources = []
    if GNEWS_API_KEY:
        raw_sources = query_gnews_api(query)
    if not raw_sources and NEWS_API_KEY:
        raw_sources = query_news_api(query)
    if not raw_sources:
        raw_sources = query_google_news_rss(query)

    # 3. Filter Semantically Relevant Headlines
    relevant_sources = []
    for s in raw_sources:
        title = s.get("title", "")
        if is_headline_semantically_relevant(query_words, title):
            relevant_sources.append(s)

    # 4. Analyze Web Consensus
    has_fact_checks = len(fact_checks) > 0
    has_relevant_news = len(relevant_sources) > 0
    
    # Check if fact-checkers debunked it
    is_debunked = False
    for fc in fact_checks:
        rating_lower = fc.get("rating", "").lower()
        if any(w in rating_lower for w in ["false", "pants on fire", "fake", "hoax", "misleading", "incorrect", "unproven", "satire"]):
            is_debunked = True
            break
            
    # Check for credible domain matches among relevant sources
    credible_matches = 0
    for src in relevant_sources:
        src_text = (src.get("source", "") + " " + src.get("url", "")).lower()
        if any(dom in src_text for dom in CREDIBLE_DOMAINS):
            credible_matches += 1

    # Evaluate Final Web Verdict
    is_uncorroborated_hoax = False

    if is_debunked:
        verdict = "DEBUNKED_BY_FACT_CHECKERS"
        summary = f"Flagged and debunked by independent fact-checkers ({fact_checks[0]['publisher']}): rating '{fact_checks[0]['rating']}'."
    elif is_critical and not has_relevant_news:
        # High-impact critical claim with 0 relevant matching headlines confirming it
        is_uncorroborated_hoax = True
        verdict = "UNCORROBORATED_CRITICAL_CLAIM"
        summary = f"Uncorroborated death/event rumor. If this major event were true, every global news wire would report it. Zero matching news reports confirm this claim."
    elif has_relevant_news and credible_matches > 0:
        verdict = "CORROBORATED_BY_LIVE_NEWS"
        summary = f"Corroborating coverage confirmed across {len(relevant_sources)} live news sources including {relevant_sources[0]['source']}."
    elif has_relevant_news:
        verdict = "MATCHING_NEWS_FOUND"
        summary = f"Found {len(relevant_sources)} related articles reporting on this topic on the live web."
    else:
        verdict = "NO_LIVE_COVERAGE"
        summary = "No active reporting found on major live news feeds. Likely an unverified claim, rumor, or historical text."

    return {
        "status": "SUCCESS",
        "query_used": query,
        "web_verdict": verdict,
        "is_debunked": is_debunked,
        "is_uncorroborated_hoax": is_uncorroborated_hoax,
        "sources_count": len(relevant_sources),
        "credible_sources_count": credible_matches,
        "fact_checks": fact_checks,
        "live_sources": relevant_sources,
        "web_summary": summary
    }
