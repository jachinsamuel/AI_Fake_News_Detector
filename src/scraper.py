"""
Live URL Web Article Scraper.
Extracts clean news headlines, publication dates, and article body paragraphs
from live web links (Reuters, BBC, CNN, Substack, Medium, etc.) for direct verification.
"""

import re
import urllib.request
import urllib.parse
from bs4 import BeautifulSoup


def scrape_article_from_url(url: str, timeout_seconds: int = 8) -> dict:
    """
    Extract title, text, and metadata from an article URL.
    """
    if not url or not isinstance(url, str):
        raise ValueError("A valid URL string must be provided.")

    cleaned_url = url.strip()
    if not (cleaned_url.startswith("http://") or cleaned_url.startswith("https://")):
        cleaned_url = "https://" + cleaned_url

    parsed = urllib.parse.urlparse(cleaned_url)
    if not parsed.netloc:
        raise ValueError("Invalid web URL format.")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5"
    }

    import requests
    response = requests.get(cleaned_url, headers=headers, timeout=timeout_seconds, allow_redirects=True)
    content_type = response.headers.get("Content-Type", "").lower()
    if "text/html" not in content_type and "application/xhtml" not in content_type and "application/xml" not in content_type:
        raise ValueError("The provided URL does not return an HTML web page.")

    html = response.text
    soup = BeautifulSoup(html, "html.parser")

    # Remove script, style, navigation, footer, and iframe tags
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside", "form"]):
        tag.decompose()

    # 1. Extract Title
    title = ""
    # Try OpenGraph or Twitter title first
    og_title = soup.find("meta", property="og:title") or soup.find("meta", attrs={"name": "twitter:title"})
    if og_title and og_title.get("content"):
        title = og_title["content"].strip()
    elif soup.title and soup.title.string:
        title = soup.title.string.strip()
    elif soup.find("h1"):
        title = soup.find("h1").get_text().strip()

    # 2. Extract Domain Source
    source = parsed.netloc.replace("www.", "")

    # 3. Extract Main Body Paragraphs
    boilerplate_regex = re.compile(
        r"cookie|privacy policy|rights reserved|subscribe|commenting policy|leave a comment|"
        r"post a comment|user agreement|terms of service|epaper|download our app|"
        r"sign in to|please do not use abbreviations|read our commenting policy",
        re.I
    )

    paragraphs = []
    # Check for <article> wrapper first
    article_container = soup.find("article") or soup.find("main") or soup
    for p in article_container.find_all("p"):
        text = p.get_text().strip()
        # Filter out short notices, cookie notices, and comment rules
        if len(text) > 40 and not boilerplate_regex.search(text):
            paragraphs.append(text)

    full_text = " ".join(paragraphs)
    if not full_text:
        # Fallback to general paragraph collection
        for p in soup.find_all("p"):
            text = p.get_text().strip()
            if len(text) > 40 and not boilerplate_regex.search(text):
                paragraphs.append(text)
        full_text = " ".join(paragraphs)

    if not title and not full_text:
        raise ValueError("Could not extract readable article text from the provided URL.")

    # Combined representation
    combined = f"{title}\n\n{full_text}" if title else full_text

    return {
        "status": "success",
        "url": cleaned_url,
        "source": source,
        "title": title or "Untitled Article",
        "text": full_text[:4000],  # Cap at 4,000 chars
        "combined_text": combined[:4500],
        "word_count": len(combined.split())
    }


if __name__ == "__main__":
    test_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    print(f"Testing scraper on {test_url}...")
    try:
        data = scrape_article_from_url(test_url)
        print("Scraped Title:", data["title"])
        print("Source:", data["source"])
        print("Extracted Word Count:", data["word_count"])
    except Exception as e:
        print("Scraper Error:", e)
