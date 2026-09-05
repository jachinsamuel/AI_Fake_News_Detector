"""
In-Memory Thread-Safe LRU Cache with TTL and Persistent Disk Backing.
Caches prediction results and web verification payloads to deliver
instant (< 1ms) responses for repeated queries, eliminating redundant API and model calls.
Survives server restarts via atomic disk-store synchronization.
"""

import os
import re
import json
import time
import hashlib
import threading
from collections import OrderedDict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "data")
CACHE_FILE = os.path.join(DATA_DIR, "cache_store.json")


class QueryCache:
    def __init__(self, maxsize: int = 1000, ttl_seconds: int = 86400, persist_to_disk: bool = True):
        self.maxsize = maxsize
        self.ttl = ttl_seconds
        self.persist_to_disk = persist_to_disk
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        
        # Telemetry metrics
        self.hits = 0
        self.misses = 0
        self.saved_latency_ms = 0.0

        if self.persist_to_disk:
            self._load_from_disk()

    def _normalize_text(self, text: str) -> str:
        """Strip surrounding punctuation, quotes, and normalize whitespace."""
        cleaned = text.lower().strip()
        # Strip leading and trailing quotes or brackets
        cleaned = re.sub(r'^[“"\'\s\[\({<]+|[”"\'\s\]\)}>]+$', '', cleaned)
        # Strip trailing sentence punctuation
        cleaned = re.sub(r'[.,;?!]+$', '', cleaned)
        # Collapse whitespace
        return " ".join(cleaned.split())

    def _hash_key(self, text: str, check_web: bool = True) -> str:
        """Create a normalized SHA-256 hash for query text."""
        normalized = self._normalize_text(text)
        key_str = f"{normalized}::check_web={check_web}"
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()

    def _load_from_disk(self):
        """Warm up cache from persistent disk storage."""
        if not os.path.exists(CACHE_FILE):
            return
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                stored = json.load(f)
            now = time.time()
            valid_count = 0
            for key, entry in stored.items():
                if now - entry.get("timestamp", 0) <= self.ttl:
                    self.cache[key] = entry
                    valid_count += 1
                    if len(self.cache) >= self.maxsize:
                        break
        except Exception:
            pass

    def _save_to_disk_async(self):
        """Safely serialize active cache entries to disk without blocking the main thread."""
        if not self.persist_to_disk:
            return
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            # Take a snapshot under lock
            with self.lock:
                snapshot = dict(self.cache)
            
            temp_file = CACHE_FILE + ".tmp"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False)
            if os.path.exists(CACHE_FILE):
                os.replace(temp_file, CACHE_FILE)
            else:
                os.rename(temp_file, CACHE_FILE)
        except Exception:
            pass

    def get(self, text: str, check_web: bool = True):
        """Retrieve cached result if valid and unexpired."""
        key = self._hash_key(text, check_web)
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
                
            entry = self.cache[key]
            # Check TTL
            if time.time() - entry["timestamp"] > self.ttl:
                del self.cache[key]
                self.misses += 1
                return None
                
            # Move to end for LRU policy
            self.cache.move_to_end(key)
            self.hits += 1
            result = dict(entry["data"])
            result["cached"] = True
            
            # Estimate latency saved (typical web query takes ~350ms)
            self.saved_latency_ms += 350.0
            return result

    def set(self, text: str, data: dict, check_web: bool = True):
        """Store result with current timestamp and persist to disk."""
        key = self._hash_key(text, check_web)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = {
                "timestamp": time.time(),
                "data": data
            }
            # Evict oldest entry if size exceeded
            if len(self.cache) > self.maxsize:
                self.cache.popitem(last=False)

        # Trigger non-blocking disk sync periodically
        if self.persist_to_disk and len(self.cache) % 3 == 0:
            threading.Thread(target=self._save_to_disk_async, daemon=True).start()

    def clear(self):
        """Clear all cached entries in memory and disk."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            self.saved_latency_ms = 0.0
        if self.persist_to_disk and os.path.exists(CACHE_FILE):
            try:
                os.remove(CACHE_FILE)
            except Exception:
                pass

    def stats(self) -> dict:
        """Return cache health telemetry and performance metrics."""
        with self.lock:
            total_lookups = self.hits + self.misses
            hit_ratio = round((self.hits / total_lookups * 100), 1) if total_lookups > 0 else 0.0
            return {
                "total_entries": len(self.cache),
                "max_size": self.maxsize,
                "ttl_seconds": self.ttl,
                "hits": self.hits,
                "misses": self.misses,
                "hit_ratio_percent": hit_ratio,
                "estimated_saved_latency_sec": round(self.saved_latency_ms / 1000.0, 2)
            }


# Global singleton instance
_cache_instance = QueryCache(maxsize=1000, ttl_seconds=86400, persist_to_disk=True)


def get_cache() -> QueryCache:
    return _cache_instance
