"""
In-Memory Thread-Safe LRU Cache with TTL (Time-To-Live).
Caches prediction results and web verification payloads to deliver
instant (< 5ms) responses for repeated queries and eliminate duplicate API calls.
"""

import time
import hashlib
import threading
from collections import OrderedDict


class QueryCache:
    def __init__(self, maxsize: int = 500, ttl_seconds: int = 3600):
        self.maxsize = maxsize
        self.ttl = ttl_seconds
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def _hash_key(self, text: str, check_web: bool = True) -> str:
        """Create a normalized SHA-256 hash for query text."""
        normalized = " ".join(text.lower().strip().split())
        key_str = f"{normalized}::check_web={check_web}"
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()

    def get(self, text: str, check_web: bool = True):
        """Retrieve cached result if valid and unexpired."""
        key = self._hash_key(text, check_web)
        with self.lock:
            if key not in self.cache:
                return None
                
            entry = self.cache[key]
            # Check TTL
            if time.time() - entry["timestamp"] > self.ttl:
                del self.cache[key]
                return None
                
            # Move to end for LRU policy
            self.cache.move_to_end(key)
            result = dict(entry["data"])
            result["cached"] = True
            return result

    def set(self, text: str, data: dict, check_web: bool = True):
        """Store result with current timestamp."""
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

    def clear(self):
        """Clear all cached entries."""
        with self.lock:
            self.cache.clear()

    def stats(self) -> dict:
        """Return cache health telemetry."""
        with self.lock:
            return {
                "total_entries": len(self.cache),
                "max_size": self.maxsize,
                "ttl_seconds": self.ttl
            }


# Global singleton instance
_cache_instance = QueryCache(maxsize=500, ttl_seconds=3600)


def get_cache() -> QueryCache:
    return _cache_instance
