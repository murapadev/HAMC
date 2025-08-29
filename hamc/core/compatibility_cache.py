from typing import Dict, Set, Tuple, Callable, Any, Optional, TypeVar, Generic, Union
from functools import lru_cache
from weakref import WeakKeyDictionary
from threading import Lock
import time
import logging
from ..config.advanced_config import get_cache_config

T = TypeVar('T')

class CompatibilityCache(Generic[T]):
    """Thread-safe cache for compatibility checks between elements.
    
    Uses WeakKeyDictionary to prevent memory leaks and Lock for thread safety.
    Caches are specific to validator functions to prevent incorrect reuse.
    Implements TTL, LRU eviction and cache statistics.
    """
    
    def __init__(self, max_size: Optional[int] = None, ttl: Optional[float] = None):
        """Initialize compatibility cache.
        
        Args:
            max_size: Maximum number of entries per validator cache
            ttl: Time-to-live in seconds for cache entries
        """
        config = get_cache_config()
        self._caches: WeakKeyDictionary = WeakKeyDictionary()
        self._lock = Lock()
        self._max_size = max_size if max_size is not None else config.compatibility_cache_max_size
        self._ttl = ttl if ttl is not None else config.compatibility_cache_ttl  # Time-to-live in seconds
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self.logger = logging.getLogger(__name__)

    def are_compatible(self, value1: T, value2: T, 
                      validator: Callable[[T, T], bool]) -> bool:
        """Check if two values are compatible using the provided validator.
        
        Args:
            value1: First value to check
            value2: Second value to check
            validator: Function that determines compatibility
            
        Returns:
            bool: True if values are compatible, False otherwise
        """
        # Get or create cache for this validator
        with self._lock:
            if validator not in self._caches:
                self._caches[validator] = {}
            cache = self._caches[validator]
            
            # Check cache
            key = (value1, value2)
            entry = cache.get(key)
            if entry:
                result, timestamp = entry
                if time.time() - timestamp < self._ttl:
                    self._hits += 1
                    return result
                # Entry expired
            
            # Check reverse key
            rev_key = (value2, value1)
            entry = cache.get(rev_key)
            if entry:
                result, timestamp = entry
                if time.time() - timestamp < self._ttl:
                    self._hits += 1
                    return result
                # Entry expired
            
            # Cache miss, compute and cache result
            self._misses += 1
            result = validator(value1, value2)
            
            # Evict oldest entry if cache full
            if len(cache) >= self._max_size:
                try:
                    oldest_key = min(cache.items(), key=lambda x: x[1][1])[0]
                    del cache[oldest_key]
                    self._evictions += 1
                except ValueError:
                    # Handle empty cache edge case
                    pass
                
            cache[key] = (result, time.time())
            return result

    def prefetch(self, values: Set[T], validator: Callable[[T, T], bool]) -> None:
        """Prefetch compatibility results for a set of values.
        
        Args:
            values: Set of values to prefetch compatibility for
            validator: Function that determines compatibility
        """
        with self._lock:
            if validator not in self._caches:
                self._caches[validator] = {}
            cache = self._caches[validator]
            
            for v1 in values:
                for v2 in values:
                    if v1 != v2:
                        key = (v1, v2)
                        if key not in cache:
                            result = validator(v1, v2)
                            cache[key] = (result, time.time())

    def clear(self) -> None:
        """Clear all cached compatibility results."""
        with self._lock:
            self._caches.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dict[str, Any]: Statistics about cache usage
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_ratio = self._hits / total_requests if total_requests > 0 else 0.0
            cache_sizes = {str(validator): len(cache) 
                          for validator, cache in self._caches.items()}
            
            return {
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_ratio": hit_ratio,
                "total_validators": len(self._caches),
                "cache_sizes": cache_sizes
            }
