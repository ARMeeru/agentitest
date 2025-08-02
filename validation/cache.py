# validation/cache.py
"""
Validation result caching for deterministic and repeatable results.

This module provides caching capabilities for validation results to ensure
deterministic behavior and improve performance by avoiding redundant validations.
"""

import hashlib
import json
import time
import pickle
import logging
from typing import Dict, Any, Optional, List, Union, Set
from datetime import datetime, timedelta, timezone
from dataclasses import asdict
from pathlib import Path

from .core import (
    ValidationResult,
    ValidationContext,
    ValidationStatus,
    ConfidenceScore,
    ValidationType
)


class CacheEntry:
    """Individual cache entry with metadata."""
    
    def __init__(
        self,
        result: ValidationResult,
        cache_key: str,
        created_at: datetime,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.result = result
        self.cache_key = cache_key
        self.created_at = created_at
        self.ttl_seconds = ttl_seconds
        self.metadata = metadata or {}
        
        # Mark result as cached
        self.result.from_cache = True
        self.result.cache_key = cache_key
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_seconds is None:
            return False  # No expiration
        
        age = (datetime.now(timezone.utc) - self.created_at).total_seconds()
        return age > self.ttl_seconds
    
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert cache entry to dictionary for serialization."""
        return {
            "result": self.result.to_dict(),
            "cache_key": self.cache_key,
            "created_at": self.created_at.isoformat(),
            "ttl_seconds": self.ttl_seconds,
            "metadata": self.metadata,
            "age_seconds": self.age_seconds()
        }


class ValidationResultCache:
    """
    Cache for validation results with configurable storage and expiration.
    
    This cache provides deterministic and repeatable validation results by storing
    validation outcomes and reusing them for identical inputs. Supports both
    in-memory and persistent storage options.
    """
    
    def __init__(
        self,
        max_entries: int = 1000,
        default_ttl_seconds: Optional[int] = 3600,  # 1 hour default
        enable_persistence: bool = False,
        cache_directory: Optional[str] = None,
        enable_compression: bool = True
    ):
        """
        Initialize validation result cache.
        
        Args:
            max_entries: Maximum number of entries to keep in memory
            default_ttl_seconds: Default TTL for cache entries (None = no expiration)
            enable_persistence: Enable persistent disk storage
            cache_directory: Directory for persistent cache files
            enable_compression: Enable compression for stored entries
        """
        self.max_entries = max_entries
        self.default_ttl_seconds = default_ttl_seconds
        self.enable_persistence = enable_persistence
        self.enable_compression = enable_compression
        
        # In-memory cache
        self._cache: Dict[str, CacheEntry] = {}
        self._access_times: Dict[str, datetime] = {}
        
        # Persistent storage
        if enable_persistence:
            self.cache_directory = Path(cache_directory or "./validation_cache")
            self.cache_directory.mkdir(exist_ok=True)
            self._load_persistent_cache()
        else:
            self.cache_directory = None
        
        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expired_entries": 0,
            "total_gets": 0,
            "total_sets": 0
        }
    
    def get(
        self,
        cache_key: str,
        ignore_expired: bool = False
    ) -> Optional[ValidationResult]:
        """
        Get validation result from cache.
        
        Args:
            cache_key: Cache key to lookup
            ignore_expired: Whether to return expired entries
            
        Returns:
            Cached ValidationResult or None if not found/expired
        """
        self._stats["total_gets"] += 1
        
        # Check in-memory cache first
        entry = self._cache.get(cache_key)
        
        if entry is None and self.enable_persistence:
            # Try to load from persistent storage
            entry = self._load_from_disk(cache_key)
            if entry:
                # Add to in-memory cache
                self._cache[cache_key] = entry
        
        if entry is None:
            self._stats["misses"] += 1
            return None
        
        # Check if expired
        if not ignore_expired and entry.is_expired():
            self._stats["expired_entries"] += 1
            self._remove_entry(cache_key)
            return None
        
        # Update access time
        self._access_times[cache_key] = datetime.now(timezone.utc)
        self._stats["hits"] += 1
        
        logging.debug(f"Cache hit for key: {cache_key[:16]}... (age: {entry.age_seconds():.1f}s)")
        
        return entry.result
    
    def set(
        self,
        cache_key: str,
        result: ValidationResult,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store validation result in cache.
        
        Args:
            cache_key: Cache key for storage
            result: ValidationResult to cache
            ttl_seconds: TTL for this entry (overrides default)
            metadata: Additional metadata for the cache entry
        """
        self._stats["total_sets"] += 1
        
        # Use provided TTL or default
        entry_ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        
        # Create cache entry
        entry = CacheEntry(
            result=result,
            cache_key=cache_key,
            created_at=datetime.now(timezone.utc),
            ttl_seconds=entry_ttl,
            metadata=metadata
        )
        
        # Check if we need to evict entries
        if len(self._cache) >= self.max_entries:
            self._evict_lru_entries()
        
        # Store in memory
        self._cache[cache_key] = entry
        self._access_times[cache_key] = datetime.now(timezone.utc)
        
        # Store persistently if enabled
        if self.enable_persistence:
            self._save_to_disk(cache_key, entry)
        
        logging.debug(f"Cached result for key: {cache_key[:16]}... (TTL: {entry_ttl}s)")
    
    def generate_cache_key(
        self,
        validation_type: ValidationType,
        expected: Any,
        actual: Any,
        context: Optional[ValidationContext] = None,
        **kwargs
    ) -> str:
        """
        Generate deterministic cache key from validation inputs.
        
        Args:
            validation_type: Type of validation
            expected: Expected value
            actual: Actual value
            context: Validation context
            **kwargs: Additional parameters
            
        Returns:
            Deterministic cache key string
        """
        # Create deterministic representation of inputs
        cache_data = {
            "validation_type": validation_type.value,
            "expected": self._serialize_for_key(expected),
            "actual": self._serialize_for_key(actual),
            "kwargs": {k: self._serialize_for_key(v) for k, v in sorted(kwargs.items())}
        }
        
        # Include relevant context data (but not dynamic fields like timestamps)
        if context:
            context_data = {
                "validation_type": context.validation_type.value,
                "metadata": context.metadata,
                "configuration": context.configuration,
                # Exclude dynamic fields
                # "validation_id", "correlation_id", "timestamp"
            }
            cache_data["context"] = context_data
        
        # Generate hash
        cache_json = json.dumps(cache_data, sort_keys=True, default=str)
        cache_hash = hashlib.sha256(cache_json.encode()).hexdigest()
        
        return f"{validation_type.value}_{cache_hash[:16]}"
    
    def invalidate(self, cache_key: str) -> bool:
        """
        Invalidate a specific cache entry.
        
        Args:
            cache_key: Cache key to invalidate
            
        Returns:
            True if entry was found and removed, False otherwise
        """
        if cache_key in self._cache:
            self._remove_entry(cache_key)
            return True
        
        # Also remove from persistent storage
        if self.enable_persistence:
            cache_file = self.cache_directory / f"{cache_key}.cache"
            if cache_file.exists():
                cache_file.unlink()
                return True
        
        return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._access_times.clear()
        
        # Clear persistent storage
        if self.enable_persistence and self.cache_directory:
            for cache_file in self.cache_directory.glob("*.cache"):
                cache_file.unlink()
        
        logging.info("Validation cache cleared")
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries from cache.
        
        Returns:
            Number of entries removed
        """
        expired_keys = []
        
        for key, entry in self._cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_entry(key)
        
        self._stats["expired_entries"] += len(expired_keys)
        
        if expired_keys:
            logging.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0.0
        
        return {
            **self._stats,
            "hit_rate": hit_rate,
            "cache_size": len(self._cache),
            "max_entries": self.max_entries,
            "memory_usage_mb": self._estimate_memory_usage() / (1024 * 1024),
            "oldest_entry_age": self._get_oldest_entry_age(),
            "default_ttl_seconds": self.default_ttl_seconds,
            "persistent_storage": self.enable_persistence
        }
    
    def get_cache_info(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific cache entry."""
        entry = self._cache.get(cache_key)
        if entry:
            return entry.to_dict()
        return None
    
    def list_cache_keys(self, validation_type: Optional[ValidationType] = None) -> List[str]:
        """List all cache keys, optionally filtered by validation type."""
        keys = list(self._cache.keys())
        
        if validation_type:
            prefix = f"{validation_type.value}_"
            keys = [key for key in keys if key.startswith(prefix)]
        
        return sorted(keys)
    
    # Private methods
    
    def _serialize_for_key(self, obj: Any, _seen: Optional[Set] = None) -> Any:
        """Serialize object for deterministic key generation."""
        if _seen is None:
            _seen = set()
        
        # Prevent recursion by tracking object IDs
        obj_id = id(obj)
        if obj_id in _seen:
            return f"<recursive_ref_{obj_id}>"
        
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, (list, tuple)):
            _seen.add(obj_id)
            result = [self._serialize_for_key(item, _seen) for item in obj]
            _seen.remove(obj_id)
            return result
        elif isinstance(obj, dict):
            _seen.add(obj_id)
            result = {k: self._serialize_for_key(v, _seen) for k, v in sorted(obj.items())}
            _seen.remove(obj_id)
            return result
        elif hasattr(obj, '__dict__'):
            _seen.add(obj_id)
            result = {k: self._serialize_for_key(v, _seen) for k, v in sorted(obj.__dict__.items())}
            _seen.remove(obj_id)
            return result
        else:
            return str(obj)
    
    def _evict_lru_entries(self) -> None:
        """Evict least recently used entries to make space."""
        if not self._access_times:
            return
        
        # Find least recently used entry
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        self._remove_entry(lru_key)
        self._stats["evictions"] += 1
        
        logging.debug(f"Evicted LRU cache entry: {lru_key[:16]}...")
    
    def _remove_entry(self, cache_key: str) -> None:
        """Remove entry from both memory and persistent storage."""
        # Remove from memory
        self._cache.pop(cache_key, None)
        self._access_times.pop(cache_key, None)
        
        # Remove from persistent storage
        if self.enable_persistence and self.cache_directory:
            cache_file = self.cache_directory / f"{cache_key}.cache"
            if cache_file.exists():
                cache_file.unlink()
    
    def _save_to_disk(self, cache_key: str, entry: CacheEntry) -> None:
        """Save cache entry to persistent storage."""
        if not self.cache_directory:
            return
        
        try:
            cache_file = self.cache_directory / f"{cache_key}.cache"
            
            # Prepare data for serialization
            data = {
                "result_dict": entry.result.to_dict(),
                "cache_key": entry.cache_key,
                "created_at": entry.created_at.isoformat(),
                "ttl_seconds": entry.ttl_seconds,
                "metadata": entry.metadata
            }
            
            if self.enable_compression:
                # Use pickle with compression
                import gzip
                with gzip.open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
            else:
                # Use JSON
                with open(cache_file, 'w') as f:
                    json.dump(data, f)
                    
        except Exception as e:
            logging.warning(f"Failed to save cache entry to disk: {e}")
    
    def _load_from_disk(self, cache_key: str) -> Optional[CacheEntry]:
        """Load cache entry from persistent storage."""
        if not self.cache_directory:
            return None
        
        cache_file = self.cache_directory / f"{cache_key}.cache"
        if not cache_file.exists():
            return None
        
        try:
            if self.enable_compression:
                # Load pickle with compression
                import gzip
                with gzip.open(cache_file, 'rb') as f:
                    data = pickle.load(f)
            else:
                # Load JSON
                with open(cache_file, 'r') as f:
                    data = json.load(f)
            
            # Reconstruct ValidationResult from dict
            result_dict = data["result_dict"]
            
            # Reconstruct objects
            context_data = result_dict.get("context", {})
            context = ValidationContext(
                validation_id=context_data.get("validation_id", "unknown"),
                correlation_id=context_data.get("correlation_id"),
                validation_type=ValidationType(context_data.get("validation_type", "custom")),
                timestamp=datetime.fromisoformat(context_data.get("timestamp", datetime.now(timezone.utc).isoformat())),
                metadata=context_data.get("metadata", {}),
                configuration=context_data.get("configuration", {})
            )
            
            confidence_data = result_dict.get("confidence_score", {})
            confidence_score = ConfidenceScore(
                value=confidence_data.get("value", 0.0),
                components=confidence_data.get("components", {}),
                method=confidence_data.get("method", "unknown"),
                reliability=confidence_data.get("reliability", 1.0)
            )
            
            result = ValidationResult(
                status=ValidationStatus(result_dict.get("status", "failed")),
                confidence_score=confidence_score,
                validation_type=ValidationType(result_dict.get("validation_type", "custom")),
                context=context,
                expected_value=result_dict.get("expected_value"),
                actual_value=result_dict.get("actual_value"),
                message=result_dict.get("message", ""),
                details=result_dict.get("details", {}),
                errors=result_dict.get("errors", []),
                warnings=result_dict.get("warnings", []),
                execution_time_ms=result_dict.get("execution_time_ms"),
                retry_count=result_dict.get("retry_count", 0)
            )
            
            # Create cache entry
            entry = CacheEntry(
                result=result,
                cache_key=data["cache_key"],
                created_at=datetime.fromisoformat(data["created_at"]),
                ttl_seconds=data["ttl_seconds"],
                metadata=data["metadata"]
            )
            
            return entry
            
        except Exception as e:
            logging.warning(f"Failed to load cache entry from disk: {e}")
            # Remove corrupted file
            try:
                cache_file.unlink()
            except:
                pass
            return None
    
    def _load_persistent_cache(self) -> None:
        """Load all cache entries from persistent storage."""
        if not self.cache_directory or not self.cache_directory.exists():
            return
        
        loaded_count = 0
        for cache_file in self.cache_directory.glob("*.cache"):
            cache_key = cache_file.stem
            entry = self._load_from_disk(cache_key)
            
            if entry and not entry.is_expired():
                self._cache[cache_key] = entry
                self._access_times[cache_key] = entry.created_at
                loaded_count += 1
        
        if loaded_count > 0:
            logging.info(f"Loaded {loaded_count} cache entries from persistent storage")
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage in bytes."""
        try:
            import sys
            total_size = 0
            
            for entry in self._cache.values():
                # Rough estimation
                total_size += sys.getsizeof(entry.result.to_dict())
                total_size += sys.getsizeof(entry.cache_key)
                total_size += sys.getsizeof(entry.metadata)
            
            return total_size
        except:
            return 0
    
    def _get_oldest_entry_age(self) -> Optional[float]:
        """Get age of oldest entry in seconds."""
        if not self._cache:
            return None
        
        oldest_time = min(entry.created_at for entry in self._cache.values())
        return (datetime.now(timezone.utc) - oldest_time).total_seconds()