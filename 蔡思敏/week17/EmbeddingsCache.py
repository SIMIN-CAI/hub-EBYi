"""
Embeddings Cache Module
Reduce computational costs and improve performance by caching embedding vectors.
"""

import hashlib
import json
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict

from redis import Redis
from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema
from redisvl.utils.vectorize import BaseTextVectorizer


@dataclass
class EmbeddingCacheEntry:
    """Represents a single embedding cache entry."""
    id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]
    created_at: float
    updated_at: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingCacheEntry":
        return cls(**data)


class EmbeddingsCache:
    """
    Cache embeddings to reduce computational costs and improve performance.
    
    Features:
    - Automatic embedding caching with TTL support
    - Efficient vector storage and retrieval
    - Metadata association with cached embeddings
    - Batch operations for bulk caching
    
    Example:
        >>> from redisvl.extensions.cache.embeddings import EmbeddingsCache
        >>> from redisvl.utils.vectorize import HFTextVectorizer
        >>> 
        >>> # Initialize embedding cache
        >>> embed_cache = EmbeddingsCache(
        ...     name="embed_cache",
        ...     ttl=3600,
        ...     redis_url="redis://localhost:6379"
        ... )
        >>> 
        >>> # Initialize vectorizer
        >>> vectorizer = HFTextVectorizer()
        >>> 
        >>> # Get or compute embedding
        >>> embedding = embed_cache.get_or_compute(
        ...     text="What is the capital of France?",
        ...     vectorizer=vectorizer
        ... )
    """
    
    def __init__(
        self,
        name: str = "embeddings_cache",
        prefix: str = "emb",
        ttl: int = 3600,
        redis_url: str = "redis://localhost:6379",
        distance_metric: str = "cosine",
        algorithm: str = "flat",
        **kwargs
    ):
        """
        Initialize the embeddings cache.
        
        Args:
            name: Name of the Redis index
            prefix: Key prefix for cache entries
            ttl: Time-to-live in seconds (0 means no expiration)
            redis_url: Redis connection URL
            distance_metric: Distance metric for vector search (cosine, euclidean, dotproduct)
            algorithm: Vector indexing algorithm (flat, hnsw)
            **kwargs: Additional arguments for Redis connection
        """
        self.name = name
        self.prefix = prefix
        self.ttl = ttl
        self.redis_url = redis_url
        self.distance_metric = distance_metric
        self.algorithm = algorithm
        
        # Connect to Redis
        self.redis_client = Redis.from_url(redis_url, **kwargs)
        
        # Create schema
        self.schema = self._create_schema()
        
        # Initialize index
        self.index = SearchIndex(schema=self.schema, redis_client=self.redis_client)
        self.index.create(overwrite=False)
    
    def _create_schema(self) -> IndexSchema:
        """Create the index schema for embedding cache."""
        return IndexSchema.from_dict({
            "index": {
                "name": self.name,
                "prefix": f"{self.prefix}:emb",
                "storage_type": "json"
            },
            "fields": [
                {
                    "name": "text",
                    "type": "text",
                    "attrs": {
                        "sortable": True
                    }
                },
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "algorithm": self.algorithm,
                        "datatype": "float32",
                        "dims": 768,  # Will be updated dynamically
                        "distance_metric": self.distance_metric
                    }
                },
                {
                    "name": "metadata",
                    "type": "tag"
                },
                {
                    "name": "created_at",
                    "type": "numeric",
                    "attrs": {
                        "sortable": True
                    }
                },
                {
                    "name": "updated_at",
                    "type": "numeric",
                    "attrs": {
                        "sortable": True
                    }
                }
            ]
        })
    
    def _generate_id(self, text: str) -> str:
        """Generate a unique ID for the text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get(self, text: str) -> Optional[List[float]]:
        """
        Retrieve cached embedding for given text.
        
        Args:
            text: Input text
            
        Returns:
            Cached embedding if found, None otherwise
        """
        cache_id = self._generate_id(text)
        key = f"{self.prefix}:emb:{cache_id}"
        
        try:
            data = self.redis_client.json().get(key)
            if data:
                return data.get("embedding")
        except Exception as e:
            print(f"Error retrieving from cache: {e}")
        
        return None
    
    def set(
        self,
        text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Cache an embedding for given text.
        
        Args:
            text: Input text
            embedding: Embedding vector
            metadata: Optional metadata to associate
            
        Returns:
            Cache entry ID
        """
        cache_id = self._generate_id(text)
        key = f"{self.prefix}:emb:{cache_id}"
        
        now = time.time()
        entry = EmbeddingCacheEntry(
            id=cache_id,
            text=text,
            embedding=embedding,
            metadata=metadata or {},
            created_at=now,
            updated_at=now
        )
        
        # Update schema dims if needed
        dims = len(embedding)
        if self.schema.fields["embedding"].attrs.dims != dims:
            self.schema.fields["embedding"].attrs.dims = dims
            self.index = SearchIndex(schema=self.schema, redis_client=self.redis_client)
            self.index.create(overwrite=True)
        
        # Store in Redis
        self.redis_client.json().set(key, "$", entry.to_dict())
        
        # Set TTL if configured
        if self.ttl > 0:
            self.redis_client.expire(key, self.ttl)
        
        return cache_id
    
    def get_or_compute(
        self,
        text: str,
        vectorizer: BaseTextVectorizer,
        metadata: Optional[Dict[str, Any]] = None,
        force_recompute: bool = False
    ) -> List[float]:
        """
        Get cached embedding or compute using vectorizer.
        
        Args:
            text: Input text
            vectorizer: Text vectorizer instance
            metadata: Optional metadata
            force_recompute: Force recomputation even if cached
            
        Returns:
            Embedding vector
        """
        if not force_recompute:
            cached = self.get(text)
            if cached is not None:
                return cached
        
        # Compute new embedding
        embedding = vectorizer.embed(text=text)
        self.set(text, embedding, metadata)
        
        return embedding
    
    def batch_get(self, texts: List[str]) -> Dict[str, Optional[List[float]]]:
        """
        Batch retrieve embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Dictionary mapping texts to their embeddings (None if not cached)
        """
        results = {}
        for text in texts:
            results[text] = self.get(text)
        return results
    
    def batch_set(
        self,
        text_embedding_pairs: List[tuple],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Batch cache multiple embeddings.
        
        Args:
            text_embedding_pairs: List of (text, embedding) tuples
            metadatas: Optional list of metadata dicts
            
        Returns:
            List of cache entry IDs
        """
        if metadatas is None:
            metadatas = [{}] * len(text_embedding_pairs)
        
        ids = []
        for (text, embedding), metadata in zip(text_embedding_pairs, metadatas):
            cache_id = self.set(text, embedding, metadata)
            ids.append(cache_id)
        
        return ids
    
    def delete(self, text: str) -> bool:
        """
        Delete cached embedding for given text.
        
        Args:
            text: Input text
            
        Returns:
            True if deleted, False otherwise
        """
        cache_id = self._generate_id(text)
        key = f"{self.prefix}:emb:{cache_id}"
        
        try:
            result = self.redis_client.delete(key)
            return result > 0
        except Exception as e:
            print(f"Error deleting from cache: {e}")
            return False
    
    def clear(self) -> int:
        """
        Clear all cached embeddings.
        
        Returns:
            Number of entries cleared
        """
        pattern = f"{self.prefix}:emb:*"
        keys = self.redis_client.keys(pattern)
        
        if keys:
            count = self.redis_client.delete(*keys)
            return count
        return 0
    
    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        pattern = f"{self.prefix}:emb:*"
        keys = self.redis_client.keys(pattern)
        
        return {
            "total_entries": len(keys),
            "index_name": self.name,
            "prefix": self.prefix,
            "ttl": self.ttl,
            "distance_metric": self.distance_metric
        }
