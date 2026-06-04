"""
Semantic Cache Module
Increase application throughput and reduce LLM costs by leveraging previously generated knowledge.
"""

import hashlib
import json
import time
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict

from redis import Redis
from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema
from redisvl.query import VectorQuery
from redisvl.utils.vectorize import BaseTextVectorizer, HFTextVectorizer


@dataclass
class SemanticCacheEntry:
    """Represents a single semantic cache entry."""
    id: str
    prompt: str
    response: str
    prompt_embedding: List[float]
    metadata: Dict[str, Any]
    timestamp: float
    hit_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['metadata'] = json.dumps(data['metadata'])
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticCacheEntry":
        if isinstance(data.get('metadata'), str):
            data['metadata'] = json.loads(data['metadata'])
        return cls(**data)


class SemanticCache:
    """
    Semantic cache for LLM responses to reduce costs and improve throughput.
    
    Features:
    - Semantic similarity-based caching
    - Configurable distance threshold for cache hits
    - TTL support for automatic cache expiration
    - Hit tracking and statistics
    - Batch operations
    
    Example:
        >>> from redisvl.extensions.cache.llm import SemanticCache
        >>> 
        >>> # Initialize cache with TTL and semantic distance threshold
        >>> llmcache = SemanticCache(
        ...     name="llmcache",
        ...     ttl=360,
        ...     redis_url="redis://localhost:6379",
        ...     distance_threshold=0.1  # Lower is stricter
        ... )
        >>> 
        >>> # Store user queries and LLM responses
        >>> llmcache.store(
        ...     prompt="What is the capital city of France?",
        ...     response="Paris"
        ... )
        >>> 
        >>> # Check cache with similar prompt
        >>> response = llmcache.check(prompt="What is France's capital city?")
        >>> print(response[0]["response"])
        Paris
    """
    
    def __init__(
        self,
        name: str = "llm_cache",
        prefix: str = "llm",
        ttl: int = 3600,
        redis_url: str = "redis://localhost:6379",
        distance_threshold: float = 0.1,
        distance_metric: str = "cosine",
        algorithm: str = "flat",
        vectorizer: Optional[BaseTextVectorizer] = None,
        **kwargs
    ):
        """
        Initialize the semantic cache.
        
        Args:
            name: Name of the Redis index
            prefix: Key prefix for cache entries
            ttl: Time-to-live in seconds (0 means no expiration)
            redis_url: Redis connection URL
            distance_threshold: Maximum distance for cache hit (COSINE: 0-2, lower is stricter)
            distance_metric: Distance metric (cosine, euclidean, dotproduct)
            algorithm: Vector indexing algorithm (flat, hnsw)
            vectorizer: Text vectorizer for embedding prompts
            **kwargs: Additional arguments for Redis connection
        """
        self.name = name
        self.prefix = prefix
        self.ttl = ttl
        self.redis_url = redis_url
        self.distance_threshold = distance_threshold
        self.distance_metric = distance_metric
        self.algorithm = algorithm
        
        # Initialize vectorizer (default to HuggingFace)
        self.vectorizer = vectorizer or HFTextVectorizer()
        
        # Connect to Redis
        self.redis_client = Redis.from_url(redis_url, **kwargs)
        
        # Create schema
        self.schema = self._create_schema()
        
        # Initialize index
        self.index = SearchIndex(schema=self.schema, redis_client=self.redis_client)
        self.index.create(overwrite=False)
    
    def _create_schema(self) -> IndexSchema:
        """Create the index schema for semantic cache."""
        return IndexSchema.from_dict({
            "index": {
                "name": self.name,
                "prefix": f"{self.prefix}:cache",
                "storage_type": "json"
            },
            "fields": [
                {
                    "name": "prompt",
                    "type": "text",
                    "attrs": {
                        "sortable": True
                    }
                },
                {
                    "name": "response",
                    "type": "text"
                },
                {
                    "name": "prompt_embedding",
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
                    "name": "timestamp",
                    "type": "numeric",
                    "attrs": {
                        "sortable": True
                    }
                },
                {
                    "name": "hit_count",
                    "type": "numeric",
                    "attrs": {
                        "sortable": True
                    }
                }
            ]
        })
    
    def _generate_id(self, prompt: str) -> str:
        """Generate a unique ID for the prompt."""
        return hashlib.md5(prompt.encode('utf-8')).hexdigest()
    
    def store(
        self,
        prompt: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a prompt-response pair in the semantic cache.
        
        Args:
            prompt: User's question/prompt
            response: LLM's response
            metadata: Optional metadata to associate
            
        Returns:
            Cache entry ID
        """
        # Generate embedding for prompt
        prompt_embedding = self.vectorizer.embed(text=prompt)
        
        cache_id = self._generate_id(prompt)
        key = f"{self.prefix}:cache:{cache_id}"
        
        now = time.time()
        entry = SemanticCacheEntry(
            id=cache_id,
            prompt=prompt,
            response=response,
            prompt_embedding=prompt_embedding,
            metadata=metadata or {},
            timestamp=now,
            hit_count=0
        )
        
        # Update schema dims if needed
        dims = len(prompt_embedding)
        if self.schema.fields["prompt_embedding"].attrs.dims != dims:
            self.schema.fields["prompt_embedding"].attrs.dims = dims
            self.index = SearchIndex(schema=self.schema, redis_client=self.redis_client)
            self.index.create(overwrite=True)
        
        # Store in Redis
        self.redis_client.json().set(key, "$", entry.to_dict())
        
        # Set TTL if configured
        if self.ttl > 0:
            self.redis_client.expire(key, self.ttl)
        
        return cache_id
    
    def check(
        self,
        prompt: str,
        num_results: int = 1
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Check if a semantically similar prompt exists in cache.
        
        Args:
            prompt: User's question/prompt to check
            num_results: Number of similar results to return
            
        Returns:
            List of cache hits with response and similarity score, or None if no match
        """
        # Generate embedding for the query prompt
        prompt_embedding = self.vectorizer.embed(text=prompt)
        
        # Build vector query
        query = VectorQuery(
            vector=prompt_embedding,
            vector_field_name="prompt_embedding",
            num_results=num_results,
            return_fields=[
                "prompt",
                "response",
                "metadata",
                "timestamp",
                "hit_count",
                "vector_score"
            ],
            filter_expression=None
        )
        
        # Execute search
        results = self.index.query(query)
        
        if not results:
            return None
        
        # Filter by distance threshold
        cache_hits = []
        for result in results:
            distance = float(result.get("vector_score", 2.0))
            
            if distance <= self.distance_threshold:
                # Increment hit count
                cache_id = result.get("id")
                if cache_id:
                    self._increment_hit_count(cache_id)
                
                cache_hits.append({
                    "prompt": result.get("prompt"),
                    "response": result.get("response"),
                    "similarity": 1.0 - (distance / 2.0),  # Convert distance to similarity
                    "distance": distance,
                    "metadata": json.loads(result.get("metadata", "{}")),
                    "timestamp": float(result.get("timestamp", 0)),
                    "hit_count": int(result.get("hit_count", 0)) + 1
                })
        
        return cache_hits if cache_hits else None
    
    def _increment_hit_count(self, cache_id: str):
        """Increment the hit count for a cache entry."""
        key = f"{self.prefix}:cache:{cache_id}"
        
        try:
            current = self.redis_client.json().get(key, "$.hit_count")
            if current and len(current) > 0:
                new_count = current[0] + 1
                self.redis_client.json().set(key, "$.hit_count", new_count)
        except Exception as e:
            print(f"Error incrementing hit count: {e}")
    
    def get_or_generate(
        self,
        prompt: str,
        generator_func,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Check cache first, if miss then generate response and cache it.
        
        Args:
            prompt: User's question/prompt
            generator_func: Function to call if cache miss (should return response string)
            metadata: Optional metadata
            
        Returns:
            Dictionary with response and cache info
        """
        # Check cache
        cache_hits = self.check(prompt)
        
        if cache_hits:
            return {
                "response": cache_hits[0]["response"],
                "source": "cache",
                "similarity": cache_hits[0]["similarity"],
                "original_prompt": cache_hits[0]["prompt"]
            }
        
        # Cache miss - generate response
        response = generator_func(prompt)
        
        # Store in cache
        self.store(prompt, response, metadata)
        
        return {
            "response": response,
            "source": "generated",
            "similarity": None
        }
    
    def batch_store(
        self,
        prompt_response_pairs: List[Tuple[str, str]],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Batch store multiple prompt-response pairs.
        
        Args:
            prompt_response_pairs: List of (prompt, response) tuples
            metadatas: Optional list of metadata dicts
            
        Returns:
            List of cache entry IDs
        """
        if metadatas is None:
            metadatas = [{}] * len(prompt_response_pairs)
        
        ids = []
        for (prompt, response), metadata in zip(prompt_response_pairs, metadatas):
            cache_id = self.store(prompt, response, metadata)
            ids.append(cache_id)
        
        return ids
    
    def delete(self, prompt: str) -> bool:
        """
        Delete cached response for given prompt.
        
        Args:
            prompt: User's question/prompt
            
        Returns:
            True if deleted, False otherwise
        """
        cache_id = self._generate_id(prompt)
        key = f"{self.prefix}:cache:{cache_id}"
        
        try:
            result = self.redis_client.delete(key)
            return result > 0
        except Exception as e:
            print(f"Error deleting from cache: {e}")
            return False
    
    def clear(self) -> int:
        """
        Clear all cached responses.
        
        Returns:
            Number of entries cleared
        """
        pattern = f"{self.prefix}:cache:*"
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
        pattern = f"{self.prefix}:cache:*"
        keys = self.redis_client.keys(pattern)
        
        total_hits = 0
        for key in keys:
            try:
                data = self.redis_client.json().get(key)
                if data:
                    total_hits += data.get("hit_count", 0)
            except:
                pass
        
        return {
            "total_entries": len(keys),
            "total_hits": total_hits,
            "hit_rate": total_hits / max(len(keys), 1),
            "index_name": self.name,
            "distance_threshold": self.distance_threshold,
            "ttl": self.ttl
        }
