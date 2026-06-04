"""
Semantic Router Module
Intelligent query classification and routing based on semantic similarity.
"""

import json
import time
from typing import List, Optional, Dict, Any, Callable, Tuple
from dataclasses import dataclass, asdict

from redis import Redis
from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema
from redisvl.query import VectorQuery
from redisvl.utils.vectorize import BaseTextVectorizer, HFTextVectorizer


@dataclass
class Route:
    """Represents a routing destination."""
    name: str
    description: str
    embedding: List[float]
    handler: Optional[Callable] = None
    metadata: Dict[str, Any] = None
    created_at: float = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop('handler', None)  # Don't serialize handler function
        data['metadata'] = str(data['metadata'])
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Route":
        if isinstance(data.get('metadata'), str):
            import json
            try:
                data['metadata'] = json.loads(data['metadata'])
            except:
                data['metadata'] = {}
        data.pop('handler', None)
        return cls(**data)


@dataclass
class RoutingResult:
    """Result of semantic routing."""
    route_name: str
    confidence: float
    distance: float
    metadata: Dict[str, Any]
    all_candidates: List[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SemanticRouter:
    """
    Intelligent query classification and routing based on semantic similarity.
    
    Features:
    - Semantic-based query classification
    - Dynamic route registration
    - Confidence scoring
    - Fallback routing
    - Multiple candidate routes
    
    Example:
        >>> from redisvl.extensions.router import SemanticRouter
        >>> 
        >>> # Initialize router
        >>> router = SemanticRouter(
        ...     name="query_router",
        ...     redis_url="redis://localhost:6379"
        ... )
        >>> 
        >>> # Register routes
        >>> router.add_route(
        ...     name="customer_support",
        ...     description="Handle customer support questions",
        ...     examples=["How do I reset my password?", "I need help with billing"]
        ... )
        >>> 
        >>> router.add_route(
        ...     name="product_info",
        ...     description="Provide product information",
        ...     examples=["What features does this have?", "Tell me about pricing"]
        ... )
        >>> 
        >>> # Route a query
        >>> result = router.route("How can I change my account password?")
        >>> print(result.route_name)  # customer_support
        >>> print(result.confidence)  # 0.85
    """
    
    def __init__(
        self,
        name: str = "semantic_router",
        prefix: str = "route",
        redis_url: str = "redis://localhost:6379",
        distance_metric: str = "cosine",
        algorithm: str = "flat",
        default_route: Optional[str] = "general",
        confidence_threshold: float = 0.7,
        vectorizer: Optional[BaseTextVectorizer] = None,
        **kwargs
    ):
        """
        Initialize semantic router.
        
        Args:
            name: Name of the Redis index
            prefix: Key prefix for routes
            redis_url: Redis connection URL
            distance_metric: Distance metric for vector search
            algorithm: Vector indexing algorithm (flat, hnsw)
            default_route: Default route name when no confident match
            confidence_threshold: Minimum confidence for routing (0-1)
            vectorizer: Text vectorizer for embedding queries
            **kwargs: Additional arguments for Redis connection
        """
        self.name = name
        self.prefix = prefix
        self.redis_url = redis_url
        self.distance_metric = distance_metric
        self.algorithm = algorithm
        self.default_route = default_route
        self.confidence_threshold = confidence_threshold
        
        # Initialize vectorizer
        self.vectorizer = vectorizer or HFTextVectorizer()
        
        # Connect to Redis
        self.redis_client = Redis.from_url(redis_url, **kwargs)
        
        # Create schema
        self.schema = self._create_schema()
        
        # Initialize index
        self.index = SearchIndex(schema=self.schema, redis_client=self.redis_client)
        self.index.create(overwrite=False)
        
        # Route handlers registry
        self.handlers: Dict[str, Callable] = {}
    
    def _create_schema(self) -> IndexSchema:
        """Create the index schema for semantic router."""
        return IndexSchema.from_dict({
            "index": {
                "name": self.name,
                "prefix": f"{self.prefix}:router",
                "storage_type": "json"
            },
            "fields": [
                {
                    "name": "name",
                    "type": "tag",
                    "attrs": {
                        "sortable": True
                    }
                },
                {
                    "name": "description",
                    "type": "text"
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
                }
            ]
        })
    
    def add_route(
        self,
        name: str,
        description: str,
        examples: Optional[List[str]] = None,
        handler: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a route to the router.
        
        Args:
            name: Route name (unique identifier)
            description: Route description (used for embedding)
            examples: Example queries for this route
            handler: Optional handler function for this route
            metadata: Optional metadata
            
        Returns:
            Route key
        """
        # Create composite text for better embedding
        embedding_text = description
        if examples:
            embedding_text += " " + " ".join(examples)
        
        # Generate embedding
        embedding = self.vectorizer.embed(text=embedding_text)
        
        key = f"{self.prefix}:router:{name}"
        
        route = Route(
            name=name,
            description=description,
            embedding=embedding,
            handler=handler,
            metadata=metadata or {},
            created_at=time.time()
        )
        
        # Update schema dims if needed
        dims = len(embedding)
        if self.schema.fields["embedding"].attrs.dims != dims:
            self.schema.fields["embedding"].attrs.dims = dims
            self.index = SearchIndex(schema=self.schema, redis_client=self.redis_client)
            self.index.create(overwrite=True)
        
        # Store in Redis
        self.redis_client.json().set(key, "$", route.to_dict())
        
        # Register handler if provided
        if handler:
            self.handlers[name] = handler
        
        return key
    
    def add_routes_from_config(self, routes_config: List[Dict[str, Any]]):
        """
        Add multiple routes from configuration.
        
        Args:
            routes_config: List of route configurations
                Each config should have: name, description, examples (optional),
                handler (optional), metadata (optional)
        """
        for config in routes_config:
            self.add_route(
                name=config["name"],
                description=config["description"],
                examples=config.get("examples"),
                handler=config.get("handler"),
                metadata=config.get("metadata")
            )
    
    def route(
        self,
        query: str,
        top_k: int = 3,
        use_default: bool = True
    ) -> RoutingResult:
        """
        Route a query to the most appropriate route.
        
        Args:
            query: Input query to route
            top_k: Number of candidate routes to consider
            use_default: Use default route if confidence is low
            
        Returns:
            RoutingResult with best route and confidence
        """
        # Generate embedding for query
        query_embedding = self.vectorizer.embed(text=query)
        
        # Build vector query
        vector_query = VectorQuery(
            vector=query_embedding,
            vector_field_name="embedding",
            num_results=top_k,
            return_fields=[
                "name",
                "description",
                "metadata",
                "vector_score"
            ]
        )
        
        # Execute search
        results = self.index.query(vector_query)
        
        if not results:
            # No routes found, use default
            if use_default and self.default_route:
                return RoutingResult(
                    route_name=self.default_route,
                    confidence=0.0,
                    distance=2.0,
                    metadata={"fallback": True, "reason": "no_routes"}
                )
            else:
                raise ValueError("No routes configured and no default route set")
        
        # Find best match
        best_match = results[0]
        distance = float(best_match.get("vector_score", 2.0))
        
        # Convert distance to confidence (COSINE: 0-2, lower is better)
        confidence = 1.0 - (distance / 2.0)
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            if use_default and self.default_route:
                return RoutingResult(
                    route_name=self.default_route,
                    confidence=confidence,
                    distance=distance,
                    metadata={
                        "fallback": True,
                        "reason": "low_confidence",
                        "threshold": self.confidence_threshold
                    },
                    all_candidates=self._format_candidates(results)
                )
        
        # Return best match
        return RoutingResult(
            route_name=best_match.get("name"),
            confidence=confidence,
            distance=distance,
            metadata=json.loads(best_match.get("metadata", "{}")),
            all_candidates=self._format_candidates(results)
        )
    
    def _format_candidates(self, results: List[Dict]) -> List[Dict[str, Any]]:
        """Format all candidate routes."""
        candidates = []
        for result in results:
            distance = float(result.get("vector_score", 2.0))
            confidence = 1.0 - (distance / 2.0)
            
            candidates.append({
                "route_name": result.get("name"),
                "description": result.get("description"),
                "confidence": confidence,
                "distance": distance
            })
        
        return candidates
    
    def route_and_execute(
        self,
        query: str,
        **kwargs
    ) -> Any:
        """
        Route query and execute the corresponding handler.
        
        Args:
            query: Input query
            **kwargs: Additional arguments to pass to handler
            
        Returns:
            Result from handler function
        """
        result = self.route(query)
        
        # Get handler
        handler = self.handlers.get(result.route_name)
        
        if handler:
            return handler(query, **kwargs)
        else:
            # No handler registered, return routing result
            return {
                "route": result.route_name,
                "confidence": result.confidence,
                "query": query,
                "fallback": result.metadata.get("fallback", False)
            }
    
    def update_route(
        self,
        name: str,
        description: Optional[str] = None,
        examples: Optional[List[str]] = None,
        handler: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing route.
        
        Args:
            name: Route name to update
            description: New description
            examples: New examples
            handler: New handler function
            metadata: New metadata
            
        Returns:
            True if updated, False if route doesn't exist
        """
        key = f"{self.prefix}:router:{name}"
        
        # Check if route exists
        existing = self.redis_client.json().get(key)
        if not existing:
            return False
        
        # Get current route data
        current_data = existing[0] if isinstance(existing, list) else existing
        
        # Update fields
        if description:
            current_data["description"] = description
        
        if metadata is not None:
            current_data["metadata"] = str(metadata)
        
        # Regenerate embedding if description or examples changed
        if description or examples:
            embedding_text = description or current_data["description"]
            if examples:
                embedding_text += " " + " ".join(examples)
            
            embedding = self.vectorizer.embed(text=embedding_text)
            current_data["embedding"] = embedding
        
        # Update in Redis
        self.redis_client.json().set(key, "$", current_data)
        
        # Update handler if provided
        if handler:
            self.handlers[name] = handler
        
        return True
    
    def remove_route(self, name: str) -> bool:
        """
        Remove a route.
        
        Args:
            name: Route name to remove
            
        Returns:
            True if removed, False if route doesn't exist
        """
        key = f"{self.prefix}:router:{name}"
        
        try:
            result = self.redis_client.delete(key)
            
            # Remove handler if exists
            if name in self.handlers:
                del self.handlers[name]
            
            return result > 0
        except Exception as e:
            print(f"Error removing route: {e}")
            return False
    
    def get_route(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get route details.
        
        Args:
            name: Route name
            
        Returns:
            Route details or None if not found
        """
        key = f"{self.prefix}:router:{name}"
        
        try:
            data = self.redis_client.json().get(key)
            if data:
                route_data = data[0] if isinstance(data, list) else data
                return {
                    "name": route_data.get("name"),
                    "description": route_data.get("description"),
                    "metadata": route_data.get("metadata", "{}"),
                    "has_handler": name in self.handlers
                }
        except Exception as e:
            print(f"Error getting route: {e}")
        
        return None
    
    def list_routes(self) -> List[str]:
        """
        List all route names.
        
        Returns:
            List of route names
        """
        pattern = f"{self.prefix}:router:*"
        keys = self.redis_client.keys(pattern)
        
        route_names = []
        for key in keys:
            parts = key.split(":")
            if len(parts) >= 3:
                route_names.append(parts[2])
        
        return route_names
    
    def clear_all(self) -> int:
        """
        Clear all routes.
        
        Returns:
            Number of routes cleared
        """
        pattern = f"{self.prefix}:router:*"
        keys = self.redis_client.keys(pattern)
        
        if keys:
            count = self.redis_client.delete(*keys)
            self.handlers.clear()
            return count
        return 0
    
    def stats(self) -> Dict[str, Any]:
        """
        Get router statistics.
        
        Returns:
            Dictionary with statistics
        """
        pattern = f"{self.prefix}:router:*"
        keys = self.redis_client.keys(pattern)
        
        return {
            "total_routes": len(keys),
            "registered_handlers": len(self.handlers),
            "route_names": self.list_routes(),
            "index_name": self.name,
            "confidence_threshold": self.confidence_threshold,
            "default_route": self.default_route
        }
