"""
Semantic Message History Module
Provide context management for AI agents with semantic message retrieval.
"""

import time
import uuid
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict

from redis import Redis
from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema
from redisvl.query import VectorQuery
from redisvl.utils.vectorize import BaseTextVectorizer, HFTextVectorizer


@dataclass
class Message:
    """Represents a single message in conversation history."""
    id: str
    session_id: str
    role: str  # 'user', 'assistant', 'system'
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['metadata'] = str(data['metadata'])
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        if isinstance(data.get('metadata'), str):
            import json
            try:
                data['metadata'] = json.loads(data['metadata'])
            except:
                data['metadata'] = {}
        return cls(**data)


class SemanticMessageHistory:
    """
    Manage conversation history with semantic search capabilities.
    
    Features:
    - Persistent message storage with Redis
    - Semantic search over conversation history
    - Session-based organization
    - Context retrieval based on semantic similarity
    - Message pruning and TTL support
    
    Example:
        >>> from redisvl.extensions.memory import SemanticMessageHistory
        >>> 
        >>> # Initialize message history
        >>> history = SemanticMessageHistory(
        ...     name="chat_history",
        ...     redis_url="redis://localhost:6379"
        ... )
        >>> 
        >>> # Add messages
        >>> history.add_message(
        ...     session_id="user_123",
        ...     role="user",
        ...     content="What is machine learning?"
        ... )
        >>> 
        >>> # Get recent messages
        >>> recent = history.get_recent_messages("user_123", limit=5)
        >>> 
        >>> # Search semantically
        >>> relevant = history.search_similar(
        ...     session_id="user_123",
        ...     query="AI algorithms",
        ...     top_k=3
        ... )
    """
    
    def __init__(
        self,
        name: str = "message_history",
        prefix: str = "msg",
        ttl: int = 0,
        redis_url: str = "redis://localhost:6379",
        distance_metric: str = "cosine",
        algorithm: str = "flat",
        vectorizer: Optional[BaseTextVectorizer] = None,
        **kwargs
    ):
        """
        Initialize semantic message history.
        
        Args:
            name: Name of the Redis index
            prefix: Key prefix for messages
            ttl: Time-to-live in seconds (0 means no expiration)
            redis_url: Redis connection URL
            distance_metric: Distance metric for vector search
            algorithm: Vector indexing algorithm (flat, hnsw)
            vectorizer: Text vectorizer for embedding messages
            **kwargs: Additional arguments for Redis connection
        """
        self.name = name
        self.prefix = prefix
        self.ttl = ttl
        self.redis_url = redis_url
        self.distance_metric = distance_metric
        self.algorithm = algorithm
        
        # Initialize vectorizer
        self.vectorizer = vectorizer or HFTextVectorizer()
        
        # Connect to Redis
        self.redis_client = Redis.from_url(redis_url, **kwargs)
        
        # Create schema
        self.schema = self._create_schema()
        
        # Initialize index
        self.index = SearchIndex(schema=self.schema, redis_client=self.redis_client)
        self.index.create(overwrite=False)
    
    def _create_schema(self) -> IndexSchema:
        """Create the index schema for message history."""
        return IndexSchema.from_dict({
            "index": {
                "name": self.name,
                "prefix": f"{self.prefix}:hist",
                "storage_type": "json"
            },
            "fields": [
                {
                    "name": "session_id",
                    "type": "tag",
                    "attrs": {
                        "sortable": True
                    }
                },
                {
                    "name": "role",
                    "type": "tag"
                },
                {
                    "name": "content",
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
                    "name": "timestamp",
                    "type": "numeric",
                    "attrs": {
                        "sortable": True
                    }
                }
            ]
        })
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        skip_embedding: bool = False
    ) -> str:
        """
        Add a message to conversation history.
        
        Args:
            session_id: Unique session identifier
            role: Message role ('user', 'assistant', 'system')
            content: Message content
            metadata: Optional metadata
            skip_embedding: Skip embedding generation (for system messages)
            
        Returns:
            Message ID
        """
        message_id = str(uuid.uuid4())
        key = f"{self.prefix}:hist:{session_id}:{message_id}"
        
        # Generate embedding for user/assistant messages
        embedding = None
        if not skip_embedding and role in ['user', 'assistant']:
            try:
                embedding = self.vectorizer.embed(text=content)
                
                # Update schema dims if needed
                dims = len(embedding)
                if self.schema.fields["embedding"].attrs.dims != dims:
                    self.schema.fields["embedding"].attrs.dims = dims
                    self.index = SearchIndex(schema=self.schema, redis_client=self.redis_client)
                    self.index.create(overwrite=True)
            except Exception as e:
                print(f"Error generating embedding: {e}")
        
        message = Message(
            id=message_id,
            session_id=session_id,
            role=role,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            timestamp=time.time()
        )
        
        # Store in Redis
        self.redis_client.json().set(key, "$", message.to_dict())
        
        # Set TTL if configured
        if self.ttl > 0:
            self.redis_client.expire(key, self.ttl)
        
        return message_id
    
    def get_recent_messages(
        self,
        session_id: str,
        limit: int = 10,
        roles: Optional[List[str]] = None
    ) -> List[Message]:
        """
        Get most recent messages from a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages to return
            roles: Filter by roles (None means all roles)
            
        Returns:
            List of messages sorted by timestamp (newest first)
        """
        # Use Redis SORT to get recent messages
        pattern = f"{self.prefix}:hist:{session_id}:*"
        keys = self.redis_client.keys(pattern)
        
        messages = []
        for key in keys:
            try:
                data = self.redis_client.json().get(key)
                if data:
                    msg = Message.from_dict(data)
                    
                    # Filter by roles if specified
                    if roles and msg.role not in roles:
                        continue
                    
                    messages.append(msg)
            except Exception as e:
                print(f"Error retrieving message {key}: {e}")
        
        # Sort by timestamp (newest first) and limit
        messages.sort(key=lambda x: x.timestamp, reverse=True)
        
        return messages[:limit]
    
    def search_similar(
        self,
        session_id: Optional[str] = None,
        query: str = "",
        top_k: int = 5,
        distance_threshold: float = 1.0,
        roles: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for semantically similar messages.
        
        Args:
            session_id: Optional session filter (None searches all sessions)
            query: Search query
            top_k: Number of results to return
            distance_threshold: Maximum distance for matches
            roles: Filter by roles
            
        Returns:
            List of matching messages with similarity scores
        """
        # Generate embedding for query
        query_embedding = self.vectorizer.embed(text=query)
        
        # Build filter expression
        from redisvl.query.filter import Tag
        
        filter_parts = []
        if session_id:
            filter_parts.append(Tag("session_id") == session_id)
        
        if roles:
            role_filter = None
            for role in roles:
                role_condition = Tag("role") == role
                role_filter = role_condition if role_filter is None else role_filter | role_condition
            filter_parts.append(role_filter)
        
        filter_expression = None
        if filter_parts:
            filter_expression = filter_parts[0]
            for part in filter_parts[1:]:
                filter_expression = filter_expression & part
        
        # Build vector query
        vector_query = VectorQuery(
            vector=query_embedding,
            vector_field_name="embedding",
            filter_expression=filter_expression,
            num_results=top_k,
            return_fields=[
                "session_id",
                "role",
                "content",
                "metadata",
                "timestamp",
                "vector_score"
            ]
        )
        
        # Execute search
        results = self.index.query(vector_query)
        
        # Filter by distance threshold
        matches = []
        for result in results:
            distance = float(result.get("vector_score", 2.0))
            
            if distance <= distance_threshold:
                matches.append({
                    "session_id": result.get("session_id"),
                    "role": result.get("role"),
                    "content": result.get("content"),
                    "similarity": 1.0 - (distance / 2.0),
                    "distance": distance,
                    "timestamp": float(result.get("timestamp", 0)),
                    "metadata": result.get("metadata", "{}")
                })
        
        return matches
    
    def get_context_for_prompt(
        self,
        session_id: str,
        current_prompt: str,
        max_messages: int = 5,
        distance_threshold: float = 1.0
    ) -> List[Dict[str, str]]:
        """
        Get relevant conversation context for current prompt.
        
        Combines recent messages with semantically relevant messages.
        
        Args:
            session_id: Session identifier
            current_prompt: Current user prompt
            max_messages: Maximum context messages to return
            distance_threshold: Threshold for semantic similarity
            
        Returns:
            List of message dicts formatted for LLM context
        """
        # Get recent messages
        recent = self.get_recent_messages(session_id, limit=max_messages)
        
        # Get semantically similar messages
        similar = self.search_similar(
            session_id=session_id,
            query=current_prompt,
            top_k=max_messages,
            distance_threshold=distance_threshold
        )
        
        # Combine and deduplicate
        seen_ids = set()
        context = []
        
        # Add recent messages first
        for msg in recent:
            if msg.id not in seen_ids:
                context.append({
                    "role": msg.role,
                    "content": msg.content
                })
                seen_ids.add(msg.id)
        
        # Add similar messages
        for sim in similar:
            msg_key = f"{sim['session_id']}_{sim['timestamp']}"
            if msg_key not in seen_ids:
                context.append({
                    "role": sim['role'],
                    "content": sim['content']
                })
                seen_ids.add(msg_key)
        
        # Limit to max_messages
        return context[:max_messages]
    
    def delete_session(self, session_id: str) -> int:
        """
        Delete all messages for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Number of messages deleted
        """
        pattern = f"{self.prefix}:hist:{session_id}:*"
        keys = self.redis_client.keys(pattern)
        
        if keys:
            count = self.redis_client.delete(*keys)
            return count
        return 0
    
    def delete_message(self, session_id: str, message_id: str) -> bool:
        """
        Delete a specific message.
        
        Args:
            session_id: Session identifier
            message_id: Message identifier
            
        Returns:
            True if deleted, False otherwise
        """
        key = f"{self.prefix}:hist:{session_id}:{message_id}"
        
        try:
            result = self.redis_client.delete(key)
            return result > 0
        except Exception as e:
            print(f"Error deleting message: {e}")
            return False
    
    def clear_all(self) -> int:
        """
        Clear all message history.
        
        Returns:
            Number of messages cleared
        """
        pattern = f"{self.prefix}:hist:*"
        keys = self.redis_client.keys(pattern)
        
        if keys:
            count = self.redis_client.delete(*keys)
            return count
        return 0
    
    def get_session_ids(self) -> List[str]:
        """
        Get all active session IDs.
        
        Returns:
            List of session IDs
        """
        pattern = f"{self.prefix}:hist:*"
        keys = self.redis_client.keys(pattern)
        
        session_ids = set()
        for key in keys:
            parts = key.split(":")
            if len(parts) >= 3:
                session_ids.add(parts[2])
        
        return list(session_ids)
    
    def stats(self) -> Dict[str, Any]:
        """
        Get message history statistics.
        
        Returns:
            Dictionary with statistics
        """
        pattern = f"{self.prefix}:hist:*"
        keys = self.redis_client.keys(pattern)
        
        session_ids = self.get_session_ids()
        
        return {
            "total_messages": len(keys),
            "active_sessions": len(session_ids),
            "session_ids": session_ids,
            "index_name": self.name,
            "ttl": self.ttl
        }
