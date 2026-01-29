"""
User Context Protocol (UCP) Embedding Exchange.

Handles audience/targeting context exchange between buyer and seller agents
for the RTB simulation. Embeddings represent user context, audience segments,
and targeting parameters.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import hashlib
import json
import uuid


class EmbeddingFormat(str, Enum):
    """Supported embedding formats."""
    DENSE_VECTOR = "dense_vector"       # Float vector representation
    SEGMENT_IDS = "segment_ids"         # IAB/custom segment identifiers
    SPARSE_FEATURES = "sparse_features" # Feature name -> value mapping


class ContextType(str, Enum):
    """Types of user context."""
    DEMOGRAPHIC = "demographic"
    BEHAVIORAL = "behavioral"
    CONTEXTUAL = "contextual"
    INTENT = "intent"
    FIRST_PARTY = "first_party"


@dataclass
class UserContextEmbedding:
    """
    Represents a user context embedding for audience targeting.

    Embeddings can be dense vectors (ML-based), segment IDs (IAB taxonomy),
    or sparse feature maps depending on the use case.
    """
    embedding_id: str = field(default_factory=lambda: f"emb-{uuid.uuid4().hex[:12]}")
    context_type: ContextType = ContextType.BEHAVIORAL
    format: EmbeddingFormat = EmbeddingFormat.SEGMENT_IDS
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Dense vector representation (ML-based)
    vector: Optional[list[float]] = None
    vector_dim: int = 0

    # Segment ID representation (IAB taxonomy)
    segment_ids: list[str] = field(default_factory=list)

    # Sparse feature representation
    features: dict[str, float] = field(default_factory=dict)

    # Metadata
    source: str = "unknown"  # Where the embedding came from
    confidence: float = 1.0  # Confidence score (0.0-1.0)
    ttl_seconds: int = 3600  # Time-to-live for caching

    def __post_init__(self):
        """Validate embedding based on format."""
        if self.format == EmbeddingFormat.DENSE_VECTOR:
            if self.vector is None or len(self.vector) == 0:
                raise ValueError("Dense vector format requires non-empty vector")
            self.vector_dim = len(self.vector)
        elif self.format == EmbeddingFormat.SEGMENT_IDS:
            if not self.segment_ids:
                self.segment_ids = []  # Allow empty for "no targeting"
        elif self.format == EmbeddingFormat.SPARSE_FEATURES:
            if not self.features:
                self.features = {}

    def fingerprint(self) -> str:
        """Generate a fingerprint hash for deduplication."""
        content = {
            "format": self.format.value,
            "context_type": self.context_type.value,
            "vector": self.vector,
            "segment_ids": sorted(self.segment_ids) if self.segment_ids else None,
            "features": dict(sorted(self.features.items())) if self.features else None,
        }
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        """Serialize to dictionary for message passing."""
        return {
            "embedding_id": self.embedding_id,
            "context_type": self.context_type.value,
            "format": self.format.value,
            "timestamp": self.timestamp.isoformat(),
            "vector": self.vector,
            "vector_dim": self.vector_dim,
            "segment_ids": self.segment_ids,
            "features": self.features,
            "source": self.source,
            "confidence": self.confidence,
            "ttl_seconds": self.ttl_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UserContextEmbedding":
        """Deserialize from dictionary."""
        return cls(
            embedding_id=data["embedding_id"],
            context_type=ContextType(data["context_type"]),
            format=EmbeddingFormat(data["format"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            vector=data.get("vector"),
            vector_dim=data.get("vector_dim", 0),
            segment_ids=data.get("segment_ids", []),
            features=data.get("features", {}),
            source=data.get("source", "unknown"),
            confidence=data.get("confidence", 1.0),
            ttl_seconds=data.get("ttl_seconds", 3600),
        )

    def merge(self, other: "UserContextEmbedding") -> "UserContextEmbedding":
        """
        Merge two embeddings of the same format.

        For segments: union of segment IDs
        For vectors: weighted average by confidence
        For sparse: union with max values
        """
        if self.format != other.format:
            raise ValueError(f"Cannot merge different formats: {self.format} vs {other.format}")

        if self.format == EmbeddingFormat.SEGMENT_IDS:
            merged_segments = list(set(self.segment_ids) | set(other.segment_ids))
            return UserContextEmbedding(
                context_type=self.context_type,
                format=self.format,
                segment_ids=merged_segments,
                confidence=max(self.confidence, other.confidence),
                source=f"{self.source}+{other.source}",
            )

        elif self.format == EmbeddingFormat.DENSE_VECTOR:
            if len(self.vector) != len(other.vector):
                raise ValueError("Cannot merge vectors of different dimensions")
            total_conf = self.confidence + other.confidence
            if total_conf == 0:
                total_conf = 1.0
            merged_vector = [
                (self.vector[i] * self.confidence + other.vector[i] * other.confidence) / total_conf
                for i in range(len(self.vector))
            ]
            return UserContextEmbedding(
                context_type=self.context_type,
                format=self.format,
                vector=merged_vector,
                confidence=max(self.confidence, other.confidence),
                source=f"{self.source}+{other.source}",
            )

        elif self.format == EmbeddingFormat.SPARSE_FEATURES:
            merged_features = dict(self.features)
            for k, v in other.features.items():
                merged_features[k] = max(merged_features.get(k, 0), v)
            return UserContextEmbedding(
                context_type=self.context_type,
                format=self.format,
                features=merged_features,
                confidence=max(self.confidence, other.confidence),
                source=f"{self.source}+{other.source}",
            )


@dataclass
class UCPExchangeRequest:
    """Request to exchange user context between agents."""
    request_id: str = field(default_factory=lambda: f"ucp-req-{uuid.uuid4().hex[:8]}")
    sender_id: str = ""
    receiver_id: str = ""
    embedding: Optional[UserContextEmbedding] = None
    requested_context_types: list[ContextType] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Serialize for Redis message bus."""
        return {
            "request_id": self.request_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "embedding": self.embedding.to_dict() if self.embedding else None,
            "requested_context_types": [ct.value for ct in self.requested_context_types],
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UCPExchangeRequest":
        """Deserialize from Redis message."""
        return cls(
            request_id=data["request_id"],
            sender_id=data["sender_id"],
            receiver_id=data["receiver_id"],
            embedding=UserContextEmbedding.from_dict(data["embedding"]) if data.get("embedding") else None,
            requested_context_types=[ContextType(ct) for ct in data.get("requested_context_types", [])],
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


@dataclass
class UCPExchangeResponse:
    """Response to a UCP exchange request."""
    response_id: str = field(default_factory=lambda: f"ucp-res-{uuid.uuid4().hex[:8]}")
    request_id: str = ""
    sender_id: str = ""
    embeddings: list[UserContextEmbedding] = field(default_factory=list)
    match_score: float = 0.0  # How well the context matches (0.0-1.0)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Serialize for Redis message bus."""
        return {
            "response_id": self.response_id,
            "request_id": self.request_id,
            "sender_id": self.sender_id,
            "embeddings": [e.to_dict() for e in self.embeddings],
            "match_score": self.match_score,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UCPExchangeResponse":
        """Deserialize from Redis message."""
        return cls(
            response_id=data["response_id"],
            request_id=data["request_id"],
            sender_id=data["sender_id"],
            embeddings=[UserContextEmbedding.from_dict(e) for e in data.get("embeddings", [])],
            match_score=data.get("match_score", 0.0),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


class UCPExchange:
    """
    Manages User Context Protocol embedding exchange between agents.

    Handles the bidirectional exchange of audience/context embeddings
    between buyer and seller agents during the bid process.
    """

    def __init__(self, agent_id: str, agent_type: str):
        """
        Initialize UCP exchange handler.

        Args:
            agent_id: This agent's identifier
            agent_type: Type of agent ('buyer' or 'seller')
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self._embedding_cache: dict[str, UserContextEmbedding] = {}
        self._pending_requests: dict[str, UCPExchangeRequest] = {}

    def create_segment_embedding(
        self,
        segment_ids: list[str],
        context_type: ContextType = ContextType.BEHAVIORAL,
        source: str = "dmp",
        confidence: float = 1.0,
    ) -> UserContextEmbedding:
        """
        Create an embedding from IAB segment IDs.

        Args:
            segment_ids: List of IAB or custom segment identifiers
            context_type: Type of context (demographic, behavioral, etc.)
            source: Data source (dmp, first_party, etc.)
            confidence: Confidence in the segments (0.0-1.0)

        Returns:
            UserContextEmbedding with segment ID format
        """
        return UserContextEmbedding(
            context_type=context_type,
            format=EmbeddingFormat.SEGMENT_IDS,
            segment_ids=segment_ids,
            source=source,
            confidence=confidence,
        )

    def create_vector_embedding(
        self,
        vector: list[float],
        context_type: ContextType = ContextType.BEHAVIORAL,
        source: str = "ml_model",
        confidence: float = 1.0,
    ) -> UserContextEmbedding:
        """
        Create a dense vector embedding (ML-based).

        Args:
            vector: Float vector representation
            context_type: Type of context
            source: Model/source that generated the vector
            confidence: Confidence score

        Returns:
            UserContextEmbedding with dense vector format
        """
        return UserContextEmbedding(
            context_type=context_type,
            format=EmbeddingFormat.DENSE_VECTOR,
            vector=vector,
            source=source,
            confidence=confidence,
        )

    def create_exchange_request(
        self,
        receiver_id: str,
        embedding: Optional[UserContextEmbedding] = None,
        requested_types: Optional[list[ContextType]] = None,
    ) -> UCPExchangeRequest:
        """
        Create a request to exchange context with another agent.

        Args:
            receiver_id: Target agent ID
            embedding: Optional embedding to share
            requested_types: Types of context being requested

        Returns:
            UCPExchangeRequest ready for transmission
        """
        request = UCPExchangeRequest(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            embedding=embedding,
            requested_context_types=requested_types or [],
        )
        self._pending_requests[request.request_id] = request
        return request

    def create_exchange_response(
        self,
        request: UCPExchangeRequest,
        embeddings: list[UserContextEmbedding],
        match_score: float = 0.0,
    ) -> UCPExchangeResponse:
        """
        Create a response to a UCP exchange request.

        Args:
            request: The original request
            embeddings: Embeddings to share in response
            match_score: How well the context matches

        Returns:
            UCPExchangeResponse ready for transmission
        """
        return UCPExchangeResponse(
            request_id=request.request_id,
            sender_id=self.agent_id,
            embeddings=embeddings,
            match_score=match_score,
        )

    def cache_embedding(self, embedding: UserContextEmbedding) -> None:
        """Cache an embedding by its fingerprint."""
        self._embedding_cache[embedding.fingerprint()] = embedding

    def get_cached_embedding(self, fingerprint: str) -> Optional[UserContextEmbedding]:
        """Retrieve a cached embedding by fingerprint."""
        return self._embedding_cache.get(fingerprint)

    def calculate_match_score(
        self,
        buyer_embedding: UserContextEmbedding,
        seller_embedding: UserContextEmbedding,
    ) -> float:
        """
        Calculate how well two embeddings match.

        Args:
            buyer_embedding: Buyer's targeting criteria
            seller_embedding: Seller's audience profile

        Returns:
            Match score from 0.0 (no match) to 1.0 (perfect match)
        """
        if buyer_embedding.format != seller_embedding.format:
            return 0.0

        if buyer_embedding.format == EmbeddingFormat.SEGMENT_IDS:
            # Jaccard similarity for segment overlap
            buyer_set = set(buyer_embedding.segment_ids)
            seller_set = set(seller_embedding.segment_ids)
            if not buyer_set and not seller_set:
                return 1.0  # No targeting = match all
            if not buyer_set or not seller_set:
                return 0.5  # One side has no targeting
            intersection = len(buyer_set & seller_set)
            union = len(buyer_set | seller_set)
            return intersection / union if union > 0 else 0.0

        elif buyer_embedding.format == EmbeddingFormat.DENSE_VECTOR:
            # Cosine similarity for vectors
            if len(buyer_embedding.vector) != len(seller_embedding.vector):
                return 0.0
            dot_product = sum(
                b * s for b, s in zip(buyer_embedding.vector, seller_embedding.vector)
            )
            buyer_norm = sum(b * b for b in buyer_embedding.vector) ** 0.5
            seller_norm = sum(s * s for s in seller_embedding.vector) ** 0.5
            if buyer_norm == 0 or seller_norm == 0:
                return 0.0
            similarity = dot_product / (buyer_norm * seller_norm)
            return (similarity + 1) / 2  # Convert [-1, 1] to [0, 1]

        elif buyer_embedding.format == EmbeddingFormat.SPARSE_FEATURES:
            # Feature overlap with weighted similarity
            buyer_features = set(buyer_embedding.features.keys())
            seller_features = set(seller_embedding.features.keys())
            common = buyer_features & seller_features
            if not common:
                return 0.0
            total_similarity = sum(
                min(buyer_embedding.features[k], seller_embedding.features[k])
                / max(buyer_embedding.features[k], seller_embedding.features[k])
                for k in common
                if max(buyer_embedding.features[k], seller_embedding.features[k]) > 0
            )
            return total_similarity / len(buyer_features) if buyer_features else 0.0

        return 0.0
