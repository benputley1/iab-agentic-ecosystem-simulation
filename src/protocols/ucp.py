"""
UCP (User Context Protocol) for Audience Embeddings.

Used by sellers to describe audience capabilities and by buyers
to specify audience requirements. Enables semantic matching
between buyer needs and seller inventory.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class EmbeddingType(Enum):
    """Types of UCP embeddings."""
    USER_INTENT = "user_intent"      # Buyer query embedding
    INVENTORY = "inventory"           # Seller inventory embedding
    SEGMENT = "segment"               # Audience segment embedding


class AudienceAttribute(Enum):
    """Standard audience attributes."""
    # Demographics
    AGE_18_24 = "age_18_24"
    AGE_25_34 = "age_25_34"
    AGE_35_44 = "age_35_44"
    AGE_45_54 = "age_45_54"
    AGE_55_PLUS = "age_55_plus"
    GENDER_MALE = "gender_male"
    GENDER_FEMALE = "gender_female"
    
    # Intent signals
    IN_MARKET_AUTO = "in_market_auto"
    IN_MARKET_TRAVEL = "in_market_travel"
    IN_MARKET_FINANCE = "in_market_finance"
    IN_MARKET_RETAIL = "in_market_retail"
    IN_MARKET_TECH = "in_market_tech"
    
    # Behavioral
    HIGH_INCOME = "high_income"
    FREQUENT_TRAVELER = "frequent_traveler"
    SPORTS_ENTHUSIAST = "sports_enthusiast"
    STREAMING_HEAVY = "streaming_heavy"
    MOBILE_FIRST = "mobile_first"


@dataclass
class AudienceSpec:
    """
    Buyer's audience specification.
    
    Attributes:
        target_attributes: Required audience attributes
        preferred_attributes: Nice-to-have attributes
        excluded_attributes: Attributes to avoid
        min_reach: Minimum audience size required
        geo_targets: Geographic targeting
        context_categories: Content context preferences
        custom_signals: Custom/proprietary signals
    """
    target_attributes: list[str] = field(default_factory=list)
    preferred_attributes: list[str] = field(default_factory=list)
    excluded_attributes: list[str] = field(default_factory=list)
    min_reach: Optional[int] = None
    geo_targets: list[str] = field(default_factory=list)
    context_categories: list[str] = field(default_factory=list)
    custom_signals: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "target_attributes": self.target_attributes,
            "preferred_attributes": self.preferred_attributes,
            "excluded_attributes": self.excluded_attributes,
            "min_reach": self.min_reach,
            "geo_targets": self.geo_targets,
            "context_categories": self.context_categories,
            "custom_signals": self.custom_signals,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> AudienceSpec:
        """Deserialize from dictionary."""
        return cls(
            target_attributes=data.get("target_attributes", []),
            preferred_attributes=data.get("preferred_attributes", []),
            excluded_attributes=data.get("excluded_attributes", []),
            min_reach=data.get("min_reach"),
            geo_targets=data.get("geo_targets", []),
            context_categories=data.get("context_categories", []),
            custom_signals=data.get("custom_signals", {}),
        )


@dataclass
class Inventory:
    """
    Seller's audience inventory.
    
    Attributes:
        inventory_id: Unique identifier
        name: Human-readable name
        available_attributes: Attributes this inventory can target
        estimated_reach: Total addressable audience
        geo_coverage: Available geographies
        content_categories: Content types/categories
        minimum_spend: Minimum buy requirement
        data_freshness_hours: How recent the data is
    """
    inventory_id: str
    name: str
    available_attributes: list[str] = field(default_factory=list)
    estimated_reach: int = 0
    geo_coverage: list[str] = field(default_factory=list)
    content_categories: list[str] = field(default_factory=list)
    minimum_spend: float = 0.0
    data_freshness_hours: int = 24
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "inventory_id": self.inventory_id,
            "name": self.name,
            "available_attributes": self.available_attributes,
            "estimated_reach": self.estimated_reach,
            "geo_coverage": self.geo_coverage,
            "content_categories": self.content_categories,
            "minimum_spend": self.minimum_spend,
            "data_freshness_hours": self.data_freshness_hours,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> Inventory:
        """Deserialize from dictionary."""
        return cls(
            inventory_id=data["inventory_id"],
            name=data["name"],
            available_attributes=data.get("available_attributes", []),
            estimated_reach=data.get("estimated_reach", 0),
            geo_coverage=data.get("geo_coverage", []),
            content_categories=data.get("content_categories", []),
            minimum_spend=data.get("minimum_spend", 0.0),
            data_freshness_hours=data.get("data_freshness_hours", 24),
        )


@dataclass
class UCPEmbedding:
    """
    User Context Protocol embedding.
    
    Represents audience intent or inventory as a dense vector
    for semantic matching.
    
    Attributes:
        version: Protocol version
        embedding_type: Type of embedding (user_intent or inventory)
        dimension: Vector dimensionality
        vector: The embedding vector
        metadata: Additional metadata
        created_at: Creation timestamp
        source_hash: Hash of source data for cache invalidation
    """
    version: str = "1.0"
    embedding_type: str = "user_intent"
    dimension: int = 512
    vector: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    source_hash: Optional[str] = None
    
    def __post_init__(self):
        """Validate and normalize the embedding."""
        if self.vector and len(self.vector) != self.dimension:
            raise ValueError(
                f"Vector length {len(self.vector)} != dimension {self.dimension}"
            )
    
    def normalize(self) -> UCPEmbedding:
        """Return L2-normalized version of this embedding."""
        if not self.vector:
            return self
        
        magnitude = math.sqrt(sum(x * x for x in self.vector))
        if magnitude == 0:
            return self
        
        normalized = [x / magnitude for x in self.vector]
        return UCPEmbedding(
            version=self.version,
            embedding_type=self.embedding_type,
            dimension=self.dimension,
            vector=normalized,
            metadata=self.metadata,
            created_at=self.created_at,
            source_hash=self.source_hash,
        )
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "version": self.version,
            "embedding_type": self.embedding_type,
            "dimension": self.dimension,
            "vector": self.vector,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "source_hash": self.source_hash,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> UCPEmbedding:
        """Deserialize from dictionary."""
        return cls(
            version=data.get("version", "1.0"),
            embedding_type=data.get("embedding_type", "user_intent"),
            dimension=data.get("dimension", 512),
            vector=data.get("vector", []),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            source_hash=data.get("source_hash"),
        )


@dataclass
class MatchResult:
    """
    Result of matching query embedding to inventory embedding.
    
    Attributes:
        similarity_score: Cosine similarity (0-1)
        coverage_percentage: % of required attributes covered
        matched_capabilities: List of matched attribute names
        missing_capabilities: Required but unavailable attributes
        reach_estimate: Estimated audience size for match
        confidence: Confidence in the match quality
    """
    similarity_score: float
    coverage_percentage: float
    matched_capabilities: list[str] = field(default_factory=list)
    missing_capabilities: list[str] = field(default_factory=list)
    reach_estimate: Optional[int] = None
    confidence: float = 1.0
    
    @property
    def is_viable(self) -> bool:
        """Check if match meets minimum viability threshold."""
        return self.similarity_score >= 0.5 and self.coverage_percentage >= 0.7
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "similarity_score": self.similarity_score,
            "coverage_percentage": self.coverage_percentage,
            "matched_capabilities": self.matched_capabilities,
            "missing_capabilities": self.missing_capabilities,
            "reach_estimate": self.reach_estimate,
            "confidence": self.confidence,
            "is_viable": self.is_viable,
        }


class UCPProtocol:
    """
    UCP (User Context Protocol) for audience embeddings.
    
    Used by sellers to describe audience capabilities
    and by buyers to specify audience requirements.
    
    Features:
    - Semantic embedding creation for audience specs
    - Inventory capability embedding
    - Similarity-based matching
    - Coverage analysis
    """
    
    def __init__(
        self,
        dimension: int = 512,
        version: str = "1.0",
    ):
        """
        Initialize UCP protocol handler.
        
        Args:
            dimension: Embedding vector dimension
            version: Protocol version string
        """
        self.dimension = dimension
        self.version = version
        self._attribute_vectors: dict[str, list[float]] = {}
        self._initialize_attribute_vectors()
    
    def _initialize_attribute_vectors(self) -> None:
        """Initialize base vectors for known attributes."""
        # In production, these would be learned embeddings
        # For simulation, we use deterministic pseudo-random vectors
        all_attributes = [
            # Demographics
            "age_18_24", "age_25_34", "age_35_44", "age_45_54", "age_55_plus",
            "gender_male", "gender_female",
            # Intent
            "in_market_auto", "in_market_travel", "in_market_finance",
            "in_market_retail", "in_market_tech",
            # Behavioral
            "high_income", "frequent_traveler", "sports_enthusiast",
            "streaming_heavy", "mobile_first",
            # Geo (common)
            "geo_us", "geo_uk", "geo_eu", "geo_apac",
            # Context
            "context_news", "context_sports", "context_entertainment",
            "context_business", "context_tech",
        ]
        
        for attr in all_attributes:
            # Deterministic seed from attribute name
            seed = int(hashlib.md5(attr.encode()).hexdigest()[:8], 16)
            rng = random.Random(seed)
            vector = [rng.gauss(0, 1) for _ in range(self.dimension)]
            # Normalize
            magnitude = math.sqrt(sum(x * x for x in vector))
            self._attribute_vectors[attr] = [x / magnitude for x in vector]
    
    def _get_attribute_vector(self, attribute: str) -> list[float]:
        """Get or create vector for an attribute."""
        if attribute not in self._attribute_vectors:
            # Create deterministic vector for unknown attribute
            seed = int(hashlib.md5(attribute.encode()).hexdigest()[:8], 16)
            rng = random.Random(seed)
            vector = [rng.gauss(0, 1) for _ in range(self.dimension)]
            magnitude = math.sqrt(sum(x * x for x in vector))
            self._attribute_vectors[attribute] = [x / magnitude for x in vector]
        return self._attribute_vectors[attribute]
    
    def _combine_vectors(
        self,
        attributes: list[str],
        weights: Optional[list[float]] = None,
    ) -> list[float]:
        """Combine multiple attribute vectors into one."""
        if not attributes:
            return [0.0] * self.dimension
        
        if weights is None:
            weights = [1.0] * len(attributes)
        
        combined = [0.0] * self.dimension
        for attr, weight in zip(attributes, weights):
            vec = self._get_attribute_vector(attr)
            for i in range(self.dimension):
                combined[i] += vec[i] * weight
        
        # Normalize
        magnitude = math.sqrt(sum(x * x for x in combined))
        if magnitude > 0:
            combined = [x / magnitude for x in combined]
        
        return combined
    
    def _compute_source_hash(self, data: dict) -> str:
        """Compute hash of source data for caching."""
        import json
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]
    
    async def create_query_embedding(
        self,
        audience_spec: AudienceSpec,
    ) -> UCPEmbedding:
        """
        Create buyer query embedding from audience specification.
        
        Args:
            audience_spec: The buyer's audience requirements
            
        Returns:
            UCPEmbedding representing the buyer's intent
        """
        # Weight target attributes higher than preferred
        all_attrs = (
            [(a, 1.0) for a in audience_spec.target_attributes] +
            [(a, 0.5) for a in audience_spec.preferred_attributes] +
            [(f"geo_{g.lower()}", 0.8) for g in audience_spec.geo_targets] +
            [(f"context_{c.lower()}", 0.6) for c in audience_spec.context_categories]
        )
        
        if not all_attrs:
            vector = [0.0] * self.dimension
        else:
            attrs = [a for a, _ in all_attrs]
            weights = [w for _, w in all_attrs]
            vector = self._combine_vectors(attrs, weights)
        
        return UCPEmbedding(
            version=self.version,
            embedding_type=EmbeddingType.USER_INTENT.value,
            dimension=self.dimension,
            vector=vector,
            metadata={
                "target_count": len(audience_spec.target_attributes),
                "preferred_count": len(audience_spec.preferred_attributes),
                "geo_count": len(audience_spec.geo_targets),
                "min_reach": audience_spec.min_reach,
            },
            source_hash=self._compute_source_hash(audience_spec.to_dict()),
        )
    
    async def create_inventory_embedding(
        self,
        inventory: Inventory,
    ) -> UCPEmbedding:
        """
        Create seller inventory embedding.
        
        Args:
            inventory: The seller's audience inventory
            
        Returns:
            UCPEmbedding representing inventory capabilities
        """
        all_attrs = (
            [(a, 1.0) for a in inventory.available_attributes] +
            [(f"geo_{g.lower()}", 0.8) for g in inventory.geo_coverage] +
            [(f"context_{c.lower()}", 0.6) for c in inventory.content_categories]
        )
        
        if not all_attrs:
            vector = [0.0] * self.dimension
        else:
            attrs = [a for a, _ in all_attrs]
            weights = [w for _, w in all_attrs]
            vector = self._combine_vectors(attrs, weights)
        
        return UCPEmbedding(
            version=self.version,
            embedding_type=EmbeddingType.INVENTORY.value,
            dimension=self.dimension,
            vector=vector,
            metadata={
                "inventory_id": inventory.inventory_id,
                "name": inventory.name,
                "attribute_count": len(inventory.available_attributes),
                "estimated_reach": inventory.estimated_reach,
            },
            source_hash=self._compute_source_hash(inventory.to_dict()),
        )
    
    async def match_embeddings(
        self,
        query: UCPEmbedding,
        inventory: UCPEmbedding,
        query_spec: Optional[AudienceSpec] = None,
        inventory_data: Optional[Inventory] = None,
    ) -> MatchResult:
        """
        Calculate similarity between query and inventory embeddings.
        
        Args:
            query: Buyer query embedding
            inventory: Seller inventory embedding
            query_spec: Original audience spec (for coverage analysis)
            inventory_data: Original inventory (for coverage analysis)
            
        Returns:
            MatchResult with similarity and coverage metrics
        """
        # Cosine similarity (vectors should already be normalized)
        if not query.vector or not inventory.vector:
            similarity = 0.0
        else:
            dot_product = sum(q * i for q, i in zip(query.vector, inventory.vector))
            similarity = max(0.0, min(1.0, (dot_product + 1) / 2))  # Map to 0-1
        
        # Coverage analysis if specs provided
        matched = []
        missing = []
        coverage = 1.0
        
        if query_spec and inventory_data:
            required = set(query_spec.target_attributes)
            available = set(inventory_data.available_attributes)
            
            matched = list(required & available)
            missing = list(required - available)
            
            if required:
                coverage = len(matched) / len(required)
        
        # Estimate reach
        reach_estimate = None
        if inventory_data:
            # Simple reach estimation based on coverage
            reach_estimate = int(inventory_data.estimated_reach * coverage * similarity)
        
        return MatchResult(
            similarity_score=similarity,
            coverage_percentage=coverage,
            matched_capabilities=matched,
            missing_capabilities=missing,
            reach_estimate=reach_estimate,
            confidence=0.8 if query_spec and inventory_data else 0.6,
        )
    
    async def find_best_matches(
        self,
        query: UCPEmbedding,
        inventory_embeddings: list[tuple[Inventory, UCPEmbedding]],
        query_spec: Optional[AudienceSpec] = None,
        min_similarity: float = 0.3,
        top_k: int = 10,
    ) -> list[tuple[Inventory, MatchResult]]:
        """
        Find best matching inventories for a query.
        
        Args:
            query: Buyer query embedding
            inventory_embeddings: List of (inventory, embedding) tuples
            query_spec: Original audience spec for coverage
            min_similarity: Minimum similarity threshold
            top_k: Maximum results to return
            
        Returns:
            Sorted list of (inventory, match_result) tuples
        """
        results = []
        
        for inv, inv_embedding in inventory_embeddings:
            match = await self.match_embeddings(
                query, inv_embedding,
                query_spec=query_spec,
                inventory_data=inv,
            )
            if match.similarity_score >= min_similarity:
                results.append((inv, match))
        
        # Sort by similarity score descending
        results.sort(key=lambda x: x[1].similarity_score, reverse=True)
        
        return results[:top_k]
