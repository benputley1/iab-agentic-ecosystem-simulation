"""
AA (Agentic Audience) Protocol.

Extension of UCP for agent-specific audience operations.
Provides audience discovery, validation, and reach estimation
capabilities for autonomous agent workflows.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from .ucp import AudienceSpec, UCPProtocol, UCPEmbedding


class SegmentType(Enum):
    """Types of audience segments."""
    FIRST_PARTY = "first_party"       # Advertiser's own data
    SECOND_PARTY = "second_party"     # Partner data
    THIRD_PARTY = "third_party"       # Data provider
    CONTEXTUAL = "contextual"         # Context-based
    BEHAVIORAL = "behavioral"         # Behavior-based
    LOOKALIKE = "lookalike"           # Modeled audiences


class ValidationStatus(Enum):
    """Validation result status."""
    VALID = "valid"
    INVALID = "invalid"
    PARTIAL = "partial"
    PENDING = "pending"


@dataclass
class CampaignBrief:
    """
    Campaign brief for audience discovery.
    
    Attributes:
        brief_id: Unique identifier
        advertiser: Advertiser name/ID
        objective: Campaign objective
        budget: Total budget
        target_audience: Audience description
        kpis: Key performance indicators
        constraints: Targeting constraints
        geo_targets: Geographic targets
        start_date: Campaign start
        end_date: Campaign end
    """
    brief_id: str
    advertiser: str
    objective: str
    budget: float
    target_audience: str
    kpis: dict[str, float] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    geo_targets: list[str] = field(default_factory=list)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    @classmethod
    def create(
        cls,
        advertiser: str,
        objective: str,
        budget: float,
        target_audience: str,
        **kwargs,
    ) -> CampaignBrief:
        """Create a new campaign brief with auto-generated ID."""
        return cls(
            brief_id=str(uuid.uuid4()),
            advertiser=advertiser,
            objective=objective,
            budget=budget,
            target_audience=target_audience,
            **kwargs,
        )
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "brief_id": self.brief_id,
            "advertiser": self.advertiser,
            "objective": self.objective,
            "budget": self.budget,
            "target_audience": self.target_audience,
            "kpis": self.kpis,
            "constraints": self.constraints,
            "geo_targets": self.geo_targets,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> CampaignBrief:
        """Deserialize from dictionary."""
        return cls(
            brief_id=data["brief_id"],
            advertiser=data["advertiser"],
            objective=data["objective"],
            budget=data["budget"],
            target_audience=data["target_audience"],
            kpis=data.get("kpis", {}),
            constraints=data.get("constraints", {}),
            geo_targets=data.get("geo_targets", []),
            start_date=datetime.fromisoformat(data["start_date"]) if data.get("start_date") else None,
            end_date=datetime.fromisoformat(data["end_date"]) if data.get("end_date") else None,
        )


@dataclass
class AudienceSegment:
    """
    Discovered audience segment.
    
    Attributes:
        segment_id: Unique identifier
        name: Segment name
        segment_type: Type of segment
        attributes: Defining attributes
        estimated_size: Estimated audience size
        cpm_range: (min, max) CPM range
        match_score: Relevance to query (0-1)
        provider: Data provider name
        embedding: UCP embedding for this segment
    """
    segment_id: str
    name: str
    segment_type: SegmentType
    attributes: list[str] = field(default_factory=list)
    estimated_size: int = 0
    cpm_range: tuple[float, float] = (0.0, 0.0)
    match_score: float = 0.0
    provider: Optional[str] = None
    embedding: Optional[UCPEmbedding] = None
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "segment_id": self.segment_id,
            "name": self.name,
            "segment_type": self.segment_type.value,
            "attributes": self.attributes,
            "estimated_size": self.estimated_size,
            "cpm_range": list(self.cpm_range),
            "match_score": self.match_score,
            "provider": self.provider,
            "has_embedding": self.embedding is not None,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> AudienceSegment:
        """Deserialize from dictionary."""
        return cls(
            segment_id=data["segment_id"],
            name=data["name"],
            segment_type=SegmentType(data["segment_type"]),
            attributes=data.get("attributes", []),
            estimated_size=data.get("estimated_size", 0),
            cpm_range=tuple(data.get("cpm_range", [0.0, 0.0])),
            match_score=data.get("match_score", 0.0),
            provider=data.get("provider"),
        )


@dataclass
class ValidationResult:
    """
    Result of targeting validation.
    
    Attributes:
        status: Validation status
        is_achievable: Whether targeting is achievable
        warnings: Non-blocking issues
        errors: Blocking issues
        suggested_modifications: Suggested fixes
        estimated_reach: Estimated reachable audience
        coverage_by_inventory: Coverage per inventory source
    """
    status: ValidationStatus
    is_achievable: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    suggested_modifications: list[str] = field(default_factory=list)
    estimated_reach: int = 0
    coverage_by_inventory: dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "status": self.status.value,
            "is_achievable": self.is_achievable,
            "warnings": self.warnings,
            "errors": self.errors,
            "suggested_modifications": self.suggested_modifications,
            "estimated_reach": self.estimated_reach,
            "coverage_by_inventory": self.coverage_by_inventory,
        }


@dataclass
class ReachEstimate:
    """
    Audience reach estimate for budget.
    
    Attributes:
        total_reach: Total unique users reachable
        impressions: Estimated impressions
        frequency: Average frequency per user
        effective_cpm: Blended CPM
        budget_utilization: % of budget that can be spent
        reach_by_segment: Breakdown by segment
        confidence_interval: (low, high) confidence bounds
    """
    total_reach: int
    impressions: int
    frequency: float
    effective_cpm: float
    budget_utilization: float
    reach_by_segment: dict[str, int] = field(default_factory=dict)
    confidence_interval: tuple[float, float] = (0.8, 1.2)
    
    @property
    def cost_per_reach(self) -> float:
        """Calculate cost per unique user reached."""
        if self.total_reach == 0:
            return 0.0
        return (self.impressions * self.effective_cpm / 1000) / self.total_reach
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "total_reach": self.total_reach,
            "impressions": self.impressions,
            "frequency": self.frequency,
            "effective_cpm": self.effective_cpm,
            "budget_utilization": self.budget_utilization,
            "reach_by_segment": self.reach_by_segment,
            "confidence_interval": list(self.confidence_interval),
            "cost_per_reach": self.cost_per_reach,
        }


class AAProtocol:
    """
    AA (Agentic Audience) protocol for audience targeting.
    
    Extension of UCP for agent-specific audience operations.
    Provides discovery, validation, and reach estimation.
    
    Features:
    - Audience segment discovery from campaign briefs
    - Targeting feasibility validation
    - Reach and budget forecasting
    - Integration with UCP embeddings
    """
    
    def __init__(self, ucp_protocol: Optional[UCPProtocol] = None):
        """
        Initialize AA protocol handler.
        
        Args:
            ucp_protocol: Optional UCP protocol instance for embeddings
        """
        self.ucp = ucp_protocol or UCPProtocol()
        self._segment_catalog: dict[str, AudienceSegment] = {}
        self._initialize_sample_segments()
    
    def _initialize_sample_segments(self) -> None:
        """Initialize sample segment catalog for simulation."""
        sample_segments = [
            AudienceSegment(
                segment_id="seg_auto_intenders",
                name="Auto Purchase Intenders",
                segment_type=SegmentType.THIRD_PARTY,
                attributes=["in_market_auto", "high_income"],
                estimated_size=15_000_000,
                cpm_range=(2.50, 4.00),
                provider="Oracle Data Cloud",
            ),
            AudienceSegment(
                segment_id="seg_business_travelers",
                name="Frequent Business Travelers",
                segment_type=SegmentType.BEHAVIORAL,
                attributes=["frequent_traveler", "high_income", "in_market_travel"],
                estimated_size=8_000_000,
                cpm_range=(3.00, 5.00),
                provider="Experian",
            ),
            AudienceSegment(
                segment_id="seg_sports_fans",
                name="Sports Enthusiasts 18-34",
                segment_type=SegmentType.BEHAVIORAL,
                attributes=["sports_enthusiast", "age_18_24", "age_25_34"],
                estimated_size=25_000_000,
                cpm_range=(1.50, 3.00),
                provider="Nielsen",
            ),
            AudienceSegment(
                segment_id="seg_streaming_heavy",
                name="Heavy Streaming Users",
                segment_type=SegmentType.FIRST_PARTY,
                attributes=["streaming_heavy", "mobile_first"],
                estimated_size=40_000_000,
                cpm_range=(2.00, 3.50),
                provider="Publisher Direct",
            ),
            AudienceSegment(
                segment_id="seg_finance_intenders",
                name="Financial Services Intenders",
                segment_type=SegmentType.THIRD_PARTY,
                attributes=["in_market_finance", "high_income", "age_35_44"],
                estimated_size=12_000_000,
                cpm_range=(4.00, 7.00),
                provider="Acxiom",
            ),
        ]
        
        for seg in sample_segments:
            self._segment_catalog[seg.segment_id] = seg
    
    def _extract_attributes_from_brief(self, brief: CampaignBrief) -> list[str]:
        """Extract targeting attributes from campaign brief text."""
        # Simple keyword extraction for simulation
        # In production, this would use NLP/LLM
        attributes = []
        text = brief.target_audience.lower()
        
        keyword_mapping = {
            "auto": "in_market_auto",
            "car": "in_market_auto",
            "vehicle": "in_market_auto",
            "travel": "in_market_travel",
            "vacation": "in_market_travel",
            "finance": "in_market_finance",
            "investment": "in_market_finance",
            "bank": "in_market_finance",
            "retail": "in_market_retail",
            "shop": "in_market_retail",
            "tech": "in_market_tech",
            "technology": "in_market_tech",
            "affluent": "high_income",
            "luxury": "high_income",
            "premium": "high_income",
            "sports": "sports_enthusiast",
            "athlete": "sports_enthusiast",
            "stream": "streaming_heavy",
            "mobile": "mobile_first",
            "young": "age_18_24",
            "millennial": "age_25_34",
            "gen z": "age_18_24",
        }
        
        for keyword, attr in keyword_mapping.items():
            if keyword in text and attr not in attributes:
                attributes.append(attr)
        
        return attributes
    
    async def discover_segments(
        self,
        brief: CampaignBrief,
        min_match_score: float = 0.3,
        max_segments: int = 10,
    ) -> list[AudienceSegment]:
        """
        Discover relevant audience segments from campaign brief.
        
        Args:
            brief: Campaign brief with targeting requirements
            min_match_score: Minimum relevance score
            max_segments: Maximum segments to return
            
        Returns:
            List of matching AudienceSegments sorted by relevance
        """
        # Extract attributes from brief
        brief_attrs = self._extract_attributes_from_brief(brief)
        brief_attrs.extend([f"geo_{g.lower()}" for g in brief.geo_targets])
        
        if not brief_attrs:
            # Return top segments by size if no attributes extracted
            segments = list(self._segment_catalog.values())
            segments.sort(key=lambda s: s.estimated_size, reverse=True)
            return segments[:max_segments]
        
        # Create brief embedding
        audience_spec = AudienceSpec(
            target_attributes=brief_attrs,
            geo_targets=brief.geo_targets,
        )
        brief_embedding = await self.ucp.create_query_embedding(audience_spec)
        
        # Score each segment
        results = []
        for segment in self._segment_catalog.values():
            # Calculate attribute overlap
            segment_attrs = set(segment.attributes)
            brief_set = set(brief_attrs)
            
            if segment_attrs and brief_set:
                overlap = len(segment_attrs & brief_set)
                union = len(segment_attrs | brief_set)
                jaccard = overlap / union if union > 0 else 0
            else:
                jaccard = 0.0
            
            # Adjust for budget fit (higher CPM segments need bigger budgets)
            min_cpm, max_cpm = segment.cpm_range
            avg_cpm = (min_cpm + max_cpm) / 2
            min_budget_needed = segment.estimated_size * avg_cpm / 1000 * 0.01  # 1% reach
            budget_fit = min(1.0, brief.budget / min_budget_needed) if min_budget_needed > 0 else 1.0
            
            # Combined score
            match_score = jaccard * 0.7 + budget_fit * 0.3
            
            if match_score >= min_match_score:
                segment_copy = AudienceSegment(
                    segment_id=segment.segment_id,
                    name=segment.name,
                    segment_type=segment.segment_type,
                    attributes=segment.attributes,
                    estimated_size=segment.estimated_size,
                    cpm_range=segment.cpm_range,
                    match_score=match_score,
                    provider=segment.provider,
                )
                results.append(segment_copy)
        
        # Sort by match score
        results.sort(key=lambda s: s.match_score, reverse=True)
        
        return results[:max_segments]
    
    async def validate_targeting(
        self,
        segments: list[AudienceSegment],
        inventory_id: str,
        min_reach: int = 10000,
    ) -> ValidationResult:
        """
        Validate targeting is achievable for given inventory.
        
        Args:
            segments: Target audience segments
            inventory_id: Inventory to validate against
            min_reach: Minimum required reach
            
        Returns:
            ValidationResult with feasibility assessment
        """
        warnings = []
        errors = []
        suggestions = []
        
        if not segments:
            return ValidationResult(
                status=ValidationStatus.INVALID,
                is_achievable=False,
                errors=["No segments specified"],
                suggested_modifications=["Add at least one audience segment"],
            )
        
        # Calculate combined reach (simplified - assumes some overlap)
        total_size = sum(s.estimated_size for s in segments)
        # Assume 30% overlap between segments
        overlap_factor = 1 - (0.3 * (len(segments) - 1)) if len(segments) > 1 else 1
        estimated_reach = int(total_size * max(0.4, overlap_factor))
        
        # Check minimum reach
        if estimated_reach < min_reach:
            errors.append(f"Estimated reach {estimated_reach:,} below minimum {min_reach:,}")
            suggestions.append("Consider broadening targeting or adding segments")
        
        # Check for conflicting attributes
        all_attrs = set()
        for seg in segments:
            for attr in seg.attributes:
                all_attrs.add(attr)
        
        # Age conflicts
        age_attrs = [a for a in all_attrs if a.startswith("age_")]
        if len(age_attrs) > 2:
            warnings.append(f"Multiple age ranges specified: {age_attrs}")
        
        # Estimate coverage
        coverage = {
            inventory_id: min(1.0, estimated_reach / 1_000_000),  # Simplified
        }
        
        # Determine status
        is_achievable = len(errors) == 0
        if warnings and not errors:
            status = ValidationStatus.PARTIAL
        elif errors:
            status = ValidationStatus.INVALID
        else:
            status = ValidationStatus.VALID
        
        return ValidationResult(
            status=status,
            is_achievable=is_achievable,
            warnings=warnings,
            errors=errors,
            suggested_modifications=suggestions,
            estimated_reach=estimated_reach,
            coverage_by_inventory=coverage,
        )
    
    async def estimate_reach(
        self,
        segments: list[AudienceSegment],
        budget: float,
        target_frequency: float = 3.0,
    ) -> ReachEstimate:
        """
        Estimate audience reach for given budget.
        
        Args:
            segments: Target audience segments
            budget: Total budget in dollars
            target_frequency: Desired average frequency
            
        Returns:
            ReachEstimate with reach and impression forecasts
        """
        if not segments or budget <= 0:
            return ReachEstimate(
                total_reach=0,
                impressions=0,
                frequency=0.0,
                effective_cpm=0.0,
                budget_utilization=0.0,
            )
        
        # Calculate blended CPM
        total_size = sum(s.estimated_size for s in segments)
        weighted_cpm = 0.0
        for seg in segments:
            weight = seg.estimated_size / total_size if total_size > 0 else 0
            avg_cpm = (seg.cpm_range[0] + seg.cpm_range[1]) / 2
            weighted_cpm += avg_cpm * weight
        
        effective_cpm = weighted_cpm if weighted_cpm > 0 else 2.50  # Default
        
        # Calculate impressions from budget
        impressions = int((budget / effective_cpm) * 1000)
        
        # Calculate reach from impressions and frequency
        ideal_reach = int(impressions / target_frequency)
        
        # Cap reach at available audience (with overlap factor)
        overlap_factor = 1 - (0.3 * (len(segments) - 1)) if len(segments) > 1 else 1
        max_reach = int(total_size * max(0.4, overlap_factor))
        
        actual_reach = min(ideal_reach, max_reach)
        actual_frequency = impressions / actual_reach if actual_reach > 0 else 0
        
        # Budget utilization
        achievable_imps = actual_reach * target_frequency
        achievable_budget = achievable_imps * effective_cpm / 1000
        utilization = min(1.0, achievable_budget / budget) if budget > 0 else 0
        
        # Reach by segment
        reach_by_segment = {}
        for seg in segments:
            seg_share = seg.estimated_size / total_size if total_size > 0 else 0
            reach_by_segment[seg.segment_id] = int(actual_reach * seg_share)
        
        return ReachEstimate(
            total_reach=actual_reach,
            impressions=impressions,
            frequency=round(actual_frequency, 2),
            effective_cpm=round(effective_cpm, 2),
            budget_utilization=round(utilization, 2),
            reach_by_segment=reach_by_segment,
            confidence_interval=(0.85, 1.15),
        )
    
    def register_segment(self, segment: AudienceSegment) -> None:
        """
        Register a new segment in the catalog.
        
        Args:
            segment: Segment to register
        """
        self._segment_catalog[segment.segment_id] = segment
    
    def get_segment(self, segment_id: str) -> Optional[AudienceSegment]:
        """
        Get segment by ID.
        
        Args:
            segment_id: Segment identifier
            
        Returns:
            AudienceSegment or None if not found
        """
        return self._segment_catalog.get(segment_id)
    
    def list_segments(self) -> list[AudienceSegment]:
        """Get all registered segments."""
        return list(self._segment_catalog.values())
