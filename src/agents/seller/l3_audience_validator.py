"""
L3 Audience Validator Agent - Audience validation and coverage.

Handles validating audience segments, checking targeting capability,
and calculating audience coverage.
"""

from dataclasses import dataclass, field
from typing import Optional

from ..base import FunctionalAgent, ToolDefinition


@dataclass
class AudienceSpec:
    """Audience targeting specification."""
    
    segments: list[str] = field(default_factory=list)
    demographics: dict = field(default_factory=dict)
    geo_targeting: list[str] = field(default_factory=list)
    behavioral: list[str] = field(default_factory=list)
    contextual: list[str] = field(default_factory=list)
    custom_audiences: list[str] = field(default_factory=list)
    exclusions: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    @property
    def is_empty(self) -> bool:
        """Check if spec has any targeting."""
        return not any([
            self.segments,
            self.demographics,
            self.geo_targeting,
            self.behavioral,
            self.contextual,
            self.custom_audiences,
        ])


@dataclass
class ValidationResult:
    """Result of audience validation."""
    
    valid: bool
    achievable: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    unsupported_segments: list[str] = field(default_factory=list)
    supported_segments: list[str] = field(default_factory=list)
    estimated_reach: int = 0
    confidence: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class Coverage:
    """Audience coverage calculation."""
    
    product_id: str
    audience_spec: AudienceSpec
    total_reach: int
    matched_impressions: int
    coverage_pct: float
    overlap_with_inventory: float
    segment_breakdown: dict = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class AudienceValidatorAgent(FunctionalAgent):
    """
    Audience validation and coverage.
    
    Tools:
    - AudienceValidation: Validate audience segments
    - AudienceCapability: Check targeting capability
    - CoverageCalculator: Calculate audience coverage
    
    This agent handles:
    - Validating audience segments are supported
    - Checking targeting achievability
    - Calculating coverage against inventory
    - Providing recommendations for targeting optimization
    """
    
    # Standard supported segments
    SUPPORTED_SEGMENTS = {
        "auto_intenders", "tech_enthusiasts", "parents",
        "sports_fans", "travel_intenders", "home_improvers",
        "business_professionals", "entertainment_seekers",
        "health_wellness", "fashion_beauty", "gamers",
        "finance_investors", "food_cooking", "pet_owners",
        "education_seekers", "luxury_shoppers",
    }
    
    # Supported demographics
    SUPPORTED_DEMOGRAPHICS = {
        "age": ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
        "gender": ["male", "female", "all"],
        "income": ["low", "medium", "high", "affluent"],
        "education": ["high_school", "college", "graduate"],
    }
    
    def __init__(self, **kwargs):
        """Initialize AudienceValidatorAgent."""
        kwargs.setdefault("name", "AudienceValidatorAgent")
        super().__init__(**kwargs)
    
    def _register_tools(self) -> None:
        """Register audience validation tools."""
        self.register_tool(
            ToolDefinition(
                name="AudienceValidation",
                description="Validate audience targeting specification",
                parameters={
                    "segments": {"type": "array", "items": {"type": "string"}},
                    "demographics": {"type": "object"},
                    "geo_targeting": {"type": "array", "items": {"type": "string"}},
                    "behavioral": {"type": "array", "items": {"type": "string"}}
                },
                required_params=["segments"]
            ),
            handler=self._handle_audience_validation
        )
        
        self.register_tool(
            ToolDefinition(
                name="AudienceCapability",
                description="Check targeting capability for audience spec",
                parameters={
                    "segments": {"type": "array", "items": {"type": "string"}},
                    "demographics": {"type": "object"},
                    "product_id": {"type": "string"}
                },
                required_params=["segments"]
            ),
            handler=self._handle_audience_capability
        )
        
        self.register_tool(
            ToolDefinition(
                name="CoverageCalculator",
                description="Calculate audience coverage against inventory",
                parameters={
                    "product_id": {"type": "string"},
                    "segments": {"type": "array", "items": {"type": "string"}},
                    "demographics": {"type": "object"},
                    "geo_targeting": {"type": "array", "items": {"type": "string"}}
                },
                required_params=["product_id", "segments"]
            ),
            handler=self._handle_coverage_calculator
        )
    
    def get_system_prompt(self) -> str:
        """Get system prompt for audience validation."""
        return """You are an Audience Validator Agent responsible for validating targeting specifications.

Your responsibilities:
1. Validate that audience segments are supported
2. Check targeting capability against inventory
3. Calculate expected coverage and reach
4. Provide recommendations for targeting optimization

Available tools:
- AudienceValidation: Validate audience specs
- AudienceCapability: Check targeting capability
- CoverageCalculator: Calculate coverage

Help optimize targeting for maximum reach while maintaining relevance."""
    
    def _handle_audience_validation(
        self,
        segments: list,
        demographics: Optional[dict] = None,
        geo_targeting: Optional[list] = None,
        behavioral: Optional[list] = None,
    ) -> dict:
        """Handle AudienceValidation tool."""
        supported = [s for s in segments if s.lower() in self.SUPPORTED_SEGMENTS]
        unsupported = [s for s in segments if s.lower() not in self.SUPPORTED_SEGMENTS]
        
        return {
            "valid": len(unsupported) == 0,
            "achievable": len(supported) > 0 or len(segments) == 0,
            "supported": supported,
            "unsupported": unsupported,
            "issues": [f"Unsupported segment: {s}" for s in unsupported],
            "warnings": [],
            "estimated_reach": 5000000,
            "confidence": 0.85,
        }
    
    def _handle_audience_capability(
        self,
        segments: list,
        demographics: Optional[dict] = None,
        product_id: Optional[str] = None,
    ) -> dict:
        """Handle AudienceCapability tool."""
        supported_count = sum(1 for s in segments if s.lower() in self.SUPPORTED_SEGMENTS)
        total_segments = len(segments) if segments else 1
        
        return {
            "fully_supported": supported_count == len(segments),
            "partially_supported": supported_count > 0,
            "support_rate": supported_count / total_segments,
            "supported_segments": [s for s in segments if s.lower() in self.SUPPORTED_SEGMENTS],
            "unsupported_segments": [s for s in segments if s.lower() not in self.SUPPORTED_SEGMENTS],
        }
    
    def _handle_coverage_calculator(
        self,
        product_id: str,
        segments: list,
        demographics: Optional[dict] = None,
        geo_targeting: Optional[list] = None,
    ) -> dict:
        """Handle CoverageCalculator tool."""
        base_coverage = 0.7
        if segments:
            base_coverage *= 0.85 ** len(segments)
        
        total_reach = 5000000
        matched = int(total_reach * base_coverage)
        
        return {
            "total_reach": total_reach,
            "matched_impressions": matched,
            "coverage_pct": round(base_coverage * 100, 1),
            "overlap": round(base_coverage * 0.8, 2),
            "breakdown": {s: round(base_coverage * 100, 1) for s in segments},
            "recommendations": [],
        }
    
    async def validate_audience(self, spec: AudienceSpec) -> ValidationResult:
        """Validate audience targeting is achievable."""
        result = self._handle_audience_validation(
            segments=spec.segments,
            demographics=spec.demographics,
            geo_targeting=spec.geo_targeting,
            behavioral=spec.behavioral,
        )
        
        return ValidationResult(
            valid=result.get("valid", True),
            achievable=result.get("achievable", True),
            issues=result.get("issues", []),
            warnings=result.get("warnings", []),
            unsupported_segments=result.get("unsupported", []),
            supported_segments=result.get("supported", []),
            estimated_reach=result.get("estimated_reach", 0),
            confidence=result.get("confidence", 0.8),
        )
    
    async def calculate_coverage(
        self,
        spec: AudienceSpec,
        inventory: str,
    ) -> Coverage:
        """Calculate coverage for audience against inventory."""
        result = self._handle_coverage_calculator(
            product_id=inventory,
            segments=spec.segments,
            demographics=spec.demographics,
            geo_targeting=spec.geo_targeting,
        )
        
        recommendations = []
        coverage_pct = result.get("coverage_pct", 0.0)
        if coverage_pct < 30:
            recommendations.append("Consider broadening audience targeting for better scale")
        if len(spec.segments) > 3:
            recommendations.append("Consolidate similar audience segments to improve match rate")
        
        return Coverage(
            product_id=inventory,
            audience_spec=spec,
            total_reach=result.get("total_reach", 0),
            matched_impressions=result.get("matched_impressions", 0),
            coverage_pct=coverage_pct,
            overlap_with_inventory=result.get("overlap", 0.0),
            segment_breakdown=result.get("breakdown", {}),
            recommendations=recommendations,
        )
    
    async def check_capability(
        self,
        spec: AudienceSpec,
        inventory: Optional[str] = None,
    ) -> dict:
        """Check targeting capability for audience spec."""
        return self._handle_audience_capability(
            segments=spec.segments,
            demographics=spec.demographics,
            product_id=inventory,
        )
    
    async def get_segment_recommendations(
        self,
        current_segments: list[str],
        target_reach: int,
    ) -> list[str]:
        """Get recommendations for additional segments."""
        current_set = {s.lower() for s in current_segments}
        available = [s for s in self.SUPPORTED_SEGMENTS if s not in current_set]
        return available[:3]
