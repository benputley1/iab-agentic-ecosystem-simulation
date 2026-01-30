"""L3 Audience Planner Agent for Buyer Agent System.

The Audience Planner Agent handles audience discovery and planning:
- AudienceDiscovery: Find relevant audience segments
- AudienceMatching: Match segments against inventory
- CoverageEstimation: Estimate reach and coverage
"""

from typing import Any, Optional
from datetime import datetime
import uuid
import random

from .l3_base import (
    FunctionalAgent,
    ToolResult,
    ToolExecutionStatus,
    AgentContext,
)
from .l3_tools import (
    ToolSchema,
    CampaignBrief,
    AudienceSegment,
    CoverageEstimate,
)


class AudiencePlannerAgent(FunctionalAgent[list[AudienceSegment]]):
    """Audience discovery and planning functional agent.
    
    This L3 agent specializes in finding and evaluating audience
    segments for advertising campaigns.
    
    Tools:
        - AudienceDiscovery: Find audience segments
        - AudienceMatching: Match against inventory
        - CoverageEstimation: Estimate reach
    
    Example:
        ```python
        agent = AudiencePlannerAgent(context)
        
        # Discover audiences from a brief
        segments = await agent.discover_audiences(CampaignBrief(
            campaign_id="camp-001",
            objective="awareness",
            target_audience_description="Tech-savvy millennials interested in finance",
            budget=50000.0,
            geography=["US", "UK"],
        ))
        
        # Estimate coverage
        coverage = await agent.estimate_coverage(
            segments=[seg.segment_id for seg in segments],
            inventory=["inv-001", "inv-002"],
        )
        ```
    """
    
    TOOLS = ["AudienceDiscovery", "AudienceMatching", "CoverageEstimation"]
    
    def __init__(
        self,
        context: AgentContext,
        segment_catalog: Optional[dict] = None,
        **kwargs,
    ):
        """Initialize audience planner agent.
        
        Args:
            context: Agent context with buyer/scenario info
            segment_catalog: Optional pre-defined segment catalog
            **kwargs: Additional args passed to FunctionalAgent
        """
        super().__init__(context, **kwargs)
        self._segment_catalog = segment_catalog or self._default_segment_catalog()
    
    @property
    def system_prompt(self) -> str:
        """System prompt for audience planner agent."""
        return f"""You are an Audience Planning Specialist for a programmatic advertising buyer.
Your role is to discover, analyze, and plan audience targeting for campaigns.

Buyer ID: {self.context.buyer_id}
Scenario: {self.context.scenario}
Campaign: {self.context.campaign_id or "not specified"}

Your capabilities:
1. AudienceDiscovery - Find relevant audience segments based on campaign objectives
2. AudienceMatching - Match segments against available inventory
3. CoverageEstimation - Estimate reach and frequency

Audience planning considerations:
- Match segment demographics to campaign goals
- Consider segment size vs. specificity tradeoff
- Evaluate match rates with available inventory
- Project reach and frequency metrics

When recommending audiences:
- Prioritize segments with high match scores
- Consider overlap between segments
- Balance reach and precision
- Account for budget constraints

Always provide data-driven audience recommendations."""
    
    @property
    def available_tools(self) -> list[dict]:
        """Tools available to this agent."""
        return [
            ToolSchema.audience_discovery(),
            ToolSchema.audience_matching(),
            ToolSchema.coverage_estimation(),
        ]
    
    async def _execute_tool(self, name: str, params: dict) -> ToolResult:
        """Execute an audience planning tool.
        
        Args:
            name: Tool name
            params: Tool parameters from LLM
            
        Returns:
            ToolResult with execution outcome
        """
        if name == "AudienceDiscovery":
            return await self._tool_audience_discovery(params)
        elif name == "AudienceMatching":
            return await self._tool_audience_matching(params)
        elif name == "CoverageEstimation":
            return await self._tool_coverage_estimation(params)
        else:
            return ToolResult(
                tool_name=name,
                status=ToolExecutionStatus.FAILED,
                error=f"Unknown tool: {name}",
            )
    
    # -------------------------------------------------------------------------
    # High-Level Methods
    # -------------------------------------------------------------------------
    
    async def discover_audiences(
        self,
        brief: CampaignBrief,
    ) -> list[AudienceSegment]:
        """Find relevant audience segments for a campaign.
        
        Args:
            brief: Campaign brief with targeting requirements
            
        Returns:
            List of matching audience segments
        """
        result = await self._tool_audience_discovery({
            "objective": brief.objective,
            "target_description": brief.target_audience_description,
            "geography": brief.geography or [],
            "demographics": brief.demographics or {},
        })
        
        if result.success and result.data:
            return result.data
        return []
    
    async def match_audiences(
        self,
        segment_ids: list[str],
        channel: Optional[str] = None,
        min_match_rate: float = 0.3,
    ) -> list[dict]:
        """Match audience segments against available inventory.
        
        Args:
            segment_ids: Segments to match
            channel: Optional channel filter
            min_match_rate: Minimum match rate threshold
            
        Returns:
            List of inventory matches
        """
        result = await self._tool_audience_matching({
            "segment_ids": segment_ids,
            "channel": channel,
            "min_match_rate": min_match_rate,
        })
        
        if result.success and result.data:
            return result.data
        return []
    
    async def estimate_coverage(
        self,
        segments: list[str],
        inventory: list[str],
        budget: Optional[float] = None,
        duration_days: int = 30,
    ) -> CoverageEstimate:
        """Estimate audience coverage across segments and inventory.
        
        Args:
            segments: Audience segment IDs
            inventory: Inventory IDs to estimate against
            budget: Optional budget constraint
            duration_days: Campaign duration
            
        Returns:
            Coverage estimation
        """
        result = await self._tool_coverage_estimation({
            "segment_ids": segments,
            "inventory_ids": inventory,
            "budget": budget,
            "duration_days": duration_days,
        })
        
        if result.success and result.data:
            return result.data
        
        raise ValueError(f"Failed to estimate coverage: {result.error}")
    
    # -------------------------------------------------------------------------
    # Tool Implementations
    # -------------------------------------------------------------------------
    
    async def _tool_audience_discovery(self, params: dict) -> ToolResult:
        """Execute AudienceDiscovery tool."""
        try:
            target_description = params.get("target_description") or ""
            objective = params.get("objective") or "awareness"
            geography = params.get("geography") or []
            demographics = params.get("demographics") or {}
            interests = params.get("interests") or []
            
            # Score segments based on criteria
            scored_segments = []
            
            for segment_id, segment_data in self._segment_catalog.items():
                score = self._calculate_segment_match_score(
                    segment_data,
                    target_description,
                    objective,
                    geography,
                    demographics,
                    interests,
                )
                
                if score > 0.15:  # Lower minimum threshold to get more matches
                    scored_segments.append((segment_id, segment_data, score))
            
            # Sort by score and take top matches
            scored_segments.sort(key=lambda x: x[2], reverse=True)
            top_segments = scored_segments[:5]
            
            # Convert to AudienceSegment objects
            segments = [
                AudienceSegment(
                    segment_id=seg_id,
                    name=seg_data["name"],
                    description=seg_data["description"],
                    size=seg_data["size"],
                    match_score=score,
                    demographics=seg_data["demographics"],
                    interests=seg_data["interests"],
                )
                for seg_id, seg_data, score in top_segments
            ]
            
            return ToolResult(
                tool_name="AudienceDiscovery",
                status=ToolExecutionStatus.SUCCESS,
                data=segments,
            )
            
        except Exception as e:
            return ToolResult(
                tool_name="AudienceDiscovery",
                status=ToolExecutionStatus.FAILED,
                error=str(e),
            )
    
    async def _tool_audience_matching(self, params: dict) -> ToolResult:
        """Execute AudienceMatching tool."""
        try:
            segment_ids = params.get("segment_ids", [])
            channel = params.get("channel")
            min_match_rate = params.get("min_match_rate", 0.3)
            
            # Generate matches for each segment
            matches = []
            
            for segment_id in segment_ids:
                segment_data = self._segment_catalog.get(segment_id, {})
                segment_size = segment_data.get("size", 100000)
                
                # Generate realistic inventory matches
                num_matches = random.randint(3, 8)
                
                for i in range(num_matches):
                    match_rate = random.uniform(0.25, 0.75)
                    
                    if match_rate >= min_match_rate:
                        match = {
                            "segment_id": segment_id,
                            "inventory_id": f"INV-{uuid.uuid4().hex[:8]}",
                            "inventory_name": f"Premium {channel or 'Display'} Package {i+1}",
                            "match_rate": round(match_rate, 3),
                            "matched_users": int(segment_size * match_rate),
                            "available_impressions": random.randint(500000, 5000000),
                            "cpm_range": {
                                "min": round(random.uniform(10, 15), 2),
                                "max": round(random.uniform(20, 35), 2),
                            },
                            "channel": channel or "display",
                        }
                        matches.append(match)
            
            # Sort by match rate
            matches.sort(key=lambda x: x["match_rate"], reverse=True)
            
            return ToolResult(
                tool_name="AudienceMatching",
                status=ToolExecutionStatus.SUCCESS,
                data=matches,
            )
            
        except Exception as e:
            return ToolResult(
                tool_name="AudienceMatching",
                status=ToolExecutionStatus.FAILED,
                error=str(e),
            )
    
    async def _tool_coverage_estimation(self, params: dict) -> ToolResult:
        """Execute CoverageEstimation tool."""
        try:
            segment_ids = params.get("segment_ids", [])
            inventory_ids = params.get("inventory_ids", [])
            budget = params.get("budget")
            duration_days = params.get("duration_days", 30)
            
            # Calculate total audience size
            total_audience = 0
            segment_details = []
            
            for segment_id in segment_ids:
                segment_data = self._segment_catalog.get(segment_id, {})
                size = segment_data.get("size", 100000)
                total_audience += size
                
                segment_details.append({
                    "segment_id": segment_id,
                    "name": segment_data.get("name", "Unknown"),
                    "size": size,
                    "estimated_reach": int(size * random.uniform(0.3, 0.7)),
                })
            
            # Account for overlap (de-duplicate)
            overlap_factor = 0.15 if len(segment_ids) > 1 else 0.0
            unique_users = int(total_audience * (1 - overlap_factor))
            
            # Calculate reach based on budget and duration
            avg_cpm = 18.0
            if budget:
                impressions = (budget / avg_cpm) * 1000
                reach = min(unique_users, int(impressions * 0.7))  # Not all impressions are unique
            else:
                reach = int(unique_users * 0.5)
            
            # Calculate frequency
            total_impressions = reach * random.uniform(3, 8)
            frequency = total_impressions / reach if reach > 0 else 0
            
            # Coverage percentage
            coverage_pct = (reach / unique_users * 100) if unique_users > 0 else 0
            
            # Inventory match rate
            inventory_match = random.uniform(0.5, 0.85)
            
            coverage = CoverageEstimate(
                total_reach=reach,
                unique_users=unique_users,
                frequency=round(frequency, 1),
                coverage_percentage=round(coverage_pct, 1),
                segments=segment_details,
                inventory_match=round(inventory_match, 2),
            )
            
            return ToolResult(
                tool_name="CoverageEstimation",
                status=ToolExecutionStatus.SUCCESS,
                data=coverage,
            )
            
        except Exception as e:
            return ToolResult(
                tool_name="CoverageEstimation",
                status=ToolExecutionStatus.FAILED,
                error=str(e),
            )
    
    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------
    
    def _default_segment_catalog(self) -> dict:
        """Generate a default segment catalog for testing."""
        return {
            "SEG-TECH-EARLY": {
                "name": "Tech Early Adopters",
                "description": "Users who frequently engage with tech content and adopt new technologies early",
                "size": 2500000,
                "demographics": {"age": "25-44", "income": "high", "education": "college+"},
                "interests": ["technology", "gadgets", "startups", "software"],
            },
            "SEG-FINANCE-HNW": {
                "name": "High Net Worth Finance",
                "description": "High income individuals interested in investment and wealth management",
                "size": 1500000,
                "demographics": {"age": "35-64", "income": "very_high", "education": "college+"},
                "interests": ["investing", "finance", "stocks", "real estate"],
            },
            "SEG-AUTO-INTEND": {
                "name": "Auto Intenders",
                "description": "Users actively researching vehicle purchases",
                "size": 3000000,
                "demographics": {"age": "25-54", "income": "medium_high", "family": "mixed"},
                "interests": ["cars", "automotive", "ev", "trucks"],
            },
            "SEG-RETAIL-LUX": {
                "name": "Luxury Retail Shoppers",
                "description": "Affluent consumers who shop for premium brands",
                "size": 1800000,
                "demographics": {"age": "30-55", "income": "high", "location": "urban"},
                "interests": ["luxury", "fashion", "designer", "travel"],
            },
            "SEG-HEALTH-FIT": {
                "name": "Health & Fitness Enthusiasts",
                "description": "Active users interested in health, fitness, and wellness",
                "size": 4500000,
                "demographics": {"age": "18-44", "income": "medium", "lifestyle": "active"},
                "interests": ["fitness", "health", "nutrition", "sports"],
            },
            "SEG-PARENT-YOUNG": {
                "name": "Parents of Young Children",
                "description": "Parents with children under 12 years old",
                "size": 5000000,
                "demographics": {"age": "25-44", "income": "medium_high", "family": "children"},
                "interests": ["parenting", "education", "family", "toys"],
            },
            "SEG-BUSI-SMB": {
                "name": "SMB Decision Makers",
                "description": "Business owners and decision makers at small-medium businesses",
                "size": 2000000,
                "demographics": {"age": "30-54", "income": "high", "role": "decision_maker"},
                "interests": ["business", "b2b", "software", "marketing"],
            },
            "SEG-TRAVEL-FREQ": {
                "name": "Frequent Travelers",
                "description": "Users who travel frequently for business or leisure",
                "size": 3500000,
                "demographics": {"age": "25-54", "income": "high", "lifestyle": "mobile"},
                "interests": ["travel", "hotels", "airlines", "destinations"],
            },
        }
    
    def _calculate_segment_match_score(
        self,
        segment_data: dict,
        target_description: str,
        objective: str,
        geography: list,
        demographics: dict,
        interests: list,
    ) -> float:
        """Calculate match score between segment and targeting criteria."""
        score = 0.0
        
        # Text matching on description
        target_lower = target_description.lower()
        seg_name = segment_data.get("name", "").lower()
        seg_desc = segment_data.get("description", "").lower()
        seg_interests = [i.lower() for i in segment_data.get("interests", [])]
        
        # Name/description match - check individual words
        target_words = set(target_lower.split())
        for word in target_words:
            if len(word) > 3:  # Skip short words
                if word in seg_name:
                    score += 0.25
                if word in seg_desc:
                    score += 0.15
                # Also check if segment interests contain the word
                for seg_interest in seg_interests:
                    if word in seg_interest or seg_interest in word:
                        score += 0.15
        
        # Interest matching
        for interest in interests:
            if interest.lower() in seg_interests:
                score += 0.2
        
        # Interest keywords in target description
        for seg_interest in seg_interests:
            if seg_interest in target_lower:
                score += 0.15
        
        # Demographic matching
        seg_demo = segment_data.get("demographics", {})
        for key, value in demographics.items():
            if key in seg_demo and value.lower() in str(seg_demo[key]).lower():
                score += 0.15
        
        # Objective-based boost
        objective_boosts = {
            "awareness": ["tech", "travel", "health", "early", "enthusiasts"],
            "consideration": ["finance", "auto", "luxury", "intend", "hnw"],
            "conversion": ["retail", "parent", "business", "smb", "decision"],
        }
        
        for boost_word in objective_boosts.get(objective, []):
            if boost_word in seg_name.lower() or boost_word in seg_desc.lower():
                score += 0.15
        
        # Base score for all segments (minimum relevance)
        score += 0.1
        
        # Cap at 1.0
        return min(1.0, score)
