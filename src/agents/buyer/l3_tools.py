"""Tool definitions and registry for L3 Functional Agents.

This module defines the tools available to L3 agents and provides
a registry for tool lookup and schema generation.
"""

from dataclasses import dataclass
from typing import Any, Optional, Callable, Awaitable
from enum import Enum


# =============================================================================
# Data Types for Tool I/O
# =============================================================================


@dataclass
class SearchCriteria:
    """Criteria for product/inventory search."""
    
    channel: Optional[str] = None
    max_cpm: Optional[float] = None
    min_impressions: Optional[int] = None
    targeting: Optional[dict] = None
    deal_type: Optional[str] = None
    query: Optional[str] = None


@dataclass
class Product:
    """Advertising product/inventory item."""
    
    product_id: str
    seller_id: str
    name: str
    channel: str
    base_cpm: float
    floor_price: float
    available_impressions: int
    targeting: list[str]
    deal_type: str


@dataclass
class AvailsResult:
    """Result from availability check."""
    
    available: bool
    impressions: int
    cpm: Optional[float]
    deal_type: Optional[str]
    seller_id: str


@dataclass 
class OrderSpec:
    """Specification for creating an order."""
    
    campaign_id: str
    buyer_id: str
    name: str
    budget: float
    channel: str = "display"
    targeting: Optional[dict] = None


@dataclass
class Order:
    """Created order."""
    
    order_id: str
    campaign_id: str
    buyer_id: str
    name: str
    budget: float
    status: str
    created_at: str


@dataclass
class Deal:
    """Deal to book."""
    
    order_id: str
    seller_id: str
    product_id: str
    impressions: int
    cpm: float
    deal_type: str


@dataclass
class BookingConfirmation:
    """Confirmation of booked deal."""
    
    deal_id: str
    order_id: str
    seller_id: str
    impressions: int
    cpm: float
    total_cost: float
    status: str
    booked_at: str


@dataclass
class Metrics:
    """Campaign performance metrics."""
    
    campaign_id: str
    impressions: int
    clicks: int
    conversions: int
    spend: float
    ctr: float
    cpm: float
    cpa: Optional[float]
    period_start: str
    period_end: str


@dataclass
class Attribution:
    """Attribution analysis result."""
    
    model_type: str
    total_conversions: int
    attributed_value: float
    channels: dict[str, float]
    touchpoints: list[dict]


@dataclass
class CampaignBrief:
    """Brief for audience discovery."""
    
    campaign_id: str
    objective: str
    target_audience_description: str
    budget: float
    geography: Optional[list[str]] = None
    demographics: Optional[dict] = None


@dataclass
class AudienceSegment:
    """Discovered audience segment."""
    
    segment_id: str
    name: str
    description: str
    size: int
    match_score: float
    demographics: dict
    interests: list[str]


@dataclass
class CoverageEstimate:
    """Audience coverage estimation."""
    
    total_reach: int
    unique_users: int
    frequency: float
    coverage_percentage: float
    segments: list[dict]
    inventory_match: float


# =============================================================================
# Tool Schemas (Anthropic format)
# =============================================================================


class ToolSchema:
    """Tool schema generator for Anthropic API."""
    
    @staticmethod
    def product_search() -> dict:
        """ProductSearch tool schema."""
        return {
            "name": "ProductSearch",
            "description": "Search for available advertising inventory across sellers. "
                          "Use this to find products matching campaign requirements.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "Ad channel to search (display, video, ctv, native, mobile)",
                    },
                    "max_cpm": {
                        "type": "number",
                        "description": "Maximum CPM price to filter results",
                    },
                    "min_impressions": {
                        "type": "integer",
                        "description": "Minimum available impressions required",
                    },
                    "query": {
                        "type": "string",
                        "description": "Natural language search query",
                    },
                    "targeting": {
                        "type": "object",
                        "description": "Targeting requirements (geo, demo, interests)",
                    },
                },
                "required": [],
            },
        }
    
    @staticmethod
    def avails_check() -> dict:
        """AvailsCheck tool schema."""
        return {
            "name": "AvailsCheck",
            "description": "Check inventory availability with a specific seller. "
                          "Returns available impressions and pricing.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "Product ID to check availability for",
                    },
                    "seller_id": {
                        "type": "string",
                        "description": "Seller ID to check with",
                    },
                    "impressions": {
                        "type": "integer",
                        "description": "Number of impressions to check availability for",
                    },
                    "channel": {
                        "type": "string",
                        "description": "Ad channel (display, video, ctv)",
                    },
                },
                "required": ["impressions"],
            },
        }
    
    @staticmethod
    def pricing_lookup() -> dict:
        """PricingLookup tool schema."""
        return {
            "name": "PricingLookup",
            "description": "Get current pricing for inventory. Returns CPM ranges "
                          "and floor prices for products.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "Product ID to get pricing for",
                    },
                    "seller_id": {
                        "type": "string",
                        "description": "Seller ID",
                    },
                    "channel": {
                        "type": "string",
                        "description": "Ad channel",
                    },
                    "deal_type": {
                        "type": "string",
                        "description": "Deal type (OA, PD, PG)",
                    },
                },
                "required": [],
            },
        }
    
    @staticmethod
    def competitive_intel() -> dict:
        """CompetitiveIntel tool schema."""
        return {
            "name": "CompetitiveIntel",
            "description": "Get competitive intelligence and market analysis. "
                          "Returns market trends and competitor activity.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "Channel to analyze",
                    },
                    "timeframe": {
                        "type": "string",
                        "description": "Analysis timeframe (7d, 30d, 90d)",
                    },
                    "geography": {
                        "type": "string",
                        "description": "Geographic market to analyze",
                    },
                },
                "required": [],
            },
        }
    
    @staticmethod
    def create_order() -> dict:
        """CreateOrder tool schema."""
        return {
            "name": "CreateOrder",
            "description": "Create a new advertising order. This is the first step "
                          "before creating line items and booking inventory.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "campaign_id": {
                        "type": "string",
                        "description": "Campaign ID to associate order with",
                    },
                    "name": {
                        "type": "string",
                        "description": "Order name",
                    },
                    "budget": {
                        "type": "number",
                        "description": "Total order budget",
                    },
                    "channel": {
                        "type": "string",
                        "description": "Primary channel for this order",
                    },
                },
                "required": ["campaign_id", "name", "budget"],
            },
        }
    
    @staticmethod
    def create_line() -> dict:
        """CreateLine tool schema."""
        return {
            "name": "CreateLine",
            "description": "Create a line item within an order. Line items specify "
                          "targeting, budget allocation, and inventory selection.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Order ID to add line to",
                    },
                    "name": {
                        "type": "string",
                        "description": "Line item name",
                    },
                    "product_id": {
                        "type": "string",
                        "description": "Product/inventory to book",
                    },
                    "impressions": {
                        "type": "integer",
                        "description": "Impressions to book",
                    },
                    "cpm": {
                        "type": "number",
                        "description": "CPM bid price",
                    },
                    "targeting": {
                        "type": "object",
                        "description": "Targeting parameters",
                    },
                },
                "required": ["order_id", "name", "impressions", "cpm"],
            },
        }
    
    @staticmethod
    def book_line() -> dict:
        """BookLine tool schema."""
        return {
            "name": "BookLine",
            "description": "Book a line item to commit inventory. This finalizes "
                          "the deal and reserves the impressions.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "line_id": {
                        "type": "string",
                        "description": "Line item ID to book",
                    },
                    "order_id": {
                        "type": "string",
                        "description": "Order ID containing the line",
                    },
                    "seller_id": {
                        "type": "string",
                        "description": "Seller to book with",
                    },
                    "deal_type": {
                        "type": "string",
                        "description": "Deal type (OA, PD, PG)",
                    },
                },
                "required": ["line_id", "order_id", "seller_id"],
            },
        }
    
    @staticmethod
    def reserve_line() -> dict:
        """ReserveLine tool schema."""
        return {
            "name": "ReserveLine",
            "description": "Reserve inventory without full booking. Creates a hold "
                          "on impressions for a limited time.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "line_id": {
                        "type": "string",
                        "description": "Line item ID to reserve",
                    },
                    "order_id": {
                        "type": "string",
                        "description": "Order ID",
                    },
                    "hold_duration_hours": {
                        "type": "integer",
                        "description": "How long to hold the reservation",
                    },
                },
                "required": ["line_id", "order_id"],
            },
        }
    
    @staticmethod
    def get_metrics() -> dict:
        """GetMetrics tool schema."""
        return {
            "name": "GetMetrics",
            "description": "Pull performance metrics for a campaign. Returns "
                          "impressions, clicks, conversions, and cost data.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "campaign_id": {
                        "type": "string",
                        "description": "Campaign ID to get metrics for",
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date (YYYY-MM-DD)",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date (YYYY-MM-DD)",
                    },
                    "granularity": {
                        "type": "string",
                        "description": "Data granularity (hourly, daily, weekly)",
                    },
                },
                "required": ["campaign_id"],
            },
        }
    
    @staticmethod
    def generate_report() -> dict:
        """GenerateReport tool schema."""
        return {
            "name": "GenerateReport",
            "description": "Generate a performance report. Creates formatted "
                          "reports with charts and insights.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "campaign_id": {
                        "type": "string",
                        "description": "Campaign ID to report on",
                    },
                    "report_type": {
                        "type": "string",
                        "description": "Report type (performance, attribution, audience)",
                    },
                    "format": {
                        "type": "string",
                        "description": "Output format (json, csv, pdf)",
                    },
                    "date_range": {
                        "type": "string",
                        "description": "Date range (last_7d, last_30d, custom)",
                    },
                },
                "required": ["campaign_id", "report_type"],
            },
        }
    
    @staticmethod
    def attribution_analysis() -> dict:
        """AttributionAnalysis tool schema."""
        return {
            "name": "AttributionAnalysis",
            "description": "Run attribution modeling on conversions. Determines "
                          "which touchpoints contributed to conversions.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "campaign_id": {
                        "type": "string",
                        "description": "Campaign ID to analyze",
                    },
                    "model_type": {
                        "type": "string",
                        "description": "Attribution model (last_touch, first_touch, linear, time_decay)",
                    },
                    "conversion_window_days": {
                        "type": "integer",
                        "description": "Lookback window for attribution",
                    },
                },
                "required": ["campaign_id"],
            },
        }
    
    @staticmethod
    def audience_discovery() -> dict:
        """AudienceDiscovery tool schema."""
        return {
            "name": "AudienceDiscovery",
            "description": "Discover relevant audience segments for a campaign. "
                          "Finds segments matching target criteria.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "objective": {
                        "type": "string",
                        "description": "Campaign objective (awareness, consideration, conversion)",
                    },
                    "target_description": {
                        "type": "string",
                        "description": "Description of target audience",
                    },
                    "geography": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Target geographies",
                    },
                    "demographics": {
                        "type": "object",
                        "description": "Demographic criteria (age, gender, income)",
                    },
                    "interests": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Interest categories",
                    },
                },
                "required": ["target_description"],
            },
        }
    
    @staticmethod
    def audience_matching() -> dict:
        """AudienceMatching tool schema."""
        return {
            "name": "AudienceMatching",
            "description": "Match audience segments against available inventory. "
                          "Finds inventory that can reach target audiences.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "segment_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Audience segment IDs to match",
                    },
                    "channel": {
                        "type": "string",
                        "description": "Channel to search for matches",
                    },
                    "min_match_rate": {
                        "type": "number",
                        "description": "Minimum match rate (0-1)",
                    },
                },
                "required": ["segment_ids"],
            },
        }
    
    @staticmethod
    def coverage_estimation() -> dict:
        """CoverageEstimation tool schema."""
        return {
            "name": "CoverageEstimation",
            "description": "Estimate audience coverage and reach. Calculates "
                          "expected reach, frequency, and coverage percentage.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "segment_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Audience segments to include",
                    },
                    "inventory_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Inventory to estimate against",
                    },
                    "budget": {
                        "type": "number",
                        "description": "Budget for estimation",
                    },
                    "duration_days": {
                        "type": "integer",
                        "description": "Campaign duration",
                    },
                },
                "required": ["segment_ids"],
            },
        }


# =============================================================================
# Tool Registry
# =============================================================================


BUYER_TOOLS = {
    # Research tools
    "ProductSearch": ToolSchema.product_search,
    "AvailsCheck": ToolSchema.avails_check,
    "PricingLookup": ToolSchema.pricing_lookup,
    "CompetitiveIntel": ToolSchema.competitive_intel,
    
    # Execution tools
    "CreateOrder": ToolSchema.create_order,
    "CreateLine": ToolSchema.create_line,
    "BookLine": ToolSchema.book_line,
    "ReserveLine": ToolSchema.reserve_line,
    
    # Reporting tools
    "GetMetrics": ToolSchema.get_metrics,
    "GenerateReport": ToolSchema.generate_report,
    "AttributionAnalysis": ToolSchema.attribution_analysis,
    
    # Audience tools
    "AudienceDiscovery": ToolSchema.audience_discovery,
    "AudienceMatching": ToolSchema.audience_matching,
    "CoverageEstimation": ToolSchema.coverage_estimation,
}


def get_tool_schema(tool_name: str) -> dict:
    """Get schema for a tool by name.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Tool schema dict for Anthropic API
        
    Raises:
        KeyError: If tool not found
    """
    schema_fn = BUYER_TOOLS[tool_name]
    return schema_fn()


def get_tools_for_agent(tool_names: list[str]) -> list[dict]:
    """Get schemas for multiple tools.
    
    Args:
        tool_names: List of tool names
        
    Returns:
        List of tool schemas
    """
    return [get_tool_schema(name) for name in tool_names]
