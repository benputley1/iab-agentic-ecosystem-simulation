"""L3 Reporting Agent for Buyer Agent System.

The Reporting Agent handles analytics and performance reporting:
- GetMetrics: Pull campaign performance metrics
- GenerateReport: Create formatted reports
- AttributionAnalysis: Run attribution modeling
"""

from typing import Any, Optional
from datetime import datetime, timedelta
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
    Metrics,
    Attribution,
)


class ReportingAgent(FunctionalAgent[Metrics]):
    """Analytics and reporting functional agent.
    
    This L3 agent specializes in pulling metrics, generating reports,
    and analyzing attribution data.
    
    Tools:
        - GetMetrics: Pull performance metrics
        - GenerateReport: Create reports
        - AttributionAnalysis: Attribution modeling
    
    Example:
        ```python
        agent = ReportingAgent(context)
        
        # Get campaign metrics
        metrics = await agent.get_campaign_metrics("camp-001")
        
        # Run attribution analysis
        attribution = await agent.analyze_attribution(
            campaign_id="camp-001",
            model_type="time_decay",
        )
        
        # Generate report via natural language
        result = await agent.execute(
            "Generate a weekly performance report for campaign camp-001"
        )
        ```
    """
    
    TOOLS = ["GetMetrics", "GenerateReport", "AttributionAnalysis"]
    
    def __init__(
        self,
        context: AgentContext,
        metrics_store: Optional[dict] = None,
        **kwargs,
    ):
        """Initialize reporting agent.
        
        Args:
            context: Agent context with buyer/scenario info
            metrics_store: Optional pre-populated metrics store
            **kwargs: Additional args passed to FunctionalAgent
        """
        super().__init__(context, **kwargs)
        self._metrics_store = metrics_store or {}
        self._reports: dict[str, dict] = {}
    
    @property
    def system_prompt(self) -> str:
        """System prompt for reporting agent."""
        return f"""You are a Reporting Analyst for a programmatic advertising buyer.
Your role is to analyze campaign performance and generate insights.

Buyer ID: {self.context.buyer_id}
Scenario: {self.context.scenario}
Campaign: {self.context.campaign_id or "all campaigns"}

Your capabilities:
1. GetMetrics - Pull performance metrics (impressions, clicks, conversions)
2. GenerateReport - Create formatted reports with analysis
3. AttributionAnalysis - Run attribution modeling on conversions

Key metrics to track:
- Impressions delivered vs. target
- CTR (Click-through rate)
- CPM (Cost per thousand)
- CPA (Cost per acquisition)
- ROAS (Return on ad spend)

When analyzing performance:
- Compare against benchmarks
- Identify trends and anomalies
- Provide actionable recommendations
- Consider attribution models (last touch, first touch, linear, time decay)

Always provide data-driven insights with clear explanations."""
    
    @property
    def available_tools(self) -> list[dict]:
        """Tools available to this agent."""
        return [
            ToolSchema.get_metrics(),
            ToolSchema.generate_report(),
            ToolSchema.attribution_analysis(),
        ]
    
    async def _execute_tool(self, name: str, params: dict) -> ToolResult:
        """Execute a reporting tool.
        
        Args:
            name: Tool name
            params: Tool parameters from LLM
            
        Returns:
            ToolResult with execution outcome
        """
        if name == "GetMetrics":
            return await self._tool_get_metrics(params)
        elif name == "GenerateReport":
            return await self._tool_generate_report(params)
        elif name == "AttributionAnalysis":
            return await self._tool_attribution_analysis(params)
        else:
            return ToolResult(
                tool_name=name,
                status=ToolExecutionStatus.FAILED,
                error=f"Unknown tool: {name}",
            )
    
    # -------------------------------------------------------------------------
    # High-Level Methods
    # -------------------------------------------------------------------------
    
    async def get_campaign_metrics(
        self,
        campaign_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        granularity: str = "daily",
    ) -> Metrics:
        """Pull campaign performance metrics.
        
        Args:
            campaign_id: Campaign to get metrics for
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            granularity: Data granularity (hourly, daily, weekly)
            
        Returns:
            Campaign metrics
        """
        result = await self._tool_get_metrics({
            "campaign_id": campaign_id,
            "start_date": start_date,
            "end_date": end_date,
            "granularity": granularity,
        })
        
        if result.success and result.data:
            return result.data
        
        raise ValueError(f"Failed to get metrics: {result.error}")
    
    async def generate_report(
        self,
        campaign_id: str,
        report_type: str = "performance",
        format: str = "json",
        date_range: str = "last_7d",
    ) -> dict:
        """Generate a performance report.
        
        Args:
            campaign_id: Campaign to report on
            report_type: Type of report (performance, attribution, audience)
            format: Output format (json, csv, pdf)
            date_range: Date range (last_7d, last_30d, custom)
            
        Returns:
            Generated report
        """
        result = await self._tool_generate_report({
            "campaign_id": campaign_id,
            "report_type": report_type,
            "format": format,
            "date_range": date_range,
        })
        
        if result.success and result.data:
            return result.data
        
        raise ValueError(f"Failed to generate report: {result.error}")
    
    async def analyze_attribution(
        self,
        campaign_id: str,
        model_type: str = "last_touch",
        conversion_window_days: int = 30,
    ) -> Attribution:
        """Run attribution analysis on conversions.
        
        Args:
            campaign_id: Campaign to analyze
            model_type: Attribution model (last_touch, first_touch, linear, time_decay)
            conversion_window_days: Lookback window
            
        Returns:
            Attribution analysis results
        """
        result = await self._tool_attribution_analysis({
            "campaign_id": campaign_id,
            "model_type": model_type,
            "conversion_window_days": conversion_window_days,
        })
        
        if result.success and result.data:
            return result.data
        
        raise ValueError(f"Failed to analyze attribution: {result.error}")
    
    # -------------------------------------------------------------------------
    # Tool Implementations
    # -------------------------------------------------------------------------
    
    async def _tool_get_metrics(self, params: dict) -> ToolResult:
        """Execute GetMetrics tool."""
        try:
            campaign_id = params.get("campaign_id", "unknown")
            
            # Check if we have stored metrics
            if campaign_id in self._metrics_store:
                return ToolResult(
                    tool_name="GetMetrics",
                    status=ToolExecutionStatus.SUCCESS,
                    data=self._metrics_store[campaign_id],
                )
            
            # Generate realistic mock metrics
            base_impressions = random.randint(100000, 1000000)
            ctr = random.uniform(0.5, 2.5) / 100  # 0.5% - 2.5%
            clicks = int(base_impressions * ctr)
            conversion_rate = random.uniform(1.0, 5.0) / 100  # 1% - 5% of clicks
            conversions = int(clicks * conversion_rate)
            spend = base_impressions / 1000 * random.uniform(12.0, 30.0)
            
            # Calculate derived metrics
            cpm = spend / (base_impressions / 1000)
            cpa = spend / conversions if conversions > 0 else None
            
            # Determine date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=7)
            
            if params.get("start_date"):
                start_date = datetime.fromisoformat(params["start_date"])
            if params.get("end_date"):
                end_date = datetime.fromisoformat(params["end_date"])
            
            metrics = Metrics(
                campaign_id=campaign_id,
                impressions=base_impressions,
                clicks=clicks,
                conversions=conversions,
                spend=round(spend, 2),
                ctr=round(ctr * 100, 2),
                cpm=round(cpm, 2),
                cpa=round(cpa, 2) if cpa else None,
                period_start=start_date.isoformat(),
                period_end=end_date.isoformat(),
            )
            
            return ToolResult(
                tool_name="GetMetrics",
                status=ToolExecutionStatus.SUCCESS,
                data=metrics,
            )
            
        except Exception as e:
            return ToolResult(
                tool_name="GetMetrics",
                status=ToolExecutionStatus.FAILED,
                error=str(e),
            )
    
    async def _tool_generate_report(self, params: dict) -> ToolResult:
        """Execute GenerateReport tool."""
        try:
            campaign_id = params.get("campaign_id", "unknown")
            report_type = params.get("report_type", "performance")
            format = params.get("format", "json")
            date_range = params.get("date_range", "last_7d")
            
            report_id = f"RPT-{uuid.uuid4().hex[:8]}"
            
            # Get metrics for the report
            metrics_result = await self._tool_get_metrics({
                "campaign_id": campaign_id,
            })
            
            metrics = metrics_result.data if metrics_result.success else None
            
            # Build report based on type
            if report_type == "performance":
                report = self._build_performance_report(
                    report_id, campaign_id, metrics, date_range
                )
            elif report_type == "attribution":
                attr_result = await self._tool_attribution_analysis({
                    "campaign_id": campaign_id,
                })
                report = self._build_attribution_report(
                    report_id, campaign_id, attr_result.data, date_range
                )
            else:
                report = self._build_general_report(
                    report_id, campaign_id, report_type, date_range
                )
            
            # Store report
            self._reports[report_id] = report
            
            return ToolResult(
                tool_name="GenerateReport",
                status=ToolExecutionStatus.SUCCESS,
                data=report,
            )
            
        except Exception as e:
            return ToolResult(
                tool_name="GenerateReport",
                status=ToolExecutionStatus.FAILED,
                error=str(e),
            )
    
    async def _tool_attribution_analysis(self, params: dict) -> ToolResult:
        """Execute AttributionAnalysis tool."""
        try:
            campaign_id = params.get("campaign_id", "unknown")
            model_type = params.get("model_type", "last_touch")
            window_days = params.get("conversion_window_days", 30)
            
            # Generate realistic attribution data
            total_conversions = random.randint(50, 500)
            attributed_value = total_conversions * random.uniform(50, 200)
            
            # Channel attribution varies by model
            channels = {
                "display": 0.0,
                "video": 0.0,
                "search": 0.0,
                "social": 0.0,
                "email": 0.0,
            }
            
            if model_type == "last_touch":
                # Last touch gives most credit to converting channel
                channels["display"] = 0.45
                channels["search"] = 0.30
                channels["social"] = 0.15
                channels["video"] = 0.07
                channels["email"] = 0.03
            elif model_type == "first_touch":
                # First touch emphasizes awareness channels
                channels["display"] = 0.35
                channels["video"] = 0.25
                channels["social"] = 0.25
                channels["search"] = 0.10
                channels["email"] = 0.05
            elif model_type == "linear":
                # Linear distributes evenly
                channels["display"] = 0.20
                channels["video"] = 0.20
                channels["search"] = 0.20
                channels["social"] = 0.20
                channels["email"] = 0.20
            else:  # time_decay
                # Time decay weights recent touchpoints
                channels["display"] = 0.30
                channels["search"] = 0.35
                channels["social"] = 0.15
                channels["video"] = 0.12
                channels["email"] = 0.08
            
            # Generate touchpoint examples
            touchpoints = [
                {
                    "path": "display → video → search → conversion",
                    "count": int(total_conversions * 0.35),
                    "value": attributed_value * 0.35,
                },
                {
                    "path": "social → display → search → conversion",
                    "count": int(total_conversions * 0.25),
                    "value": attributed_value * 0.25,
                },
                {
                    "path": "email → display → conversion",
                    "count": int(total_conversions * 0.20),
                    "value": attributed_value * 0.20,
                },
                {
                    "path": "video → search → conversion",
                    "count": int(total_conversions * 0.20),
                    "value": attributed_value * 0.20,
                },
            ]
            
            attribution = Attribution(
                model_type=model_type,
                total_conversions=total_conversions,
                attributed_value=round(attributed_value, 2),
                channels={k: round(v, 2) for k, v in channels.items()},
                touchpoints=touchpoints,
            )
            
            return ToolResult(
                tool_name="AttributionAnalysis",
                status=ToolExecutionStatus.SUCCESS,
                data=attribution,
            )
            
        except Exception as e:
            return ToolResult(
                tool_name="AttributionAnalysis",
                status=ToolExecutionStatus.FAILED,
                error=str(e),
            )
    
    # -------------------------------------------------------------------------
    # Report Builders
    # -------------------------------------------------------------------------
    
    def _build_performance_report(
        self,
        report_id: str,
        campaign_id: str,
        metrics: Optional[Metrics],
        date_range: str,
    ) -> dict:
        """Build a performance report."""
        return {
            "report_id": report_id,
            "campaign_id": campaign_id,
            "report_type": "performance",
            "date_range": date_range,
            "generated_at": datetime.utcnow().isoformat(),
            "metrics": {
                "impressions": metrics.impressions if metrics else 0,
                "clicks": metrics.clicks if metrics else 0,
                "conversions": metrics.conversions if metrics else 0,
                "spend": metrics.spend if metrics else 0.0,
                "ctr": metrics.ctr if metrics else 0.0,
                "cpm": metrics.cpm if metrics else 0.0,
                "cpa": metrics.cpa if metrics else None,
            },
            "insights": [
                f"CTR of {metrics.ctr:.2f}% is {'above' if metrics and metrics.ctr > 1.0 else 'below'} industry average",
                f"Delivered {metrics.impressions:,} impressions" if metrics else "No data",
                "Consider A/B testing creative variations",
            ],
            "recommendations": [
                "Increase bids on high-performing placements",
                "Review underperforming creatives",
                "Expand targeting to similar audiences",
            ],
        }
    
    def _build_attribution_report(
        self,
        report_id: str,
        campaign_id: str,
        attribution: Optional[Attribution],
        date_range: str,
    ) -> dict:
        """Build an attribution report."""
        return {
            "report_id": report_id,
            "campaign_id": campaign_id,
            "report_type": "attribution",
            "date_range": date_range,
            "generated_at": datetime.utcnow().isoformat(),
            "model": attribution.model_type if attribution else "unknown",
            "total_conversions": attribution.total_conversions if attribution else 0,
            "attributed_value": attribution.attributed_value if attribution else 0.0,
            "channel_contribution": attribution.channels if attribution else {},
            "top_paths": attribution.touchpoints[:3] if attribution else [],
            "insights": [
                "Search channels show strong conversion influence",
                "Display is effective for awareness but needs support",
                "Video content drives consideration stage",
            ],
        }
    
    def _build_general_report(
        self,
        report_id: str,
        campaign_id: str,
        report_type: str,
        date_range: str,
    ) -> dict:
        """Build a general report."""
        return {
            "report_id": report_id,
            "campaign_id": campaign_id,
            "report_type": report_type,
            "date_range": date_range,
            "generated_at": datetime.utcnow().isoformat(),
            "status": "generated",
            "summary": f"{report_type.title()} report for {campaign_id}",
        }
