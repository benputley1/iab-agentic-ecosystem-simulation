"""
KPI Calculations for RTB Simulation.

Provides calculations for the key metrics comparing scenarios A, B, and C.
"""

from dataclasses import dataclass
from typing import Optional

from influxdb_client import InfluxDBClient

from .collector import InfluxConfig


@dataclass
class FeeExtractionMetrics:
    """Fee extraction comparison metrics."""

    scenario: str
    gross_spend: float
    net_to_publisher: float
    intermediary_take: float
    take_rate_pct: float


@dataclass
class GoalAchievementMetrics:
    """Campaign goal achievement metrics."""

    scenario: str
    total_campaigns: int
    hit_impression_goal: int
    hit_cpm_goal: int
    avg_goal_attainment: float
    success_rate_pct: float


@dataclass
class ContextRotMetrics:
    """Context rot impact metrics."""

    scenario: str
    total_rot_events: int
    avg_keys_lost: float
    avg_recovery_accuracy: float
    day_1_attainment: float
    day_30_attainment: float
    degradation_pct: float


@dataclass
class HallucinationMetrics:
    """Hallucination rate metrics."""

    scenario: str
    agent_type: str
    total_decisions: int
    hallucinated_decisions: int
    hallucination_rate_pct: float


@dataclass
class BlockchainCostMetrics:
    """Blockchain cost metrics for Scenario C."""

    total_transactions: int
    total_sui_gas: float
    total_walrus_cost: float
    total_usd: float
    cost_per_1k_impressions: float
    comparison_exchange_fee_per_1k: float


class KPICalculator:
    """
    Calculates KPIs from InfluxDB metrics data.

    Queries InfluxDB and computes the key performance indicators
    for comparing RTB scenarios.
    """

    def __init__(self, config: Optional[InfluxConfig] = None):
        self.config = config or InfluxConfig()
        self._client: Optional[InfluxDBClient] = None

    def connect(self) -> None:
        """Establish connection to InfluxDB."""
        self._client = InfluxDBClient(
            url=self.config.url,
            token=self.config.token,
            org=self.config.org,
        )

    def close(self) -> None:
        """Close connection."""
        if self._client:
            self._client.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _query(self, flux_query: str) -> list:
        """Execute a Flux query and return results."""
        if not self._client:
            raise RuntimeError("Not connected to InfluxDB. Call connect() first.")
        query_api = self._client.query_api()
        return query_api.query(flux_query, org=self.config.org)

    # =========================================================================
    # Fee Extraction Comparison
    # =========================================================================

    def calculate_fee_extraction(self, scenario: str) -> FeeExtractionMetrics:
        """
        Calculate fee extraction metrics for a scenario.

        Implements the query from the implementation plan:
        - SUM(buyer_spend) as gross_spend
        - SUM(seller_revenue) as net_to_publisher
        - SUM(buyer_spend - seller_revenue) as intermediary_take
        - take_rate_pct
        """
        query = f'''
        from(bucket: "{self.config.bucket}")
          |> range(start: -30d)
          |> filter(fn: (r) => r._measurement == "deals")
          |> filter(fn: (r) => r.scenario == "{scenario}")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> group()
          |> reduce(
              fn: (r, accumulator) => ({{
                buyer_spend: accumulator.buyer_spend + r.buyer_spend,
                seller_revenue: accumulator.seller_revenue + r.seller_revenue
              }}),
              identity: {{buyer_spend: 0.0, seller_revenue: 0.0}}
          )
        '''
        tables = self._query(query)

        gross_spend = 0.0
        net_to_publisher = 0.0

        for table in tables:
            for record in table.records:
                gross_spend = record.values.get("buyer_spend", 0.0)
                net_to_publisher = record.values.get("seller_revenue", 0.0)

        intermediary_take = gross_spend - net_to_publisher
        take_rate_pct = (intermediary_take / gross_spend * 100) if gross_spend > 0 else 0.0

        return FeeExtractionMetrics(
            scenario=scenario,
            gross_spend=gross_spend,
            net_to_publisher=net_to_publisher,
            intermediary_take=intermediary_take,
            take_rate_pct=take_rate_pct,
        )

    def calculate_fee_extraction_all_scenarios(self) -> list[FeeExtractionMetrics]:
        """Calculate fee extraction for all scenarios."""
        return [
            self.calculate_fee_extraction("A"),
            self.calculate_fee_extraction("B"),
            self.calculate_fee_extraction("C"),
        ]

    # =========================================================================
    # Campaign Goal Achievement
    # =========================================================================

    def calculate_goal_achievement(self, scenario: str) -> GoalAchievementMetrics:
        """
        Calculate campaign goal achievement metrics for a scenario.

        Implements the query from the implementation plan:
        - COUNT(*) as total_campaigns
        - SUM(hit_impression_goal)
        - SUM(hit_cpm_goal)
        - AVG(goal_attainment_score)
        """
        query = f'''
        from(bucket: "{self.config.bucket}")
          |> range(start: -30d)
          |> filter(fn: (r) => r._measurement == "campaigns")
          |> filter(fn: (r) => r.scenario == "{scenario}")
          |> pivot(rowKey:["_time", "campaign_id"], columnKey: ["_field"], valueColumn: "_value")
          |> group()
          |> reduce(
              fn: (r, accumulator) => ({{
                count: accumulator.count + 1,
                goal_achieved_sum: accumulator.goal_achieved_sum + r.goal_achieved,
                goal_attainment_sum: accumulator.goal_attainment_sum + r.goal_attainment
              }}),
              identity: {{count: 0, goal_achieved_sum: 0.0, goal_attainment_sum: 0.0}}
          )
        '''
        tables = self._query(query)

        total_campaigns = 0
        hit_goal_sum = 0.0
        attainment_sum = 0.0

        for table in tables:
            for record in table.records:
                total_campaigns = int(record.values.get("count", 0))
                hit_goal_sum = record.values.get("goal_achieved_sum", 0.0)
                attainment_sum = record.values.get("goal_attainment_sum", 0.0)

        avg_attainment = attainment_sum / total_campaigns if total_campaigns > 0 else 0.0
        success_rate = (hit_goal_sum / total_campaigns * 100) if total_campaigns > 0 else 0.0

        return GoalAchievementMetrics(
            scenario=scenario,
            total_campaigns=total_campaigns,
            hit_impression_goal=int(hit_goal_sum),
            hit_cpm_goal=int(hit_goal_sum),  # Simplified - same metric for now
            avg_goal_attainment=avg_attainment,
            success_rate_pct=success_rate,
        )

    def calculate_goal_achievement_all_scenarios(self) -> list[GoalAchievementMetrics]:
        """Calculate goal achievement for all scenarios."""
        return [
            self.calculate_goal_achievement("A"),
            self.calculate_goal_achievement("B"),
            self.calculate_goal_achievement("C"),
        ]

    # =========================================================================
    # Context Rot Impact
    # =========================================================================

    def calculate_context_rot_impact(self, scenario: str) -> ContextRotMetrics:
        """
        Calculate context rot impact metrics for a scenario.

        Measures performance degradation over time due to context loss.
        """
        # Get total rot events and averages
        rot_query = f'''
        from(bucket: "{self.config.bucket}")
          |> range(start: -30d)
          |> filter(fn: (r) => r._measurement == "context_rot")
          |> filter(fn: (r) => r.scenario == "{scenario}")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> group()
          |> reduce(
              fn: (r, accumulator) => ({{
                count: accumulator.count + 1,
                keys_lost_sum: accumulator.keys_lost_sum + r.keys_lost,
                accuracy_sum: accumulator.accuracy_sum + r.recovery_accuracy
              }}),
              identity: {{count: 0, keys_lost_sum: 0, accuracy_sum: 0.0}}
          )
        '''

        # Get day 1 and day 30 attainment
        daily_query = f'''
        from(bucket: "{self.config.bucket}")
          |> range(start: -30d)
          |> filter(fn: (r) => r._measurement == "daily_metrics")
          |> filter(fn: (r) => r.scenario == "{scenario}")
          |> filter(fn: (r) => r._field == "goal_attainment")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> sort(columns: ["simulation_day"])
        '''

        rot_tables = self._query(rot_query)
        daily_tables = self._query(daily_query)

        total_events = 0
        keys_lost_sum = 0
        accuracy_sum = 0.0

        for table in rot_tables:
            for record in table.records:
                total_events = int(record.values.get("count", 0))
                keys_lost_sum = int(record.values.get("keys_lost_sum", 0))
                accuracy_sum = record.values.get("accuracy_sum", 0.0)

        avg_keys_lost = keys_lost_sum / total_events if total_events > 0 else 0.0
        avg_accuracy = accuracy_sum / total_events if total_events > 0 else 1.0

        day_1_attainment = 0.0
        day_30_attainment = 0.0

        for table in daily_tables:
            records = list(table.records)
            if records:
                day_1_attainment = records[0].values.get("goal_attainment", 0.0) if len(records) > 0 else 0.0
                day_30_attainment = records[-1].values.get("goal_attainment", 0.0) if len(records) > 0 else 0.0

        degradation = day_1_attainment - day_30_attainment

        return ContextRotMetrics(
            scenario=scenario,
            total_rot_events=total_events,
            avg_keys_lost=avg_keys_lost,
            avg_recovery_accuracy=avg_accuracy,
            day_1_attainment=day_1_attainment * 100,
            day_30_attainment=day_30_attainment * 100,
            degradation_pct=degradation * 100,
        )

    # =========================================================================
    # Hallucination Rates
    # =========================================================================

    def calculate_hallucination_rate(self, scenario: str, agent_type: str) -> HallucinationMetrics:
        """
        Calculate hallucination rate for a specific agent type in a scenario.
        """
        query = f'''
        from(bucket: "{self.config.bucket}")
          |> range(start: -30d)
          |> filter(fn: (r) => r._measurement == "agent_decisions")
          |> filter(fn: (r) => r.scenario == "{scenario}")
          |> filter(fn: (r) => r.agent_type == "{agent_type}")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> group()
          |> reduce(
              fn: (r, accumulator) => ({{
                total: accumulator.total + r.count,
                verified: accumulator.verified + r.verified
              }}),
              identity: {{total: 0, verified: 0}}
          )
        '''

        tables = self._query(query)

        total_decisions = 0
        verified_decisions = 0

        for table in tables:
            for record in table.records:
                total_decisions = int(record.values.get("total", 0))
                verified_decisions = int(record.values.get("verified", 0))

        hallucinated = total_decisions - verified_decisions
        rate = (hallucinated / total_decisions * 100) if total_decisions > 0 else 0.0

        return HallucinationMetrics(
            scenario=scenario,
            agent_type=agent_type,
            total_decisions=total_decisions,
            hallucinated_decisions=hallucinated,
            hallucination_rate_pct=rate,
        )

    def calculate_hallucination_rates_all(self) -> list[HallucinationMetrics]:
        """Calculate hallucination rates for all scenario/agent combinations."""
        results = []
        for scenario in ["A", "B", "C"]:
            for agent_type in ["buyer", "seller", "exchange"]:
                results.append(self.calculate_hallucination_rate(scenario, agent_type))
        return results

    # =========================================================================
    # Blockchain Costs (Scenario C)
    # =========================================================================

    def calculate_blockchain_costs(self, impressions_total: int = 0) -> BlockchainCostMetrics:
        """
        Calculate total blockchain costs for Scenario C.
        """
        query = f'''
        from(bucket: "{self.config.bucket}")
          |> range(start: -30d)
          |> filter(fn: (r) => r._measurement == "blockchain_costs")
          |> filter(fn: (r) => r.scenario == "C")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> group()
          |> reduce(
              fn: (r, accumulator) => ({{
                count: accumulator.count + r.count,
                sui_gas: accumulator.sui_gas + r.sui_gas,
                walrus_cost: accumulator.walrus_cost + r.walrus_cost,
                total_usd: accumulator.total_usd + r.total_usd
              }}),
              identity: {{count: 0, sui_gas: 0.0, walrus_cost: 0.0, total_usd: 0.0}}
          )
        '''

        tables = self._query(query)

        total_transactions = 0
        total_sui_gas = 0.0
        total_walrus = 0.0
        total_usd = 0.0

        for table in tables:
            for record in table.records:
                total_transactions = int(record.values.get("count", 0))
                total_sui_gas = record.values.get("sui_gas", 0.0)
                total_walrus = record.values.get("walrus_cost", 0.0)
                total_usd = record.values.get("total_usd", 0.0)

        # Calculate cost per 1k impressions
        # Assuming each transaction batches ~1000 impressions
        cost_per_1k = total_usd / total_transactions if total_transactions > 0 else 0.0

        return BlockchainCostMetrics(
            total_transactions=total_transactions,
            total_sui_gas=total_sui_gas,
            total_walrus_cost=total_walrus,
            total_usd=total_usd,
            cost_per_1k_impressions=cost_per_1k,
            comparison_exchange_fee_per_1k=2.50,  # Typical $15 CPM * 15% = $2.25
        )

    # =========================================================================
    # Summary Report
    # =========================================================================

    def generate_summary(self) -> dict:
        """
        Generate a complete summary of all KPIs.

        Returns a dictionary suitable for report generation.
        """
        fee_metrics = self.calculate_fee_extraction_all_scenarios()
        goal_metrics = self.calculate_goal_achievement_all_scenarios()
        context_rot_b = self.calculate_context_rot_impact("B")
        context_rot_c = self.calculate_context_rot_impact("C")
        blockchain_costs = self.calculate_blockchain_costs()

        # Calculate fee reduction from A to C
        fee_a = next((m for m in fee_metrics if m.scenario == "A"), None)
        fee_c = next((m for m in fee_metrics if m.scenario == "C"), None)

        fee_reduction = 0.0
        savings_per_100k = 0.0
        if fee_a and fee_c and fee_a.take_rate_pct > 0:
            fee_reduction = ((fee_a.take_rate_pct - fee_c.take_rate_pct) / fee_a.take_rate_pct) * 100
            savings_per_100k = (fee_a.take_rate_pct - fee_c.take_rate_pct) * 1000

        return {
            "fee_comparison": [
                {
                    "scenario": m.scenario,
                    "spend": m.gross_spend,
                    "take": m.intermediary_take,
                    "rate": m.take_rate_pct,
                }
                for m in fee_metrics
            ],
            "goal_achievement": [
                {
                    "scenario": m.scenario,
                    "total": m.total_campaigns,
                    "hit": m.hit_impression_goal,
                    "rate": m.success_rate_pct,
                }
                for m in goal_metrics
            ],
            "fee_reduction": fee_reduction,
            "savings_per_100k": savings_per_100k,
            "day1_achievement": context_rot_b.day_1_attainment,
            "day30_achievement": context_rot_b.day_30_attainment,
            "degradation": context_rot_b.degradation_pct,
            "blockchain_cost_usd": blockchain_costs.total_usd,
            "cost_per_1k": blockchain_costs.cost_per_1k_impressions,
            "sui_gas": blockchain_costs.total_sui_gas,
            "walrus_cost": blockchain_costs.total_walrus_cost,
        }
