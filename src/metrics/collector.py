"""
InfluxDB Metric Collector for RTB Simulation.

Writes simulation metrics to InfluxDB for visualization in Grafana.
"""

import os
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS


@dataclass
class InfluxConfig:
    """InfluxDB connection configuration."""

    url: str = field(default_factory=lambda: os.getenv("INFLUX_URL", "http://localhost:8086"))
    token: str = field(default_factory=lambda: os.getenv("INFLUX_TOKEN", "rtb_sim_dev_token"))
    org: str = field(default_factory=lambda: os.getenv("INFLUX_ORG", "alkimi"))
    bucket: str = field(default_factory=lambda: os.getenv("INFLUX_BUCKET", "rtb_metrics"))


class MetricCollector:
    """
    Collects and writes RTB simulation metrics to InfluxDB.

    Supports batch writing for performance and individual writes for real-time updates.
    """

    def __init__(self, config: Optional[InfluxConfig] = None):
        self.config = config or InfluxConfig()
        self._client: Optional[InfluxDBClient] = None
        self._write_api = None
        self._batch: list[Point] = []

    def connect(self) -> None:
        """Establish connection to InfluxDB."""
        self._client = InfluxDBClient(
            url=self.config.url,
            token=self.config.token,
            org=self.config.org,
        )
        self._write_api = self._client.write_api(write_options=SYNCHRONOUS)

    def close(self) -> None:
        """Close connection and flush pending writes."""
        if self._batch:
            self.flush()
        if self._client:
            self._client.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def flush(self) -> None:
        """Write all batched points to InfluxDB."""
        if not self._batch:
            return
        if not self._write_api:
            raise RuntimeError("Not connected to InfluxDB. Call connect() first.")
        self._write_api.write(bucket=self.config.bucket, record=self._batch)
        self._batch.clear()

    def _write_point(self, point: Point, batch: bool = True) -> None:
        """Write a point, optionally batching."""
        if batch:
            self._batch.append(point)
        else:
            if not self._write_api:
                raise RuntimeError("Not connected to InfluxDB. Call connect() first.")
            self._write_api.write(bucket=self.config.bucket, record=point)

    # =========================================================================
    # Deal Metrics
    # =========================================================================

    def record_deal(
        self,
        deal_id: str,
        scenario: str,
        buyer_id: str,
        seller_id: str,
        buyer_spend: float,
        seller_revenue: float,
        exchange_fee: float,
        impressions: int,
        cpm: float,
        timestamp: Optional[datetime] = None,
        batch: bool = True,
    ) -> None:
        """
        Record a completed deal transaction.

        Args:
            deal_id: Unique deal identifier
            scenario: Scenario code (A, B, or C)
            buyer_id: Buyer agent ID
            seller_id: Seller agent ID
            buyer_spend: Total buyer payment in USD
            seller_revenue: Amount received by seller in USD
            exchange_fee: Intermediary fee extracted in USD
            impressions: Number of impressions in deal
            cpm: Cost per mille (CPM) rate
            timestamp: Event time (defaults to now)
            batch: Whether to batch the write
        """
        point = (
            Point("deals")
            .tag("scenario", scenario)
            .tag("buyer_id", buyer_id)
            .tag("seller_id", seller_id)
            .tag("deal_id", deal_id)
            .field("buyer_spend", float(buyer_spend))
            .field("seller_revenue", float(seller_revenue))
            .field("exchange_fee", float(exchange_fee))
            .field("impressions", int(impressions))
            .field("cpm", float(cpm))
            .field("count", 1)
            .time(timestamp or datetime.utcnow(), WritePrecision.MS)
        )
        self._write_point(point, batch)

    # =========================================================================
    # Campaign Metrics
    # =========================================================================

    def record_campaign_status(
        self,
        campaign_id: str,
        scenario: str,
        buyer_id: str,
        budget_total: float,
        budget_spent: float,
        target_impressions: int,
        actual_impressions: int,
        goal_achieved: bool,
        goal_attainment: float,
        timestamp: Optional[datetime] = None,
        batch: bool = True,
    ) -> None:
        """
        Record campaign progress/completion status.

        Args:
            campaign_id: Unique campaign identifier
            scenario: Scenario code (A, B, or C)
            buyer_id: Buyer agent ID
            budget_total: Total campaign budget in USD
            budget_spent: Amount spent so far in USD
            target_impressions: Campaign goal impressions
            actual_impressions: Impressions delivered
            goal_achieved: Whether campaign hit its target
            goal_attainment: Percentage of goal achieved (0.0-1.0)
            timestamp: Event time (defaults to now)
            batch: Whether to batch the write
        """
        point = (
            Point("campaigns")
            .tag("scenario", scenario)
            .tag("buyer_id", buyer_id)
            .tag("campaign_id", campaign_id)
            .field("budget_total", float(budget_total))
            .field("budget_spent", float(budget_spent))
            .field("target_impressions", int(target_impressions))
            .field("actual_impressions", int(actual_impressions))
            .field("goal_achieved", 1.0 if goal_achieved else 0.0)
            .field("goal_attainment", float(goal_attainment))
            .time(timestamp or datetime.utcnow(), WritePrecision.MS)
        )
        self._write_point(point, batch)

    # =========================================================================
    # Daily Aggregate Metrics
    # =========================================================================

    def record_daily_metrics(
        self,
        scenario: str,
        simulation_day: int,
        goal_attainment: float,
        context_losses: int,
        recovery_accuracy: float,
        active_campaigns: int,
        total_spend: float,
        timestamp: Optional[datetime] = None,
        batch: bool = True,
    ) -> None:
        """
        Record end-of-day aggregate metrics.

        Args:
            scenario: Scenario code (A, B, or C)
            simulation_day: Day number in simulation (1-30)
            goal_attainment: Average goal attainment across campaigns
            context_losses: Number of context rot events
            recovery_accuracy: Average state recovery fidelity (0.0-1.0)
            active_campaigns: Number of active campaigns
            total_spend: Total spend for the day
            timestamp: Event time (defaults to now)
            batch: Whether to batch the write
        """
        point = (
            Point("daily_metrics")
            .tag("scenario", scenario)
            .field("simulation_day", int(simulation_day))
            .field("goal_attainment", float(goal_attainment))
            .field("context_losses", int(context_losses))
            .field("recovery_accuracy", float(recovery_accuracy))
            .field("active_campaigns", int(active_campaigns))
            .field("total_spend", float(total_spend))
            .time(timestamp or datetime.utcnow(), WritePrecision.MS)
        )
        self._write_point(point, batch)

    # =========================================================================
    # Hallucination Tracking
    # =========================================================================

    def record_hallucination(
        self,
        agent_id: str,
        agent_type: str,
        scenario: str,
        claim_type: str,
        claimed_value: str,
        actual_value: str,
        timestamp: Optional[datetime] = None,
        batch: bool = True,
    ) -> None:
        """
        Record a detected hallucination event.

        Args:
            agent_id: Agent that made the false claim
            agent_type: Type of agent (buyer, seller, exchange)
            scenario: Scenario code (A, B, or C)
            claim_type: Type of claim (inventory_level, price, history, etc.)
            claimed_value: What the agent claimed
            actual_value: Ground truth value
            timestamp: Event time (defaults to now)
            batch: Whether to batch the write
        """
        point = (
            Point("hallucinations")
            .tag("scenario", scenario)
            .tag("agent_id", agent_id)
            .tag("agent_type", agent_type)
            .tag("claim_type", claim_type)
            .field("count", 1)
            .field("claimed_value", str(claimed_value))
            .field("actual_value", str(actual_value))
            .time(timestamp or datetime.utcnow(), WritePrecision.MS)
        )
        self._write_point(point, batch)

    def record_hallucination_rate(
        self,
        scenario: str,
        agent_type: str,
        total_decisions: int,
        hallucinated_decisions: int,
        rate: float,
        timestamp: Optional[datetime] = None,
        batch: bool = True,
    ) -> None:
        """
        Record aggregate hallucination rate.

        Args:
            scenario: Scenario code (A, B, or C)
            agent_type: Type of agent (buyer, seller, exchange)
            total_decisions: Total decisions made
            hallucinated_decisions: Decisions based on false data
            rate: Hallucination rate (0.0-100.0 percent)
            timestamp: Event time (defaults to now)
            batch: Whether to batch the write
        """
        point = (
            Point("hallucination_rates")
            .tag("scenario", scenario)
            .tag("agent_type", agent_type)
            .field("total_decisions", int(total_decisions))
            .field("hallucinated_decisions", int(hallucinated_decisions))
            .field("rate", float(rate))
            .time(timestamp or datetime.utcnow(), WritePrecision.MS)
        )
        self._write_point(point, batch)

    # =========================================================================
    # Context Rot Tracking
    # =========================================================================

    def record_context_rot(
        self,
        agent_id: str,
        scenario: str,
        simulation_day: int,
        keys_lost: int,
        recovery_attempted: bool,
        recovery_success: bool,
        recovery_accuracy: float,
        timestamp: Optional[datetime] = None,
        batch: bool = True,
    ) -> None:
        """
        Record a context rot event.

        Args:
            agent_id: Agent that lost context
            scenario: Scenario code (A, B, or C)
            simulation_day: Day number when rot occurred
            keys_lost: Number of memory keys lost
            recovery_attempted: Whether recovery was attempted
            recovery_success: Whether recovery succeeded
            recovery_accuracy: Fidelity of recovered state (0.0-1.0)
            timestamp: Event time (defaults to now)
            batch: Whether to batch the write
        """
        point = (
            Point("context_rot")
            .tag("scenario", scenario)
            .tag("agent_id", agent_id)
            .field("simulation_day", int(simulation_day))
            .field("keys_lost", int(keys_lost))
            .field("recovery_attempted", 1 if recovery_attempted else 0)
            .field("recovery_success", 1 if recovery_success else 0)
            .field("recovery_accuracy", float(recovery_accuracy))
            .field("events_count", 1)
            .time(timestamp or datetime.utcnow(), WritePrecision.MS)
        )
        self._write_point(point, batch)

    # =========================================================================
    # Blockchain Cost Tracking (Scenario C)
    # =========================================================================

    def record_blockchain_cost(
        self,
        transaction_id: str,
        transaction_type: str,
        payload_bytes: int,
        sui_gas: float,
        walrus_cost: float,
        total_usd: float,
        timestamp: Optional[datetime] = None,
        batch: bool = True,
    ) -> None:
        """
        Record blockchain transaction costs (Scenario C only).

        Args:
            transaction_id: Unique transaction/entry ID
            transaction_type: Type (bid_request, deal, delivery, etc.)
            payload_bytes: Size of payload in bytes
            sui_gas: Estimated Sui gas cost in SUI
            walrus_cost: Estimated Walrus storage cost in SUI
            total_usd: Total cost in USD
            timestamp: Event time (defaults to now)
            batch: Whether to batch the write
        """
        point = (
            Point("blockchain_costs")
            .tag("scenario", "C")
            .tag("transaction_type", transaction_type)
            .tag("transaction_id", transaction_id)
            .field("payload_bytes", int(payload_bytes))
            .field("sui_gas", float(sui_gas))
            .field("walrus_cost", float(walrus_cost))
            .field("total_sui", float(sui_gas + walrus_cost))
            .field("total_usd", float(total_usd))
            .field("count", 1)
            .time(timestamp or datetime.utcnow(), WritePrecision.MS)
        )
        self._write_point(point, batch)

    # =========================================================================
    # Agent Activity Tracking
    # =========================================================================

    def record_agent_decision(
        self,
        agent_id: str,
        agent_type: str,
        scenario: str,
        decision_type: str,
        decision_basis_verified: bool,
        timestamp: Optional[datetime] = None,
        batch: bool = True,
    ) -> None:
        """
        Record an agent decision for hallucination analysis.

        Args:
            agent_id: Agent making the decision
            agent_type: Type of agent
            scenario: Scenario code
            decision_type: Type of decision (bid, accept, reject, counter)
            decision_basis_verified: Whether decision data was verified against ground truth
            timestamp: Event time
            batch: Whether to batch
        """
        point = (
            Point("agent_decisions")
            .tag("scenario", scenario)
            .tag("agent_id", agent_id)
            .tag("agent_type", agent_type)
            .tag("decision_type", decision_type)
            .field("verified", 1 if decision_basis_verified else 0)
            .field("count", 1)
            .time(timestamp or datetime.utcnow(), WritePrecision.MS)
        )
        self._write_point(point, batch)
