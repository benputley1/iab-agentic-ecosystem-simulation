"""
V2 Orchestrator - Integrates all V2 components for context hallucination simulation.

This orchestrator coordinates:
- TokenPressureEngine: Track context window pressure
- GroundTruthDB: Authoritative record of events
- HallucinationClassifier: Detect and classify hallucinations
- RealisticVolumeGenerator: Generate realistic bid volumes
- DecisionChainTracker: Track decision dependencies
- AgentRestartSimulator: Simulate crashes and recovery

The V2 simulation tests the hypothesis that context window pressure
causes non-linear growth in hallucination rate over campaign duration.
"""

import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from enum import Enum

from pressure.token_tracker import TokenPressureEngine, TokenPressureResult
from ground_truth.db import GroundTruthDB, Event, EventType, CampaignState
from hallucination.classifier import (
    HallucinationClassifier,
    HallucinationType,
    Hallucination,
    HallucinationResult,
    AgentDecisionForCheck,
)
from volume.generator import RealisticVolumeGenerator, BidRequest, VolumeProfile
from tracking.decision_chain import (
    DecisionChainTracker,
    AgentDecision,
    DecisionReference,
    ChainResult,
    ReferenceErrorType,
)
from resilience.restart import (
    AgentRestartSimulator,
    MockAgent,
    RestartEvent,
    AgentState,
)


@dataclass
class V2Config:
    """Configuration for V2 simulation with all feature flags."""
    
    # Simulation parameters
    volume_profile: str = "medium"  # small, medium, large, enterprise
    
    # Token pressure settings
    enable_token_pressure: bool = True
    context_limit: int = 200_000
    compression_loss_rate: float = 0.20
    
    # Hallucination detection settings
    enable_hallucination_detection: bool = True
    budget_drift_threshold: float = 0.01  # 1% tolerance
    price_anchor_threshold: float = 0.05  # 5% tolerance
    
    # Decision chain tracking
    enable_decision_tracking: bool = True
    lookback_window: int = 100
    
    # Restart simulation
    enable_restart_simulation: bool = True
    crash_probability: float = 0.01  # Per-hour
    recovery_modes: List[str] = field(default_factory=lambda: ["private_db", "ledger"])
    
    # Agent configuration
    num_campaigns: int = 3
    initial_budget_per_campaign: float = 10000.0
    
    # Hallucination injection (for testing the detector)
    inject_hallucinations: bool = True
    base_hallucination_rate: float = 0.001  # 0.1% base rate
    hallucination_growth_factor: float = 1.15  # Compounds daily
    
    # Ground truth database
    db_path: str = ":memory:"
    
    # Random seed for reproducibility
    random_seed: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration."""
        valid_profiles = ["small", "medium", "large", "enterprise"]
        if self.volume_profile not in valid_profiles:
            raise ValueError(f"volume_profile must be one of {valid_profiles}")
        
        if not 0 <= self.base_hallucination_rate <= 1:
            raise ValueError("base_hallucination_rate must be between 0 and 1")
        
        if self.hallucination_growth_factor < 1:
            raise ValueError("hallucination_growth_factor must be >= 1")


@dataclass
class DailyMetrics:
    """Metrics collected for a single simulation day."""
    
    day: int
    date: datetime
    
    # Volume metrics
    requests_generated: int = 0
    decisions_made: int = 0
    
    # Hallucination metrics
    hallucinations_detected: int = 0
    hallucination_rate: float = 0.0
    hallucination_types: Dict[str, int] = field(default_factory=dict)
    
    # Token pressure metrics
    overflow_events: int = 0
    tokens_at_end: int = 0
    cumulative_info_loss: float = 0.0
    
    # Decision chain metrics
    reference_failures: int = 0
    cascading_errors: int = 0
    reference_accuracy: float = 1.0
    
    # Restart metrics
    restart_events: int = 0
    avg_recovery_accuracy: Dict[str, float] = field(default_factory=dict)


@dataclass
class V2SimulationResult:
    """Comprehensive result from V2 simulation."""
    
    # Configuration used
    config: V2Config
    
    # Simulation metadata
    simulation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    days_simulated: int = 0
    status: str = "pending"
    
    # Per-day metrics
    daily_metrics: List[DailyMetrics] = field(default_factory=list)
    
    # Aggregate metrics
    total_requests: int = 0
    total_decisions: int = 0
    total_hallucinations: int = 0
    cumulative_hallucination_rate: float = 0.0
    
    # Token pressure aggregate
    total_overflow_events: int = 0
    final_info_retention: float = 100.0
    
    # Decision chain aggregate
    total_reference_failures: int = 0
    total_cascading_errors: int = 0
    overall_reference_accuracy: float = 1.0
    
    # Restart aggregate
    total_restart_events: int = 0
    recovery_comparison: Dict[str, float] = field(default_factory=dict)
    
    # Hallucination type distribution
    hallucination_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Critical findings
    critical_threshold_day: Optional[int] = None  # Day when hallucination rate exceeded 1%
    peak_hallucination_rate: float = 0.0
    peak_hallucination_day: int = 0
    
    def get_hallucination_curve(self) -> List[Dict[str, Any]]:
        """Get hallucination rate over time for charting."""
        return [
            {
                "day": m.day,
                "rate": m.hallucination_rate,
                "count": m.hallucinations_detected,
                "decisions": m.decisions_made,
            }
            for m in self.daily_metrics
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "simulation_id": self.simulation_id,
            "status": self.status,
            "days_simulated": self.days_simulated,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "config": {
                "volume_profile": self.config.volume_profile,
                "context_limit": self.config.context_limit,
                "num_campaigns": self.config.num_campaigns,
            },
            "totals": {
                "requests": self.total_requests,
                "decisions": self.total_decisions,
                "hallucinations": self.total_hallucinations,
                "hallucination_rate": self.cumulative_hallucination_rate,
            },
            "token_pressure": {
                "overflow_events": self.total_overflow_events,
                "info_retention_pct": self.final_info_retention,
            },
            "decision_chain": {
                "reference_failures": self.total_reference_failures,
                "cascading_errors": self.total_cascading_errors,
                "reference_accuracy": self.overall_reference_accuracy,
            },
            "restarts": {
                "total_events": self.total_restart_events,
                "recovery_comparison": self.recovery_comparison,
            },
            "hallucination_distribution": self.hallucination_distribution,
            "critical_findings": {
                "threshold_day": self.critical_threshold_day,
                "peak_rate": self.peak_hallucination_rate,
                "peak_day": self.peak_hallucination_day,
            },
            "daily_metrics": [
                {
                    "day": m.day,
                    "hallucination_rate": m.hallucination_rate,
                    "overflow_events": m.overflow_events,
                    "reference_accuracy": m.reference_accuracy,
                }
                for m in self.daily_metrics
            ],
        }


class SimulatedAgent:
    """Agent that makes decisions with potential hallucinations."""
    
    def __init__(
        self,
        agent_id: str,
        campaign_id: str,
        initial_budget: float,
        hallucination_rate: float = 0.001,
    ):
        self.agent_id = agent_id
        self.campaign_id = campaign_id
        self.initial_budget = initial_budget
        self.hallucination_rate = hallucination_rate
        
        # Simulated internal state (what agent "believes")
        self._believed_spend = 0.0
        self._believed_impressions = 0
        self._decision_count = 0
        
        # References to previous decisions
        self._recent_decisions: List[str] = []
    
    def process_request(
        self,
        request: BidRequest,
        actual_spend: float,
        actual_impressions: int,
        rng: random.Random,
    ) -> tuple[AgentDecisionForCheck, Optional[float], Optional[int]]:
        """
        Process a bid request and return a decision.
        
        May introduce hallucinations based on configured rate.
        
        Returns:
            Tuple of (decision, hallucinated_spend, hallucinated_impressions)
        """
        self._decision_count += 1
        decision_id = f"{self.agent_id}-{self._decision_count}"
        
        # Determine if this decision will have hallucinations
        will_hallucinate = rng.random() < self.hallucination_rate
        
        # Calculate believed values (may drift from actual)
        believed_spend = actual_spend
        believed_impressions = actual_impressions
        believed_floor = request.floor_price
        
        if will_hallucinate:
            # Inject various hallucination types
            hallucination_type = rng.choice([
                "budget_drift",
                "price_anchor",
                "impressions",
            ])
            
            if hallucination_type == "budget_drift":
                # Drift spend by 2-10%
                drift = rng.uniform(0.02, 0.10)
                direction = rng.choice([-1, 1])
                believed_spend = actual_spend * (1 + direction * drift)
            
            elif hallucination_type == "price_anchor":
                # Wrong floor price by 10-30%
                drift = rng.uniform(0.10, 0.30)
                direction = rng.choice([-1, 1])
                believed_floor = request.floor_price * (1 + direction * drift)
            
            elif hallucination_type == "impressions":
                # Wrong impression count by 5-15%
                drift = rng.uniform(0.05, 0.15)
                direction = rng.choice([-1, 1])
                believed_impressions = int(actual_impressions * (1 + direction * drift))
        
        # Update believed state
        self._believed_spend += believed_spend
        self._believed_impressions += believed_impressions
        
        # Create decision for checking
        decision = AgentDecisionForCheck(
            decision_id=decision_id,
            timestamp=request.timestamp,
            campaign_id=self.campaign_id,
            agent_id=self.agent_id,
            budget_remaining=self.initial_budget - self._believed_spend,
            total_spend=self._believed_spend,
            impressions_claimed=self._believed_impressions,
            floor_price_used=believed_floor,
            publisher_id=request.publisher_id,
            metadata={
                "request_id": request.request_id,
                "initial_budget": self.initial_budget,
            }
        )
        
        # Track recent decisions
        self._recent_decisions.append(decision_id)
        if len(self._recent_decisions) > 10:
            self._recent_decisions.pop(0)
        
        return decision, believed_spend if will_hallucinate else None, believed_impressions if will_hallucinate else None
    
    def get_recent_decision_ids(self) -> List[str]:
        """Get IDs of recent decisions for reference tracking."""
        return self._recent_decisions.copy()
    
    def reset_state(self) -> None:
        """Reset agent's believed state (simulates memory loss)."""
        # Partial reset - simulates incomplete recovery
        self._believed_spend *= 0.9  # Lose 10% of spend tracking
        self._believed_impressions = int(self._believed_impressions * 0.95)


class V2Orchestrator:
    """
    Orchestrate V2 simulation with all components integrated.
    
    Coordinates:
    - Volume generation
    - Agent decision making
    - Token pressure tracking
    - Hallucination detection
    - Decision chain tracking
    - Crash simulation and recovery
    """
    
    def __init__(self, config: V2Config):
        """
        Initialize the V2 orchestrator.
        
        Args:
            config: V2Config with all feature flags and parameters
        """
        self.config = config
        
        # Initialize random state
        self._rng = random.Random(config.random_seed)
        
        # Track simulation state
        self._current_day = 0
        self._base_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Initialize components based on config
        self._init_components()
    
    def _init_components(self) -> None:
        """Initialize all V2 components."""
        # Ground truth database (always needed)
        self.ground_truth = GroundTruthDB(self.config.db_path)
        
        # Token pressure engine
        if self.config.enable_token_pressure:
            self.token_engine = TokenPressureEngine(
                context_limit=self.config.context_limit,
                compression_loss=self.config.compression_loss_rate,
            )
        else:
            self.token_engine = None
        
        # Hallucination classifier
        if self.config.enable_hallucination_detection:
            self.hallucination_classifier = HallucinationClassifier(
                ground_truth=self.ground_truth,
                budget_drift_threshold=self.config.budget_drift_threshold,
                price_anchor_threshold=self.config.price_anchor_threshold,
            )
        else:
            self.hallucination_classifier = None
        
        # Decision chain tracker
        if self.config.enable_decision_tracking:
            self.decision_tracker = DecisionChainTracker(
                lookback_window=self.config.lookback_window
            )
        else:
            self.decision_tracker = None
        
        # Restart simulator
        if self.config.enable_restart_simulation:
            self.restart_simulator = AgentRestartSimulator(
                crash_probability=self.config.crash_probability,
                recovery_modes=self.config.recovery_modes,
                random_seed=self.config.random_seed,
            )
        else:
            self.restart_simulator = None
        
        # Volume generator
        self.volume_generator = RealisticVolumeGenerator(
            seed=self.config.random_seed,
            base_date=self._base_date,
        )
        
        # Initialize agents
        self._agents: Dict[str, SimulatedAgent] = {}
        self._mock_agents: Dict[str, MockAgent] = {}  # For restart simulation
        
        for i in range(self.config.num_campaigns):
            campaign_id = f"campaign-{i:03d}"
            agent_id = f"agent-{i:03d}"
            
            self._agents[campaign_id] = SimulatedAgent(
                agent_id=agent_id,
                campaign_id=campaign_id,
                initial_budget=self.config.initial_budget_per_campaign,
                hallucination_rate=self.config.base_hallucination_rate,
            )
            
            if self.config.enable_restart_simulation:
                self._mock_agents[agent_id] = MockAgent(
                    agent_id=agent_id,
                    initial_budget=self.config.initial_budget_per_campaign,
                    campaign_id=campaign_id,
                )
    
    def run_simulation(self, days: int) -> V2SimulationResult:
        """
        Run the full V2 simulation.
        
        Args:
            days: Number of days to simulate
            
        Returns:
            V2SimulationResult with comprehensive metrics
        """
        result = V2SimulationResult(config=self.config)
        result.days_simulated = days
        result.status = "running"
        
        try:
            for day in range(days):
                self._current_day = day
                
                # Update hallucination rates (non-linear growth)
                self._update_hallucination_rates(day)
                
                # Run one day of simulation
                daily_metrics = self._simulate_day(day)
                result.daily_metrics.append(daily_metrics)
                
                # Update aggregate metrics
                self._update_aggregate_metrics(result, daily_metrics)
                
                # Check for critical threshold
                if daily_metrics.hallucination_rate > 0.01 and result.critical_threshold_day is None:
                    result.critical_threshold_day = day + 1
                
                # Track peak
                if daily_metrics.hallucination_rate > result.peak_hallucination_rate:
                    result.peak_hallucination_rate = daily_metrics.hallucination_rate
                    result.peak_hallucination_day = day + 1
            
            result.status = "completed"
            
        except Exception as e:
            result.status = f"failed: {str(e)}"
            raise
        
        finally:
            result.end_time = datetime.utcnow()
            self._finalize_metrics(result)
        
        return result
    
    def _update_hallucination_rates(self, day: int) -> None:
        """Update hallucination rates based on day (non-linear growth)."""
        if not self.config.inject_hallucinations:
            return
        
        # Non-linear growth: base_rate * growth_factor^day
        # This creates the expected curve where errors accelerate after day 7-10
        current_rate = self.config.base_hallucination_rate * (
            self.config.hallucination_growth_factor ** day
        )
        
        # Cap at 20% to keep simulation realistic
        current_rate = min(0.20, current_rate)
        
        for agent in self._agents.values():
            agent.hallucination_rate = current_rate
    
    def _simulate_day(self, day: int) -> DailyMetrics:
        """Simulate one day of the campaign."""
        metrics = DailyMetrics(
            day=day + 1,  # 1-indexed for human readability
            date=self._base_date + timedelta(days=day),
        )
        
        # Generate bid requests for the day
        requests = self.volume_generator.generate_day(
            profile=self.config.volume_profile,
            day_number=day,
        )
        metrics.requests_generated = len(requests)
        
        # Track daily stats
        day_hallucinations = 0
        day_decisions = 0
        day_overflows = 0
        day_reference_failures = 0
        day_cascading = 0
        
        # Process each request
        for request in requests:
            # Select campaign/agent for this request
            campaign_id = self._rng.choice(list(self._agents.keys()))
            agent = self._agents[campaign_id]
            
            # Determine actual values (what really happened)
            bid_amount = self._rng.uniform(request.floor_price, request.floor_price * 2)
            won = self._rng.random() < 0.3  # 30% win rate
            
            if won:
                actual_spend = bid_amount
                actual_impressions = 1
            else:
                actual_spend = 0.0
                actual_impressions = 0
            
            # Record to ground truth
            if won:
                event = Event.create(
                    event_type=EventType.SPEND,
                    campaign_id=campaign_id,
                    amount=actual_spend,
                    timestamp=request.timestamp,
                )
                self.ground_truth.record_event(event)
                
                imp_event = Event.create(
                    event_type=EventType.IMPRESSION,
                    campaign_id=campaign_id,
                    amount=1,
                    timestamp=request.timestamp,
                )
                self.ground_truth.record_event(imp_event)
            
            # Agent processes request (may hallucinate)
            decision, h_spend, h_imp = agent.process_request(
                request=request,
                actual_spend=actual_spend,
                actual_impressions=actual_impressions,
                rng=self._rng,
            )
            day_decisions += 1
            
            # Token pressure tracking
            if self.token_engine:
                pressure_result = self.token_engine.add_event(
                    {"request": request.to_dict(), "decision": decision.decision_id, "agent_id": agent.agent_id}
                )
                if pressure_result.overflow:
                    day_overflows += 1
            
            # Hallucination detection
            if self.hallucination_classifier:
                self.hallucination_classifier.register_floor_price(
                    request.request_id, request.floor_price
                )
                self.hallucination_classifier.register_publisher(request.publisher_id)
                
                check_result = self.hallucination_classifier.check_decision(
                    decision,
                    actual_floor_price=request.floor_price,
                )
                if check_result.has_hallucinations:
                    day_hallucinations += len(check_result.errors)
            
            # Decision chain tracking
            if self.decision_tracker:
                # Create decision with references to recent decisions
                chain_decision = AgentDecision(
                    id=decision.decision_id,
                    timestamp=decision.timestamp,
                    agent_id=agent.agent_id,
                    decision_type="bid" if won else "no_bid",
                    value=bid_amount,
                    references=self._create_references(agent, decision),
                )
                chain_result = self.decision_tracker.record_decision(chain_decision)
                day_reference_failures += len(chain_result.failures)
                day_cascading += chain_result.cascading_errors
        
        # Check for agent restarts (hourly check, simulate 24 hours)
        day_restarts = 0
        if self.restart_simulator:
            for hour in range(24):
                sim_hour = day * 24 + hour
                for agent_id, mock_agent in self._mock_agents.items():
                    restart_event = self.restart_simulator.maybe_crash(
                        mock_agent, sim_hour
                    )
                    if restart_event:
                        day_restarts += 1
                        # Sync state loss to simulated agent
                        campaign_id = mock_agent.campaign_id
                        if campaign_id in self._agents:
                            self._agents[campaign_id].reset_state()
        
        # Update metrics
        metrics.decisions_made = day_decisions
        metrics.hallucinations_detected = day_hallucinations
        metrics.hallucination_rate = day_hallucinations / max(day_decisions, 1)
        metrics.overflow_events = day_overflows
        metrics.reference_failures = day_reference_failures
        metrics.cascading_errors = day_cascading
        metrics.restart_events = day_restarts
        
        if self.decision_tracker:
            metrics.reference_accuracy = self.decision_tracker.reference_accuracy_rate
        
        if self.token_engine:
            stats = self.token_engine.get_stats()
            metrics.cumulative_info_loss = stats.get("utilization_pct", 0.0)
            metrics.tokens_at_end = stats.get("current_tokens", 0)
        
        if self.hallucination_classifier:
            type_dist = self.hallucination_classifier.get_type_distribution()
            metrics.hallucination_types = {t.value: c for t, c in type_dist.items()}
        
        if self.restart_simulator:
            summary = self.restart_simulator.get_summary()
            metrics.avg_recovery_accuracy = summary.get("avg_accuracy_by_mode", {})
        
        return metrics
    
    def _create_references(
        self,
        agent: SimulatedAgent,
        decision: AgentDecisionForCheck,
    ) -> List[DecisionReference]:
        """Create references to recent decisions (with potential errors)."""
        references = []
        recent_ids = agent.get_recent_decision_ids()
        
        # Reference 1-3 recent decisions
        num_refs = min(len(recent_ids), self._rng.randint(1, 3))
        
        for i in range(num_refs):
            if i < len(recent_ids):
                ref_id = recent_ids[-(i + 1)]  # Most recent first
                
                # The recalled value might be wrong (hallucination in references)
                if self._rng.random() < agent.hallucination_rate:
                    recalled = self._rng.uniform(0, 100)  # Wrong value
                else:
                    # Try to get actual value
                    actual_decision = self.decision_tracker.get_decision(ref_id) if self.decision_tracker else None
                    recalled = actual_decision.value if actual_decision else 0
                
                references.append(DecisionReference(
                    decision_id=ref_id,
                    recalled_value=recalled,
                    field_name="value",
                ))
        
        return references
    
    def _update_aggregate_metrics(
        self,
        result: V2SimulationResult,
        daily: DailyMetrics,
    ) -> None:
        """Update aggregate metrics in result."""
        result.total_requests += daily.requests_generated
        result.total_decisions += daily.decisions_made
        result.total_hallucinations += daily.hallucinations_detected
        result.total_overflow_events += daily.overflow_events
        result.total_reference_failures += daily.reference_failures
        result.total_cascading_errors += daily.cascading_errors
        result.total_restart_events += daily.restart_events
    
    def _finalize_metrics(self, result: V2SimulationResult) -> None:
        """Finalize all metrics after simulation completes."""
        # Calculate cumulative hallucination rate
        if result.total_decisions > 0:
            result.cumulative_hallucination_rate = (
                result.total_hallucinations / result.total_decisions
            )
        
        # Decision chain accuracy
        if self.decision_tracker:
            result.overall_reference_accuracy = self.decision_tracker.reference_accuracy_rate
        
        # Token pressure final state
        if self.token_engine:
            # Get overall stats from the token engine
            stats = self.token_engine.get_stats()
            # Info retention = 100% - info lost
            # Estimate from total events lost vs total events processed
            total_events = len(self.token_engine.events) + stats.get("total_loss", 0)
            if total_events > 0:
                lost_pct = stats.get("total_loss", 0) / total_events * 100
                result.final_info_retention = 100.0 - lost_pct
            else:
                result.final_info_retention = 100.0
        
        # Restart recovery comparison
        if self.restart_simulator:
            summary = self.restart_simulator.get_summary()
            result.recovery_comparison = summary.get("avg_accuracy_by_mode", {})
        
        # Hallucination distribution
        if self.hallucination_classifier:
            type_dist = self.hallucination_classifier.get_type_distribution()
            result.hallucination_distribution = {
                t.value: c for t, c in type_dist.items()
            }
    
    def get_component_status(self) -> Dict[str, bool]:
        """Get status of all components."""
        return {
            "token_pressure": self.token_engine is not None,
            "hallucination_detection": self.hallucination_classifier is not None,
            "decision_tracking": self.decision_tracker is not None,
            "restart_simulation": self.restart_simulator is not None,
            "ground_truth": self.ground_truth is not None,
            "volume_generator": self.volume_generator is not None,
        }
    
    def reset(self) -> None:
        """Reset orchestrator state for a new simulation."""
        # Reset token engine by clearing its state
        if self.token_engine:
            self.token_engine.events.clear()
            self.token_engine.current_tokens = 0
            self.token_engine.overflow_count = 0
            self.token_engine.total_events_lost = 0
            self.token_engine.compression_history.clear()
        
        if self.hallucination_classifier:
            self.hallucination_classifier.reset()
        
        if self.decision_tracker:
            self.decision_tracker.clear()
        
        if self.restart_simulator:
            self.restart_simulator.restart_events.clear()
        
        # Re-initialize components (creates fresh agents, etc.)
        self._init_components()
        self._current_day = 0
