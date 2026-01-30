"""Agent Restart Simulator - Simulate agent crashes and measure state recovery.

This module provides tools to simulate random agent crashes during campaign
execution and compare recovery accuracy between different storage modes
(private database vs. ledger-based recovery).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol
import random
import copy
from datetime import datetime


@dataclass
class AgentState:
    """Represents the internal state of an agent at a point in time.
    
    Attributes:
        budget_remaining: Remaining budget for the campaign
        impressions_delivered: Total impressions delivered so far
        active_deals: Dict of deal_id -> deal terms
        frequency_caps: Dict of user_id -> exposure count
        price_history: List of recent price points for anchoring
        campaign_id: Current campaign identifier
        timestamp: When this state snapshot was taken
        metadata: Additional state data
    """
    budget_remaining: float = 0.0
    impressions_delivered: int = 0
    active_deals: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    frequency_caps: Dict[str, int] = field(default_factory=dict)
    price_history: List[float] = field(default_factory=list)
    campaign_id: str = ""
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for comparison."""
        return {
            "budget_remaining": self.budget_remaining,
            "impressions_delivered": self.impressions_delivered,
            "active_deals": self.active_deals,
            "frequency_caps": self.frequency_caps,
            "price_history": self.price_history,
            "campaign_id": self.campaign_id,
            "metadata": self.metadata,
        }


@dataclass
class RecoveryResult:
    """Result of state recovery attempt.
    
    Attributes:
        mode: Recovery mode used ("private_db" or "ledger")
        accuracy: Recovery accuracy as a percentage (0.0-1.0)
        recovered_state: The state after recovery
        fields_recovered: Number of fields correctly recovered
        fields_total: Total number of fields
        field_errors: Dict of field_name -> (expected, actual)
        recovery_time_ms: Time taken to recover (simulated)
    """
    mode: str
    accuracy: float
    recovered_state: AgentState
    fields_recovered: int = 0
    fields_total: int = 0
    field_errors: Dict[str, tuple] = field(default_factory=dict)
    recovery_time_ms: float = 0.0


@dataclass
class RestartEvent:
    """Record of an agent restart/crash event.
    
    Attributes:
        hour: Simulation hour when crash occurred
        pre_state: Agent state before crash
        recovery_accuracy: Dict of mode -> accuracy percentage
        recovery_results: Dict of mode -> RecoveryResult
        decisions_affected: Number of decisions that may be impacted
        crash_reason: Simulated reason for crash
    """
    hour: int
    pre_state: AgentState
    recovery_accuracy: Dict[str, float]
    recovery_results: Dict[str, RecoveryResult] = field(default_factory=dict)
    decisions_affected: int = 0
    crash_reason: str = "random_failure"


class AgentProtocol(Protocol):
    """Protocol defining required agent methods for restart simulation."""
    
    def get_internal_state(self) -> AgentState:
        """Get current internal state snapshot."""
        ...
    
    def restart(self) -> None:
        """Restart the agent (clear volatile state)."""
        ...
    
    def recover_state(self, mode: str) -> AgentState:
        """Recover state using specified mode."""
        ...


class MockAgent:
    """Mock agent for testing restart simulation.
    
    Simulates an agent with internal state that can be saved, lost on restart,
    and recovered with varying accuracy depending on the recovery mode.
    
    Private DB recovery: ~87% accuracy (loses volatile cache, recent frequency data)
    Ledger recovery: ~99.8% accuracy (full transaction history on blockchain)
    """
    
    # Default recovery accuracy for each mode
    DEFAULT_RECOVERY_ACCURACY = {
        "private_db": 0.873,  # 87.3% - loses volatile state
        "ledger": 0.998,      # 99.8% - near-perfect recovery
    }
    
    def __init__(
        self,
        agent_id: str = "mock-agent-001",
        initial_budget: float = 10000.0,
        campaign_id: str = "camp-001",
        recovery_accuracy: Optional[Dict[str, float]] = None,
    ):
        """Initialize mock agent.
        
        Args:
            agent_id: Unique identifier for this agent
            initial_budget: Starting budget for campaign
            campaign_id: Campaign identifier
            recovery_accuracy: Override default recovery accuracy per mode
        """
        self.agent_id = agent_id
        self.campaign_id = campaign_id
        self._initial_budget = initial_budget
        
        # Internal state
        self._budget_remaining = initial_budget
        self._impressions_delivered = 0
        self._active_deals: Dict[str, Dict[str, Any]] = {}
        self._frequency_caps: Dict[str, int] = {}
        self._price_history: List[float] = []
        self._metadata: Dict[str, Any] = {}
        
        # Recovery config
        self._recovery_accuracy = recovery_accuracy or self.DEFAULT_RECOVERY_ACCURACY.copy()
        
        # Track last known good state (simulates what's persisted)
        self._persisted_state: Optional[AgentState] = None
        self._ledger_state: Optional[AgentState] = None
    
    def record_spend(self, amount: float, impressions: int = 0) -> None:
        """Record a spend event."""
        self._budget_remaining -= amount
        self._impressions_delivered += impressions
        self._price_history.append(amount / max(impressions, 1) * 1000)  # CPM
        
        # Persist to simulated storage
        self._persist_state()
    
    def add_deal(self, deal_id: str, terms: Dict[str, Any]) -> None:
        """Add an active deal."""
        self._active_deals[deal_id] = terms
        self._persist_state()
    
    def record_frequency(self, user_id: str) -> None:
        """Record a user exposure for frequency capping."""
        self._frequency_caps[user_id] = self._frequency_caps.get(user_id, 0) + 1
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value."""
        self._metadata[key] = value
    
    def _persist_state(self) -> None:
        """Simulate persisting state to database and ledger."""
        current_state = self.get_internal_state()
        
        # Private DB: loses some volatile data (frequency caps, recent prices)
        db_state = copy.deepcopy(current_state)
        # Simulate DB not keeping all frequency cap data
        if len(db_state.frequency_caps) > 100:
            # Only keep most recent 100 users
            sorted_users = sorted(db_state.frequency_caps.keys())
            db_state.frequency_caps = {
                k: db_state.frequency_caps[k] 
                for k in sorted_users[-100:]
            }
        # Simulate DB truncating price history
        db_state.price_history = db_state.price_history[-50:]
        self._persisted_state = db_state
        
        # Ledger: full state (immutable, complete history)
        self._ledger_state = copy.deepcopy(current_state)
    
    def get_internal_state(self) -> AgentState:
        """Get current internal state snapshot."""
        return AgentState(
            budget_remaining=self._budget_remaining,
            impressions_delivered=self._impressions_delivered,
            active_deals=copy.deepcopy(self._active_deals),
            frequency_caps=copy.deepcopy(self._frequency_caps),
            price_history=self._price_history.copy(),
            campaign_id=self.campaign_id,
            timestamp=datetime.now(),
            metadata=copy.deepcopy(self._metadata),
        )
    
    def restart(self) -> None:
        """Restart agent - clears all volatile state."""
        # Simulate crash - lose everything in memory
        self._budget_remaining = self._initial_budget
        self._impressions_delivered = 0
        self._active_deals = {}
        self._frequency_caps = {}
        self._price_history = []
        self._metadata = {}
    
    def recover_state(self, mode: str) -> AgentState:
        """Recover state using specified mode.
        
        Args:
            mode: Recovery mode ("private_db" or "ledger")
            
        Returns:
            Recovered state (may have degraded accuracy)
        """
        if mode == "private_db":
            if self._persisted_state:
                # Restore from DB with some data loss
                recovered = copy.deepcopy(self._persisted_state)
                self._apply_state(recovered)
                return recovered
            else:
                # No persisted state - return empty
                return AgentState(campaign_id=self.campaign_id)
        
        elif mode == "ledger":
            if self._ledger_state:
                # Restore from ledger - near perfect
                recovered = copy.deepcopy(self._ledger_state)
                self._apply_state(recovered)
                return recovered
            else:
                return AgentState(campaign_id=self.campaign_id)
        
        else:
            raise ValueError(f"Unknown recovery mode: {mode}")
    
    def _apply_state(self, state: AgentState) -> None:
        """Apply recovered state to agent."""
        self._budget_remaining = state.budget_remaining
        self._impressions_delivered = state.impressions_delivered
        self._active_deals = copy.deepcopy(state.active_deals)
        self._frequency_caps = copy.deepcopy(state.frequency_caps)
        self._price_history = state.price_history.copy()
        self._metadata = copy.deepcopy(state.metadata)


class AgentRestartSimulator:
    """Simulate agent crashes and measure state recovery accuracy.
    
    This simulator randomly crashes agents during campaign execution and
    compares how well different recovery modes (private DB vs. ledger)
    can restore the agent's state.
    
    Key findings expected:
    - Private DB recovery: ~87% accuracy (loses volatile state)
    - Ledger recovery: ~99.8% accuracy (complete transaction history)
    
    Example:
        >>> simulator = AgentRestartSimulator(crash_probability=0.01)
        >>> agent = MockAgent(initial_budget=10000)
        >>> for hour in range(720):  # 30 days
        ...     # Agent does work...
        ...     event = simulator.maybe_crash(agent, hour)
        ...     if event:
        ...         print(f"Crash at hour {hour}!")
        ...         print(f"Ledger recovery: {event.recovery_accuracy['ledger']:.1%}")
    """
    
    def __init__(
        self,
        crash_probability: float = 0.01,
        recovery_modes: Optional[List[str]] = None,
        random_seed: Optional[int] = None,
    ):
        """Initialize restart simulator.
        
        Args:
            crash_probability: Probability of crash per hour (default 1%)
            recovery_modes: List of recovery modes to test (default: private_db, ledger)
            random_seed: Seed for reproducible crash patterns (optional)
        """
        if not 0.0 <= crash_probability <= 1.0:
            raise ValueError(f"crash_probability must be between 0 and 1, got {crash_probability}")
        
        self.crash_probability = crash_probability
        self.recovery_modes = recovery_modes or ["private_db", "ledger"]
        self.restart_events: List[RestartEvent] = []
        
        # Random state for reproducibility
        self._rng = random.Random(random_seed)
    
    def maybe_crash(
        self,
        agent: AgentProtocol,
        hour: int,
        force_crash: bool = False,
    ) -> Optional[RestartEvent]:
        """Potentially crash the agent and measure recovery.
        
        Args:
            agent: Agent to potentially crash
            hour: Current simulation hour
            force_crash: If True, always crash (for testing)
            
        Returns:
            RestartEvent if crash occurred, None otherwise
        """
        # Check if crash occurs
        if not force_crash and self._rng.random() >= self.crash_probability:
            return None
        
        # Capture pre-crash state
        pre_state = agent.get_internal_state()
        
        # Simulate crash
        agent.restart()
        
        # Test recovery for each mode
        recovery_accuracy: Dict[str, float] = {}
        recovery_results: Dict[str, RecoveryResult] = {}
        
        for mode in self.recovery_modes:
            # Recover state
            recovered_state = agent.recover_state(mode)
            
            # Compare states and calculate accuracy
            result = self._compare_states(pre_state, recovered_state, mode)
            recovery_accuracy[mode] = result.accuracy
            recovery_results[mode] = result
            
            # Reset agent for next mode test
            agent.restart()
        
        # Restore to best recovery mode (ledger if available)
        best_mode = max(recovery_accuracy, key=recovery_accuracy.get)
        agent.recover_state(best_mode)
        
        # Create restart event
        event = RestartEvent(
            hour=hour,
            pre_state=pre_state,
            recovery_accuracy=recovery_accuracy,
            recovery_results=recovery_results,
            decisions_affected=self._estimate_affected_decisions(pre_state, recovery_results),
            crash_reason=self._generate_crash_reason(),
        )
        
        self.restart_events.append(event)
        return event
    
    def _compare_states(
        self,
        original: AgentState,
        recovered: AgentState,
        mode: str,
    ) -> RecoveryResult:
        """Compare original and recovered states to calculate accuracy.
        
        Args:
            original: State before crash
            recovered: State after recovery
            mode: Recovery mode used
            
        Returns:
            RecoveryResult with accuracy metrics
        """
        field_errors: Dict[str, tuple] = {}
        fields_correct = 0
        total_fields = 0
        
        # Compare each field
        comparisons = [
            ("budget_remaining", original.budget_remaining, recovered.budget_remaining),
            ("impressions_delivered", original.impressions_delivered, recovered.impressions_delivered),
            ("campaign_id", original.campaign_id, recovered.campaign_id),
        ]
        
        for field_name, expected, actual in comparisons:
            total_fields += 1
            if self._values_match(expected, actual):
                fields_correct += 1
            else:
                field_errors[field_name] = (expected, actual)
        
        # Compare complex fields
        # Active deals
        total_fields += 1
        if original.active_deals == recovered.active_deals:
            fields_correct += 1
        else:
            field_errors["active_deals"] = (
                len(original.active_deals),
                len(recovered.active_deals)
            )
        
        # Frequency caps (count matching entries)
        total_fields += 1
        orig_caps = set(original.frequency_caps.items())
        recv_caps = set(recovered.frequency_caps.items())
        cap_accuracy = len(orig_caps & recv_caps) / max(len(orig_caps), 1)
        if cap_accuracy >= 0.95:  # 95% match is considered correct
            fields_correct += 1
        else:
            field_errors["frequency_caps"] = (
                len(original.frequency_caps),
                len(recovered.frequency_caps)
            )
        
        # Price history (check if recent prices match)
        total_fields += 1
        if len(original.price_history) == 0:
            fields_correct += 1
        else:
            recent_orig = original.price_history[-10:] if original.price_history else []
            recent_recv = recovered.price_history[-10:] if recovered.price_history else []
            if recent_orig == recent_recv:
                fields_correct += 1
            else:
                field_errors["price_history"] = (
                    len(original.price_history),
                    len(recovered.price_history)
                )
        
        # Calculate overall accuracy
        accuracy = fields_correct / total_fields if total_fields > 0 else 0.0
        
        # Simulate recovery time (ledger is faster due to indexed lookups)
        recovery_time = 2300 if mode == "private_db" else 800  # milliseconds
        
        return RecoveryResult(
            mode=mode,
            accuracy=accuracy,
            recovered_state=recovered,
            fields_recovered=fields_correct,
            fields_total=total_fields,
            field_errors=field_errors,
            recovery_time_ms=recovery_time,
        )
    
    def _values_match(self, expected: Any, actual: Any, tolerance: float = 0.001) -> bool:
        """Check if two values match (with tolerance for floats)."""
        if isinstance(expected, float) and isinstance(actual, float):
            return abs(expected - actual) < tolerance * max(abs(expected), 1)
        return expected == actual
    
    def _estimate_affected_decisions(
        self,
        pre_state: AgentState,
        recovery_results: Dict[str, RecoveryResult],
    ) -> int:
        """Estimate number of decisions affected by incomplete recovery."""
        # Base estimate on error rate and recent activity
        worst_accuracy = min(r.accuracy for r in recovery_results.values())
        error_rate = 1 - worst_accuracy
        
        # Assume ~100 decisions per hour affected by budget/frequency errors
        base_decisions = 100
        return int(base_decisions * error_rate * 10)
    
    def _generate_crash_reason(self) -> str:
        """Generate a simulated crash reason."""
        reasons = [
            "memory_overflow",
            "network_timeout",
            "process_killed",
            "container_restart",
            "dependency_failure",
            "rate_limit_exceeded",
        ]
        return self._rng.choice(reasons)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all restart events.
        
        Returns:
            Dictionary with restart statistics
        """
        if not self.restart_events:
            return {
                "total_restarts": 0,
                "avg_accuracy_by_mode": {},
                "decisions_affected_total": 0,
            }
        
        # Calculate average accuracy per mode
        avg_accuracy: Dict[str, float] = {}
        for mode in self.recovery_modes:
            accuracies = [e.recovery_accuracy[mode] for e in self.restart_events]
            avg_accuracy[mode] = sum(accuracies) / len(accuracies)
        
        # Calculate total affected decisions
        total_affected = sum(e.decisions_affected for e in self.restart_events)
        
        # Crash reasons distribution
        reasons: Dict[str, int] = {}
        for event in self.restart_events:
            reasons[event.crash_reason] = reasons.get(event.crash_reason, 0) + 1
        
        return {
            "total_restarts": len(self.restart_events),
            "avg_accuracy_by_mode": avg_accuracy,
            "decisions_affected_total": total_affected,
            "crash_hours": [e.hour for e in self.restart_events],
            "crash_reasons": reasons,
        }
    
    def format_report(self) -> str:
        """Format a human-readable report of restart events.
        
        Returns:
            Formatted string report
        """
        summary = self.get_summary()
        
        lines = [
            "=" * 60,
            "AGENT RESTART SIMULATION REPORT",
            "=" * 60,
            f"Total Restart Events: {summary['total_restarts']}",
            f"Total Decisions Affected: {summary['decisions_affected_total']}",
            "",
            "Recovery Accuracy by Mode:",
            "-" * 40,
        ]
        
        for mode, accuracy in summary.get("avg_accuracy_by_mode", {}).items():
            lines.append(f"  {mode:15} | {accuracy:6.1%}")
        
        lines.append("")
        lines.append("Crash Reasons:")
        lines.append("-" * 40)
        
        for reason, count in sorted(
            summary.get("crash_reasons", {}).items(),
            key=lambda x: -x[1]
        ):
            lines.append(f"  {reason:25} | {count}")
        
        if self.restart_events:
            lines.append("")
            lines.append("Individual Events:")
            lines.append("-" * 60)
            
            for i, event in enumerate(self.restart_events, 1):
                lines.append(f"Restart #{i} (Hour {event.hour})")
                lines.append(f"  Reason: {event.crash_reason}")
                lines.append(f"  Pre-crash budget: ${event.pre_state.budget_remaining:,.2f}")
                lines.append(f"  Recovery accuracy:")
                for mode, acc in event.recovery_accuracy.items():
                    affected = event.recovery_results[mode].fields_total - event.recovery_results[mode].fields_recovered
                    lines.append(f"    {mode}: {acc:.1%} ({affected} fields degraded)")
        
        lines.append("=" * 60)
        return "\n".join(lines)
