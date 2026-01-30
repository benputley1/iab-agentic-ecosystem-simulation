"""
Decision Chain Tracker - Track decision dependencies and detect cascading errors.

This module tracks agent decisions with their references to previous decisions,
verifying reference integrity and detecting cascading errors caused by
hallucinations or context window limitations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from collections import deque


class ReferenceErrorType(Enum):
    """Classification of reference failures."""
    
    WRONG_VALUE = "wrong_value"  # Reference exists but value doesn't match
    MISSING_REFERENCE = "missing_reference"  # Referenced decision doesn't exist
    STALE_DATA = "stale_data"  # Reference to decision outside lookback window
    CASCADING_ERROR = "cascading_error"  # Error caused by previous error


@dataclass
class DecisionReference:
    """A reference to a previous decision within an agent decision."""
    
    decision_id: str  # ID of the referenced decision
    recalled_value: Any  # The value the agent recalls/claims
    field_name: str = "value"  # Which field is being referenced
    
    def __hash__(self):
        return hash((self.decision_id, self.field_name))
    
    def __eq__(self, other):
        if not isinstance(other, DecisionReference):
            return False
        return self.decision_id == other.decision_id and self.field_name == other.field_name


@dataclass
class AgentDecision:
    """Represents a decision made by an agent with optional references to prior decisions."""
    
    id: str  # Unique identifier for this decision
    timestamp: datetime  # When the decision was made
    agent_id: str  # Which agent made the decision
    decision_type: str  # Type of decision (bid, accept, reject, etc.)
    value: Any  # The decision value (e.g., bid amount, deal terms)
    references: List[DecisionReference] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_references(self) -> List[DecisionReference]:
        """Return all references this decision makes to previous decisions."""
        return self.references
    
    def add_reference(self, ref: DecisionReference) -> None:
        """Add a reference to a previous decision."""
        self.references.append(ref)


@dataclass
class ReferenceFailure:
    """Records a failed reference check."""
    
    decision_id: str  # ID of the decision that made the bad reference
    ref_id: str  # ID of the referenced decision
    expected: Any  # What the agent claimed/recalled
    actual: Any  # What the actual value was (None if missing)
    error_type: ReferenceErrorType  # Classification of the error
    field_name: str = "value"  # Which field had the error
    timestamp: datetime = field(default_factory=datetime.utcnow)
    caused_by: Optional[str] = None  # If cascading, the original error decision_id
    
    @property
    def is_cascading(self) -> bool:
        """Check if this error was caused by a previous error."""
        return self.error_type == ReferenceErrorType.CASCADING_ERROR or self.caused_by is not None


@dataclass
class ChainResult:
    """Result of recording a decision to the tracker."""
    
    decision_id: str  # ID of the recorded decision
    success: bool  # Whether all references were valid
    failures: List[ReferenceFailure] = field(default_factory=list)
    cascading_errors: int = 0  # Number of cascading errors detected
    chain_depth: int = 0  # Depth of reference chain that was verified
    
    @property
    def total_errors(self) -> int:
        """Total number of reference failures."""
        return len(self.failures)
    
    @property
    def has_cascading_errors(self) -> bool:
        """Check if any errors were cascading."""
        return self.cascading_errors > 0


class DecisionChainTracker:
    """
    Track decision dependencies and cascading errors.
    
    Maintains a sliding window of recent decisions and verifies that
    references to previous decisions are accurate. Detects and classifies
    errors including cascading errors caused by previous failures.
    """
    
    def __init__(self, lookback_window: int = 100):
        """
        Initialize the tracker.
        
        Args:
            lookback_window: Maximum number of decisions to keep in memory.
                            References to decisions outside this window are
                            considered stale.
        """
        if lookback_window < 1:
            raise ValueError("lookback_window must be at least 1")
        
        self.lookback = lookback_window
        self.decisions: deque[AgentDecision] = deque(maxlen=lookback_window)
        self._decision_index: Dict[str, AgentDecision] = {}  # Quick lookup
        self._stale_ids: set = set()  # IDs that have been evicted from window
        self.reference_failures: List[ReferenceFailure] = []
        self._error_decisions: set = set()  # Decisions with errors
    
    def record_decision(self, decision: AgentDecision) -> ChainResult:
        """
        Record a decision and check reference integrity.
        
        Args:
            decision: The decision to record
            
        Returns:
            ChainResult with verification status and any failures
        """
        failures = []
        cascading_count = 0
        max_chain_depth = 0
        
        # Check all references
        references = decision.get_references()
        for ref in references:
            failure = self._verify_reference(decision.id, ref)
            if failure:
                failures.append(failure)
                self.reference_failures.append(failure)
                
                if failure.is_cascading:
                    cascading_count += 1
            
            # Calculate chain depth
            chain_depth = self._get_reference_chain_depth(ref.decision_id)
            max_chain_depth = max(max_chain_depth, chain_depth + 1)
        
        # Track if this decision has errors
        if failures:
            self._error_decisions.add(decision.id)
        
        # Handle eviction before adding new decision
        if len(self.decisions) >= self.lookback:
            evicted = self.decisions[0]
            self._stale_ids.add(evicted.id)
            if evicted.id in self._decision_index:
                del self._decision_index[evicted.id]
        
        # Add decision to tracker
        self.decisions.append(decision)
        self._decision_index[decision.id] = decision
        
        return ChainResult(
            decision_id=decision.id,
            success=len(failures) == 0,
            failures=failures,
            cascading_errors=cascading_count,
            chain_depth=max_chain_depth
        )
    
    def _verify_reference(
        self, 
        decision_id: str, 
        ref: DecisionReference
    ) -> Optional[ReferenceFailure]:
        """
        Verify a single reference against actual data.
        
        Returns a ReferenceFailure if verification fails, None otherwise.
        """
        # Check if reference is to a stale (evicted) decision
        if ref.decision_id in self._stale_ids:
            return ReferenceFailure(
                decision_id=decision_id,
                ref_id=ref.decision_id,
                expected=ref.recalled_value,
                actual=None,
                error_type=ReferenceErrorType.STALE_DATA,
                field_name=ref.field_name
            )
        
        # Check if referenced decision exists
        actual_decision = self._decision_index.get(ref.decision_id)
        if actual_decision is None:
            return ReferenceFailure(
                decision_id=decision_id,
                ref_id=ref.decision_id,
                expected=ref.recalled_value,
                actual=None,
                error_type=ReferenceErrorType.MISSING_REFERENCE,
                field_name=ref.field_name
            )
        
        # Get actual value from the referenced decision
        actual_value = self._get_field_value(actual_decision, ref.field_name)
        
        # Compare values
        if not self._values_match(ref.recalled_value, actual_value):
            # Determine if this is a cascading error
            error_type = ReferenceErrorType.WRONG_VALUE
            caused_by = None
            
            if ref.decision_id in self._error_decisions:
                error_type = ReferenceErrorType.CASCADING_ERROR
                caused_by = ref.decision_id
            
            return ReferenceFailure(
                decision_id=decision_id,
                ref_id=ref.decision_id,
                expected=ref.recalled_value,
                actual=actual_value,
                error_type=error_type,
                field_name=ref.field_name,
                caused_by=caused_by
            )
        
        return None
    
    def _get_field_value(self, decision: AgentDecision, field_name: str) -> Any:
        """Get a field value from a decision by name."""
        if field_name == "value":
            return decision.value
        elif field_name in ("id", "decision_id"):
            return decision.id
        elif field_name == "agent_id":
            return decision.agent_id
        elif field_name == "decision_type":
            return decision.decision_type
        elif field_name in decision.metadata:
            return decision.metadata[field_name]
        return None
    
    def _values_match(self, expected: Any, actual: Any) -> bool:
        """
        Compare two values for equality.
        
        Handles special cases like floating point comparison.
        """
        if expected is None and actual is None:
            return True
        if expected is None or actual is None:
            return False
        
        # Handle floating point comparison
        if isinstance(expected, float) and isinstance(actual, float):
            return abs(expected - actual) < 1e-9
        
        return expected == actual
    
    def _get_reference_chain_depth(self, decision_id: str) -> int:
        """
        Calculate the depth of the reference chain for a decision.
        
        Returns 0 if decision has no references or doesn't exist.
        """
        decision = self._decision_index.get(decision_id)
        if decision is None or not decision.references:
            return 0
        
        max_depth = 0
        visited = {decision_id}  # Prevent cycles
        
        for ref in decision.references:
            if ref.decision_id not in visited:
                visited.add(ref.decision_id)
                depth = 1 + self._get_reference_chain_depth(ref.decision_id)
                max_depth = max(max_depth, depth)
        
        return max_depth
    
    def get_decision(self, decision_id: str) -> Optional[AgentDecision]:
        """Get a decision by ID if it's still in the lookback window."""
        return self._decision_index.get(decision_id)
    
    def get_actual_decision(self, decision_id: str) -> Optional[AgentDecision]:
        """Alias for get_decision for API compatibility with spec."""
        return self.get_decision(decision_id)
    
    def classify_error(
        self, 
        ref: DecisionReference, 
        actual: Optional[AgentDecision]
    ) -> ReferenceErrorType:
        """
        Classify the type of reference error.
        
        This is a utility method for external use - internal verification
        handles classification automatically.
        """
        if ref.decision_id in self._stale_ids:
            return ReferenceErrorType.STALE_DATA
        
        if actual is None:
            return ReferenceErrorType.MISSING_REFERENCE
        
        if ref.decision_id in self._error_decisions:
            return ReferenceErrorType.CASCADING_ERROR
        
        return ReferenceErrorType.WRONG_VALUE
    
    def count_cascading_errors(self, decision: AgentDecision) -> int:
        """
        Count how many cascading errors would be caused by this decision.
        
        Returns the number of references to decisions that had errors.
        """
        count = 0
        for ref in decision.get_references():
            if ref.decision_id in self._error_decisions:
                count += 1
        return count
    
    @property
    def total_decisions(self) -> int:
        """Total number of decisions in the current window."""
        return len(self.decisions)
    
    @property
    def total_failures(self) -> int:
        """Total number of reference failures recorded."""
        return len(self.reference_failures)
    
    @property
    def reference_accuracy_rate(self) -> float:
        """
        Calculate the rate of accurate references.
        
        Returns 1.0 if no references have been checked.
        """
        total_refs = sum(len(d.references) for d in self.decisions)
        if total_refs == 0:
            return 1.0
        return 1.0 - (self.total_failures / total_refs)
    
    def get_error_distribution(self) -> Dict[ReferenceErrorType, int]:
        """Get distribution of error types."""
        distribution = {error_type: 0 for error_type in ReferenceErrorType}
        for failure in self.reference_failures:
            distribution[failure.error_type] += 1
        return distribution
    
    def get_cascading_chains(self) -> List[List[str]]:
        """
        Get all cascading error chains.
        
        Returns list of chains, where each chain is a list of decision IDs
        starting from the original error.
        """
        chains = []
        processed = set()
        
        for failure in self.reference_failures:
            if failure.is_cascading and failure.caused_by not in processed:
                chain = self._build_cascading_chain(failure.caused_by)
                if chain:
                    chains.append(chain)
                    processed.update(chain)
        
        return chains
    
    def _build_cascading_chain(self, start_id: str) -> List[str]:
        """Build a chain of cascading errors starting from a decision."""
        chain = [start_id]
        
        # Find all failures caused by this decision
        for failure in self.reference_failures:
            if failure.caused_by == start_id:
                chain.append(failure.decision_id)
                # Recursively find failures caused by this new error
                sub_chain = self._build_cascading_chain(failure.decision_id)
                chain.extend(sub_chain[1:])  # Skip first element to avoid duplicate
        
        return chain
    
    def clear(self) -> None:
        """Clear all tracked decisions and failures."""
        self.decisions.clear()
        self._decision_index.clear()
        self._stale_ids.clear()
        self.reference_failures.clear()
        self._error_decisions.clear()
