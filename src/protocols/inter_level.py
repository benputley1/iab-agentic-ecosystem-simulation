"""
Inter-Level Communication Protocol.

Handles communication between agent hierarchy levels (L1 <-> L2 <-> L3).
Includes context serialization for passing state between levels
and simulating context rot.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    """Status of a delegated task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ResultStatus(Enum):
    """Status of a task result."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    NEEDS_REVIEW = "needs_review"


@dataclass
class AgentContext:
    """
    Context passed between agent levels.
    
    Attributes:
        context_id: Unique identifier
        agent_id: Agent this context belongs to
        level: Agent level (1, 2, or 3)
        conversation_history: Recent message history
        working_memory: Current task state
        constraints: Active constraints
        metadata: Additional metadata
        token_count: Estimated token count
        created_at: Creation timestamp
        expires_at: Expiration timestamp
    """
    context_id: str
    agent_id: str
    level: int
    conversation_history: list[dict] = field(default_factory=list)
    working_memory: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    @classmethod
    def create(
        cls,
        agent_id: str,
        level: int,
        **kwargs,
    ) -> AgentContext:
        """Create new context with auto-generated ID."""
        return cls(
            context_id=str(uuid.uuid4()),
            agent_id=agent_id,
            level=level,
            **kwargs,
        )
    
    def is_expired(self) -> bool:
        """Check if context has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "context_id": self.context_id,
            "agent_id": self.agent_id,
            "level": self.level,
            "conversation_history": self.conversation_history,
            "working_memory": self.working_memory,
            "constraints": self.constraints,
            "metadata": self.metadata,
            "token_count": self.token_count,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> AgentContext:
        """Deserialize from dictionary."""
        return cls(
            context_id=data["context_id"],
            agent_id=data["agent_id"],
            level=data["level"],
            conversation_history=data.get("conversation_history", []),
            working_memory=data.get("working_memory", {}),
            constraints=data.get("constraints", {}),
            metadata=data.get("metadata", {}),
            token_count=data.get("token_count", 0),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
        )


@dataclass
class Task:
    """
    Task delegated between agent levels.
    
    Attributes:
        task_id: Unique identifier
        name: Task name
        description: What needs to be done
        task_type: Type/category of task
        parameters: Task parameters
        context: Associated context
        priority: Task priority
        deadline: Optional deadline
        parent_task_id: ID of parent task (if subtask)
        created_by: Agent that created this task
        assigned_to: Agent assigned to execute
    """
    task_id: str
    name: str
    description: str
    task_type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    context: Optional[AgentContext] = None
    priority: TaskPriority = TaskPriority.NORMAL
    deadline: Optional[datetime] = None
    parent_task_id: Optional[str] = None
    created_by: Optional[str] = None
    assigned_to: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        task_type: str,
        created_by: str,
        **kwargs,
    ) -> Task:
        """Create new task with auto-generated ID."""
        return cls(
            task_id=str(uuid.uuid4()),
            name=name,
            description=description,
            task_type=task_type,
            created_by=created_by,
            **kwargs,
        )
    
    def is_overdue(self) -> bool:
        """Check if task is past deadline."""
        if self.deadline is None:
            return False
        return datetime.utcnow() > self.deadline
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "task_type": self.task_type,
            "parameters": self.parameters,
            "context": self.context.to_dict() if self.context else None,
            "priority": self.priority.value,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "parent_task_id": self.parent_task_id,
            "created_by": self.created_by,
            "assigned_to": self.assigned_to,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> Task:
        """Deserialize from dictionary."""
        context = None
        if data.get("context"):
            context = AgentContext.from_dict(data["context"])
        
        return cls(
            task_id=data["task_id"],
            name=data["name"],
            description=data["description"],
            task_type=data["task_type"],
            parameters=data.get("parameters", {}),
            context=context,
            priority=TaskPriority(data.get("priority", 2)),
            deadline=datetime.fromisoformat(data["deadline"]) if data.get("deadline") else None,
            parent_task_id=data.get("parent_task_id"),
            created_by=data.get("created_by"),
            assigned_to=data.get("assigned_to"),
            status=TaskStatus(data.get("status", "pending")),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
        )


@dataclass
class Result:
    """
    Result of task execution.
    
    Attributes:
        result_id: Unique identifier
        task_id: ID of completed task
        status: Result status
        output: Task output data
        metrics: Performance metrics
        errors: Any errors encountered
        context_updates: Updates to propagate to context
        execution_time_ms: How long execution took
    """
    result_id: str
    task_id: str
    status: ResultStatus
    output: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    context_updates: dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @classmethod
    def success(
        cls,
        task_id: str,
        output: dict,
        metrics: Optional[dict] = None,
        execution_time_ms: float = 0.0,
    ) -> Result:
        """Create successful result."""
        return cls(
            result_id=str(uuid.uuid4()),
            task_id=task_id,
            status=ResultStatus.SUCCESS,
            output=output,
            metrics=metrics or {},
            execution_time_ms=execution_time_ms,
        )
    
    @classmethod
    def failure(
        cls,
        task_id: str,
        errors: list[str],
        partial_output: Optional[dict] = None,
        execution_time_ms: float = 0.0,
    ) -> Result:
        """Create failure result."""
        return cls(
            result_id=str(uuid.uuid4()),
            task_id=task_id,
            status=ResultStatus.FAILURE,
            output=partial_output or {},
            errors=errors,
            execution_time_ms=execution_time_ms,
        )
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "result_id": self.result_id,
            "task_id": self.task_id,
            "status": self.status.value,
            "output": self.output,
            "metrics": self.metrics,
            "errors": self.errors,
            "context_updates": self.context_updates,
            "execution_time_ms": self.execution_time_ms,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> Result:
        """Deserialize from dictionary."""
        return cls(
            result_id=data["result_id"],
            task_id=data["task_id"],
            status=ResultStatus(data["status"]),
            output=data.get("output", {}),
            metrics=data.get("metrics", {}),
            errors=data.get("errors", []),
            context_updates=data.get("context_updates", {}),
            execution_time_ms=data.get("execution_time_ms", 0.0),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
        )


@dataclass
class DelegationResult:
    """
    Result of delegating a task downward.
    
    Attributes:
        success: Whether delegation was accepted
        task_id: ID of delegated task
        assigned_agent: Agent that accepted the task
        estimated_completion: Estimated completion time
        rejection_reason: Reason if rejected
    """
    success: bool
    task_id: str
    assigned_agent: Optional[str] = None
    estimated_completion: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "task_id": self.task_id,
            "assigned_agent": self.assigned_agent,
            "estimated_completion": self.estimated_completion.isoformat() if self.estimated_completion else None,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class AckResult:
    """
    Acknowledgment of result reported upward.
    
    Attributes:
        acknowledged: Whether result was received
        result_id: ID of acknowledged result
        feedback: Any feedback from higher level
        next_task: Optional follow-up task
    """
    acknowledged: bool
    result_id: str
    feedback: Optional[str] = None
    next_task: Optional[Task] = None
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "acknowledged": self.acknowledged,
            "result_id": self.result_id,
            "feedback": self.feedback,
            "next_task": self.next_task.to_dict() if self.next_task else None,
        }


class ContextSerializer:
    """
    Serialize/deserialize agent context.
    
    Handles token counting and context truncation
    to simulate context rot in long-running agents.
    """
    
    # Approximate tokens per character (GPT-style tokenization)
    CHARS_PER_TOKEN = 4
    
    def __init__(self, default_limit: int = 8000):
        """
        Initialize serializer.
        
        Args:
            default_limit: Default token limit for truncation
        """
        self.default_limit = default_limit
    
    def to_tokens(self, context: AgentContext) -> int:
        """
        Calculate token count for context.
        
        Args:
            context: Context to measure
            
        Returns:
            Estimated token count
        """
        serialized = json.dumps(context.to_dict(), default=str)
        return len(serialized) // self.CHARS_PER_TOKEN
    
    def to_string(self, context: AgentContext) -> str:
        """
        Serialize context to JSON string.
        
        Args:
            context: Context to serialize
            
        Returns:
            JSON string representation
        """
        return json.dumps(context.to_dict(), default=str, indent=2)
    
    def from_string(self, data: str) -> AgentContext:
        """
        Deserialize context from JSON string.
        
        Args:
            data: JSON string
            
        Returns:
            Deserialized AgentContext
        """
        return AgentContext.from_dict(json.loads(data))
    
    def truncate_to_limit(
        self,
        context: AgentContext,
        limit: Optional[int] = None,
    ) -> AgentContext:
        """
        Truncate context to fit token limit.
        
        This simulates context rot - older/less important
        information is dropped to fit within limits.
        
        Strategy:
        1. Keep most recent conversation history
        2. Keep critical working memory
        3. Drop older messages first
        
        Args:
            context: Context to truncate
            limit: Token limit (uses default if not specified)
            
        Returns:
            Truncated AgentContext
        """
        limit = limit or self.default_limit
        current_tokens = self.to_tokens(context)
        
        if current_tokens <= limit:
            context.token_count = current_tokens
            return context
        
        # Create a copy to modify
        truncated = AgentContext(
            context_id=context.context_id,
            agent_id=context.agent_id,
            level=context.level,
            conversation_history=list(context.conversation_history),
            working_memory=dict(context.working_memory),
            constraints=dict(context.constraints),
            metadata=dict(context.metadata),
            created_at=context.created_at,
            expires_at=context.expires_at,
        )
        
        # First, trim conversation history (oldest first)
        while (
            truncated.conversation_history
            and self.to_tokens(truncated) > limit
        ):
            truncated.conversation_history.pop(0)
        
        # If still over, trim working memory (keep only keys marked critical)
        if self.to_tokens(truncated) > limit:
            critical_keys = truncated.metadata.get("critical_memory_keys", [])
            new_memory = {}
            for key in critical_keys:
                if key in truncated.working_memory:
                    new_memory[key] = truncated.working_memory[key]
            truncated.working_memory = new_memory
        
        # Last resort: trim metadata
        if self.to_tokens(truncated) > limit:
            truncated.metadata = {
                "truncated": True,
                "original_tokens": current_tokens,
            }
        
        truncated.token_count = self.to_tokens(truncated)
        truncated.metadata["context_rot_applied"] = True
        truncated.metadata["tokens_before_truncation"] = current_tokens
        
        return truncated
    
    def compute_hash(self, context: AgentContext) -> str:
        """
        Compute hash of context for change detection.
        
        Args:
            context: Context to hash
            
        Returns:
            Hash string
        """
        serialized = json.dumps(context.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]


class InterLevelProtocol:
    """
    Communication protocol between agent hierarchy levels.
    
    Handles L1 <-> L2 <-> L3 communication with proper
    context management and delegation tracking.
    """
    
    # Valid level transitions
    VALID_DELEGATIONS = {
        (1, 2): True,  # L1 -> L2
        (2, 3): True,  # L2 -> L3
    }
    
    VALID_REPORTS = {
        (3, 2): True,  # L3 -> L2
        (2, 1): True,  # L2 -> L1
    }
    
    def __init__(
        self,
        agent_id: str,
        agent_level: int,
        context_limit: int = 8000,
    ):
        """
        Initialize inter-level protocol.
        
        Args:
            agent_id: ID of agent using this protocol
            agent_level: Level of this agent (1, 2, or 3)
            context_limit: Token limit for context passing
        """
        self.agent_id = agent_id
        self.agent_level = agent_level
        self.serializer = ContextSerializer(default_limit=context_limit)
        self._pending_delegations: dict[str, Task] = {}
        self._pending_results: dict[str, Result] = {}
        self._subordinates: dict[str, str] = {}  # agent_id -> level
    
    def register_subordinate(self, agent_id: str, level: int) -> None:
        """Register a subordinate agent."""
        if level != self.agent_level + 1:
            raise ValueError(
                f"Cannot register level {level} as subordinate to level {self.agent_level}"
            )
        self._subordinates[agent_id] = str(level)
    
    async def delegate_down(
        self,
        from_level: int,
        to_level: int,
        task: Task,
    ) -> DelegationResult:
        """
        Delegate task from higher to lower level.
        
        Args:
            from_level: Source level (must be this agent's level)
            to_level: Target level (must be from_level + 1)
            task: Task to delegate
            
        Returns:
            DelegationResult with assignment details
        """
        # Validate levels
        if from_level != self.agent_level:
            return DelegationResult(
                success=False,
                task_id=task.task_id,
                rejection_reason=f"Invalid from_level {from_level} for agent at level {self.agent_level}",
            )
        
        if (from_level, to_level) not in self.VALID_DELEGATIONS:
            return DelegationResult(
                success=False,
                task_id=task.task_id,
                rejection_reason=f"Invalid delegation path L{from_level} -> L{to_level}",
            )
        
        # Truncate context if needed
        if task.context:
            task.context = self.serializer.truncate_to_limit(task.context)
        
        # Find available subordinate (simplified - in production would be load balanced)
        assigned_agent = task.assigned_to
        if not assigned_agent and self._subordinates:
            assigned_agent = list(self._subordinates.keys())[0]
        
        if not assigned_agent:
            return DelegationResult(
                success=False,
                task_id=task.task_id,
                rejection_reason="No subordinate agents available",
            )
        
        # Track delegation
        task.assigned_to = assigned_agent
        task.status = TaskStatus.IN_PROGRESS
        self._pending_delegations[task.task_id] = task
        
        return DelegationResult(
            success=True,
            task_id=task.task_id,
            assigned_agent=assigned_agent,
            estimated_completion=task.deadline,
        )
    
    async def report_up(
        self,
        from_level: int,
        to_level: int,
        result: Result,
    ) -> AckResult:
        """
        Report results from lower to higher level.
        
        Args:
            from_level: Source level (must be this agent's level)
            to_level: Target level (must be from_level - 1)
            result: Result to report
            
        Returns:
            AckResult with acknowledgment
        """
        # Validate levels
        if from_level != self.agent_level:
            return AckResult(
                acknowledged=False,
                result_id=result.result_id,
                feedback=f"Invalid from_level {from_level} for agent at level {self.agent_level}",
            )
        
        if (from_level, to_level) not in self.VALID_REPORTS:
            return AckResult(
                acknowledged=False,
                result_id=result.result_id,
                feedback=f"Invalid report path L{from_level} -> L{to_level}",
            )
        
        # Store result for parent to retrieve
        self._pending_results[result.result_id] = result
        
        # Generate feedback based on result status
        feedback = None
        if result.status == ResultStatus.FAILURE:
            feedback = "Task failed - review errors and retry if appropriate"
        elif result.status == ResultStatus.PARTIAL:
            feedback = "Partial completion - additional work may be needed"
        elif result.status == ResultStatus.NEEDS_REVIEW:
            feedback = "Result requires human review before proceeding"
        
        return AckResult(
            acknowledged=True,
            result_id=result.result_id,
            feedback=feedback,
        )
    
    def serialize_context(self, context: AgentContext) -> str:
        """
        Serialize context for passing between levels.
        
        Automatically applies truncation if needed.
        
        Args:
            context: Context to serialize
            
        Returns:
            JSON string representation
        """
        truncated = self.serializer.truncate_to_limit(context)
        return self.serializer.to_string(truncated)
    
    def deserialize_context(self, data: str) -> AgentContext:
        """
        Deserialize received context.
        
        Args:
            data: JSON string
            
        Returns:
            Deserialized AgentContext
        """
        return self.serializer.from_string(data)
    
    def get_pending_task(self, task_id: str) -> Optional[Task]:
        """Get a pending delegated task."""
        return self._pending_delegations.get(task_id)
    
    def complete_delegation(self, task_id: str, result: Result) -> None:
        """Mark a delegation as complete."""
        if task_id in self._pending_delegations:
            self._pending_delegations[task_id].status = TaskStatus.COMPLETED
            self._pending_results[result.result_id] = result
    
    def get_pending_result(self, result_id: str) -> Optional[Result]:
        """Get a pending result."""
        return self._pending_results.get(result_id)
    
    def get_delegation_stats(self) -> dict:
        """Get statistics on delegations."""
        pending = [t for t in self._pending_delegations.values() if t.status == TaskStatus.IN_PROGRESS]
        completed = [t for t in self._pending_delegations.values() if t.status == TaskStatus.COMPLETED]
        failed = [t for t in self._pending_delegations.values() if t.status == TaskStatus.FAILED]
        
        return {
            "pending_count": len(pending),
            "completed_count": len(completed),
            "failed_count": len(failed),
            "total_delegations": len(self._pending_delegations),
            "subordinate_count": len(self._subordinates),
        }
