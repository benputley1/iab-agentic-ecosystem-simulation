"""Base classes for MCP tools."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolStatus(Enum):
    """Status of a tool execution."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"


@dataclass
class ToolResult:
    """Result of a tool execution."""
    
    status: ToolStatus
    data: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        """Check if execution was successful."""
        return self.status == ToolStatus.SUCCESS
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
        }
    
    @classmethod
    def ok(cls, data: dict[str, Any], **metadata) -> "ToolResult":
        """Create a successful result."""
        return cls(status=ToolStatus.SUCCESS, data=data, metadata=metadata)
    
    @classmethod
    def fail(cls, error: str, **metadata) -> "ToolResult":
        """Create a failed result."""
        return cls(status=ToolStatus.ERROR, error=error, metadata=metadata)


@dataclass
class ToolSpec:
    """Specification for a tool (for LLM context)."""
    
    name: str
    description: str
    input_schema: dict[str, Any]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to MCP tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class Tool(ABC):
    """Base class for MCP tools.
    
    All tools must inherit from this class and implement the execute method.
    Tools are the primary interface between agents and the OpenDirect protocol.
    """
    
    name: str
    description: str
    parameters: dict[str, Any]
    
    def __init__(self):
        """Initialize tool with spec validation."""
        if not hasattr(self, 'name') or not self.name:
            raise ValueError(f"{self.__class__.__name__} must define 'name'")
        if not hasattr(self, 'description') or not self.description:
            raise ValueError(f"{self.__class__.__name__} must define 'description'")
        if not hasattr(self, 'parameters'):
            self.parameters = {}
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given arguments.
        
        Args:
            **kwargs: Tool-specific arguments
            
        Returns:
            ToolResult with execution outcome
        """
        pass
    
    def get_spec(self) -> ToolSpec:
        """Get the tool specification for LLM context."""
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema={
                "type": "object",
                "properties": self.parameters,
                "required": self._get_required_params(),
            }
        )
    
    def _get_required_params(self) -> list[str]:
        """Get list of required parameters."""
        return [
            name for name, schema in self.parameters.items()
            if schema.get("required", False)
        ]
    
    def validate_args(self, **kwargs) -> list[str]:
        """Validate arguments against schema.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required params
        for param in self._get_required_params():
            if param not in kwargs:
                errors.append(f"Missing required parameter: {param}")
        
        # Check types
        for name, value in kwargs.items():
            if name in self.parameters:
                schema = self.parameters[name]
                expected_type = schema.get("type")
                if expected_type and not self._check_type(value, expected_type):
                    errors.append(f"Invalid type for {name}: expected {expected_type}")
        
        return errors
    
    def _check_type(self, value: Any, expected: str) -> bool:
        """Check if value matches expected JSON schema type."""
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        return isinstance(value, type_map.get(expected, object))
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name})>"
