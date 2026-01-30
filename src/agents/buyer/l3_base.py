"""Base class for L3 Functional Agents.

L3 agents are the execution layer in the buyer agent hierarchy:
- L1: Portfolio Manager (strategic orchestration)
- L2: Channel Specialists (channel-specific coordination)
- L3: Functional Agents (tool execution)

This module provides the base class for L3 agents that execute
specific tools with LLM-guided decision making.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, TypeVar, Generic
from enum import Enum

import anthropic

from .config import buyer_settings


class ToolExecutionStatus(str, Enum):
    """Status of tool execution."""
    
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"
    TIMEOUT = "timeout"


@dataclass
class ToolResult:
    """Result from executing a tool."""
    
    tool_name: str
    status: ToolExecutionStatus
    data: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def success(self) -> bool:
        """Whether the tool execution succeeded."""
        return self.status == ToolExecutionStatus.SUCCESS


@dataclass
class AgentContext:
    """Context passed to functional agents."""
    
    buyer_id: str
    scenario: str
    campaign_id: Optional[str] = None
    channel: Optional[str] = None
    session_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)


T = TypeVar("T")


class FunctionalAgent(ABC, Generic[T]):
    """Base class for L3 Functional Agents.
    
    Functional agents execute specific tools to accomplish tasks.
    They use Claude Sonnet for intelligent tool selection and
    parameter extraction.
    
    Subclasses must implement:
    - system_prompt: The system prompt for the LLM
    - available_tools: List of tools this agent can use
    - _execute_tool: Execute a selected tool
    
    Example:
        ```python
        class ResearchAgent(FunctionalAgent):
            @property
            def system_prompt(self) -> str:
                return "You are a research specialist..."
            
            @property
            def available_tools(self) -> list[dict]:
                return [ProductSearchTool.schema(), ...]
            
            async def _execute_tool(self, name: str, params: dict) -> ToolResult:
                if name == "ProductSearch":
                    return await self._search_products(params)
        ```
    """
    
    def __init__(
        self,
        context: AgentContext,
        client: Optional[anthropic.AsyncAnthropic] = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
    ):
        """Initialize functional agent.
        
        Args:
            context: Agent context with buyer/scenario info
            client: Optional Anthropic client (creates one if not provided)
            model: Model to use for tool selection
            max_tokens: Maximum tokens for LLM responses
        """
        self.context = context
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._execution_history: list[ToolResult] = []
    
    @property
    def client(self) -> anthropic.AsyncAnthropic:
        """Get or create Anthropic client."""
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(
                api_key=buyer_settings.anthropic_api_key
            )
        return self._client
    
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt for this agent's LLM."""
        pass
    
    @property
    @abstractmethod
    def available_tools(self) -> list[dict]:
        """List of tool schemas available to this agent."""
        pass
    
    @property
    def agent_name(self) -> str:
        """Human-readable name for this agent."""
        return self.__class__.__name__
    
    async def execute(self, task: str) -> ToolResult:
        """Execute a task using available tools.
        
        This method:
        1. Sends the task to the LLM with available tools
        2. Processes tool calls from the response
        3. Returns the final result
        
        Args:
            task: Natural language description of the task
            
        Returns:
            ToolResult with execution outcome
        """
        start_time = datetime.utcnow()
        
        try:
            # Call LLM with tools
            response = await self.client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=self.system_prompt,
                tools=self.available_tools,
                messages=[{"role": "user", "content": task}]
            )
            
            # Process response
            for block in response.content:
                if block.type == "tool_use":
                    # Execute the tool
                    result = await self._execute_tool(
                        name=block.name,
                        params=block.input,
                    )
                    
                    # Track execution time
                    end_time = datetime.utcnow()
                    result.execution_time_ms = (
                        (end_time - start_time).total_seconds() * 1000
                    )
                    
                    self._execution_history.append(result)
                    return result
            
            # No tool was called - return text response
            text_content = next(
                (b.text for b in response.content if b.type == "text"),
                "No action taken"
            )
            
            return ToolResult(
                tool_name="none",
                status=ToolExecutionStatus.SUCCESS,
                data={"message": text_content},
            )
            
        except anthropic.APIError as e:
            return ToolResult(
                tool_name="unknown",
                status=ToolExecutionStatus.FAILED,
                error=f"API error: {e}",
            )
        except Exception as e:
            return ToolResult(
                tool_name="unknown",
                status=ToolExecutionStatus.FAILED,
                error=f"Execution error: {e}",
            )
    
    @abstractmethod
    async def _execute_tool(self, name: str, params: dict) -> ToolResult:
        """Execute a specific tool with given parameters.
        
        Args:
            name: Tool name
            params: Tool parameters extracted by LLM
            
        Returns:
            ToolResult with execution outcome
        """
        pass
    
    def get_execution_history(self) -> list[ToolResult]:
        """Get history of tool executions."""
        return self._execution_history.copy()
    
    def clear_history(self) -> None:
        """Clear execution history."""
        self._execution_history.clear()
