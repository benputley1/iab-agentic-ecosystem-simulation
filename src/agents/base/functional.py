"""
L3 Functional Agent Base Class.

Functional agents are the leaf nodes of the hierarchy that execute
specific tools (MCP tools) and return results up the chain.
Uses Claude Sonnet for efficient tool execution.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, TypeVar
from uuid import uuid4

from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field
import structlog

from .context import (
    AgentContext,
    ContextPriority,
    ContextWindow,
    StandardContextPassing,
)
from .state import (
    AgentState,
    StateManager,
    StateBackend,
    VolatileStateBackend,
)

logger = structlog.get_logger(__name__)

# Default model for L3 functional agents
L3_MODEL = "claude-sonnet-4-20250514"


class ToolResult(BaseModel):
    """Result from executing an MCP tool."""
    tool_name: str
    success: bool
    result: Any = None
    error: str | None = None
    execution_time_ms: float = 0
    token_usage: dict[str, int] = Field(default_factory=dict)
    
    @property
    def is_error(self) -> bool:
        return not self.success or self.error is not None


class ToolDefinition(BaseModel):
    """Definition of an MCP tool the agent can execute."""
    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    required_params: list[str] = Field(default_factory=list)
    
    def to_anthropic_tool(self) -> dict[str, Any]:
        """Convert to Anthropic API tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required_params
            }
        }


class FunctionalAgentState(AgentState):
    """State specific to L3 functional agents."""
    agent_type: str = "functional"
    
    # Tool execution tracking
    tools_executed: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_execution_time_ms: float = 0
    
    # Token tracking
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    
    def record_execution(self, result: ToolResult) -> None:
        """Record a tool execution."""
        self.tools_executed += 1
        if result.success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1
        self.total_execution_time_ms += result.execution_time_ms
        
        if result.token_usage:
            self.total_input_tokens += result.token_usage.get("input_tokens", 0)
            self.total_output_tokens += result.token_usage.get("output_tokens", 0)
        
        self.updated_at = time.time()
    
    @property
    def success_rate(self) -> float:
        """Tool execution success rate."""
        if self.tools_executed == 0:
            return 1.0
        return self.successful_executions / self.tools_executed
    
    @property
    def avg_execution_time_ms(self) -> float:
        """Average execution time per tool."""
        if self.tools_executed == 0:
            return 0.0
        return self.total_execution_time_ms / self.tools_executed


class FunctionalAgent(ABC):
    """
    L3 Functional Agent Base Class.
    
    Functional agents are responsible for executing specific MCP tools.
    They receive focused context from L2 specialists, execute tools,
    and return structured results.
    
    Uses Claude Sonnet for cost-effective tool execution.
    """
    
    def __init__(
        self,
        agent_id: str | None = None,
        name: str = "FunctionalAgent",
        anthropic_client: AsyncAnthropic | None = None,
        model: str = L3_MODEL,
        state_backend: StateBackend | None = None,
        max_tool_retries: int = 2
    ):
        self.agent_id = agent_id or str(uuid4())
        self.name = name
        self.model = model
        self.max_tool_retries = max_tool_retries
        
        # Anthropic client
        self.client = anthropic_client or AsyncAnthropic()
        
        # State management
        self.state_backend = state_backend or VolatileStateBackend()
        self.state_manager = StateManager(self.state_backend)
        self.state: FunctionalAgentState | None = None
        
        # Context window tracking
        self.context_window = ContextWindow(max_tokens=100000)
        
        # Tool registry
        self._tools: dict[str, ToolDefinition] = {}
        self._tool_handlers: dict[str, Callable] = {}
        
        # Logging
        self.log = structlog.get_logger(__name__).bind(
            agent_id=self.agent_id,
            agent_name=self.name,
            level="L3"
        )
        
        # Register tools from subclass
        self._register_tools()
    
    @abstractmethod
    def _register_tools(self) -> None:
        """
        Register available tools. Subclasses must implement.
        
        Example:
            self.register_tool(
                ToolDefinition(
                    name="product_search",
                    description="Search for advertising products",
                    parameters={"query": {"type": "string"}},
                    required_params=["query"]
                ),
                handler=self._handle_product_search
            )
        """
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this functional agent."""
        pass
    
    def register_tool(
        self,
        tool: ToolDefinition,
        handler: Callable[..., Any]
    ) -> None:
        """Register a tool and its handler."""
        self._tools[tool.name] = tool
        self._tool_handlers[tool.name] = handler
        self.log.debug("tool_registered", tool_name=tool.name)
    
    async def initialize(self) -> None:
        """Initialize agent state."""
        self.state = await self.state_manager.initialize_state(
            agent_id=self.agent_id,
            agent_type="functional",
            initial_data={
                "name": self.name,
                "model": self.model,
                "available_tools": list(self._tools.keys())
            }
        )
        self.log.info("agent_initialized", tools=list(self._tools.keys()))
    
    async def execute(
        self,
        context: AgentContext,
        instructions: str
    ) -> dict[str, Any]:
        """
        Execute task based on context and instructions from L2.
        
        Args:
            context: Context passed down from L2 specialist
            instructions: Specific instructions for this execution
            
        Returns:
            Dictionary with results and metadata
        """
        start_time = time.time()
        
        if self.state is None:
            await self.initialize()
        
        self.log.info(
            "execution_started",
            task=instructions[:100],
            context_items=len(context.items)
        )
        
        # Build messages
        messages = self._build_messages(context, instructions)
        
        # Get tool definitions for API
        tools = [tool.to_anthropic_tool() for tool in self._tools.values()]
        
        # Execute with tool use loop
        results = []
        iteration = 0
        max_iterations = 10  # Safety limit
        
        while iteration < max_iterations:
            iteration += 1
            
            try:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=self.get_system_prompt(),
                    messages=messages,
                    tools=tools if tools else None
                )
                
                # Track token usage
                self.context_window.record_usage(
                    response.usage.input_tokens,
                    response.usage.output_tokens
                )
                
                # Process response
                has_tool_use = False
                tool_results = []
                final_text = ""
                
                for block in response.content:
                    if block.type == "tool_use":
                        has_tool_use = True
                        tool_result = await self._execute_tool(
                            block.name,
                            block.input
                        )
                        results.append(tool_result)
                        
                        # Add to message history
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(tool_result.result) if tool_result.success else f"Error: {tool_result.error}"
                        })
                    elif block.type == "text":
                        final_text = block.text
                
                if has_tool_use:
                    # Continue conversation with tool results
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": tool_results})
                else:
                    # No more tool calls, we're done
                    break
                    
            except Exception as e:
                self.log.error("execution_error", error=str(e))
                if self.state:
                    self.state.record_error(str(e))
                raise
        
        execution_time = (time.time() - start_time) * 1000
        
        # Compile final result
        result = {
            "agent_id": self.agent_id,
            "agent_name": self.name,
            "success": all(r.success for r in results) if results else True,
            "tool_results": [r.model_dump() for r in results],
            "final_response": final_text,
            "execution_time_ms": execution_time,
            "iterations": iteration,
            "token_usage": {
                "input_tokens": self.context_window.history_tokens,
                "output_tokens": sum(r.token_usage.get("output_tokens", 0) for r in results)
            }
        }
        
        self.log.info(
            "execution_completed",
            success=result["success"],
            tools_called=len(results),
            time_ms=execution_time
        )
        
        return result
    
    def _build_messages(
        self,
        context: AgentContext,
        instructions: str
    ) -> list[dict[str, Any]]:
        """Build message list from context and instructions."""
        # Compile context into message
        context_parts = []
        
        if context.task_description:
            context_parts.append(f"Task: {context.task_description}")
        
        if context.task_constraints:
            context_parts.append(f"Constraints: {', '.join(context.task_constraints)}")
        
        # Add context items by priority
        for priority in [ContextPriority.CRITICAL, ContextPriority.HIGH, 
                         ContextPriority.MEDIUM, ContextPriority.LOW]:
            items = context.get_by_priority(priority)
            if items:
                context_parts.append(f"\n{priority.value.upper()} Context:")
                for key, value in items.items():
                    context_parts.append(f"  {key}: {value}")
        
        context_str = "\n".join(context_parts)
        
        return [
            {
                "role": "user",
                "content": f"""Context from upstream agent:
{context_str}

Instructions:
{instructions}

Execute the required tools to complete this task."""
            }
        ]
    
    async def _execute_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any]
    ) -> ToolResult:
        """Execute a specific tool with retry logic."""
        start_time = time.time()
        
        if tool_name not in self._tool_handlers:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Unknown tool: {tool_name}"
            )
        
        handler = self._tool_handlers[tool_name]
        last_error = None
        
        for attempt in range(self.max_tool_retries + 1):
            try:
                self.log.debug(
                    "tool_executing",
                    tool=tool_name,
                    attempt=attempt + 1,
                    input=tool_input
                )
                
                # Execute handler (may be sync or async)
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(**tool_input)
                else:
                    result = handler(**tool_input)
                
                execution_time = (time.time() - start_time) * 1000
                
                tool_result = ToolResult(
                    tool_name=tool_name,
                    success=True,
                    result=result,
                    execution_time_ms=execution_time
                )
                
                if self.state:
                    self.state.record_execution(tool_result)
                
                self.log.info(
                    "tool_success",
                    tool=tool_name,
                    time_ms=execution_time
                )
                
                return tool_result
                
            except Exception as e:
                last_error = str(e)
                self.log.warning(
                    "tool_attempt_failed",
                    tool=tool_name,
                    attempt=attempt + 1,
                    error=last_error
                )
                
                if attempt < self.max_tool_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))  # Backoff
        
        execution_time = (time.time() - start_time) * 1000
        
        tool_result = ToolResult(
            tool_name=tool_name,
            success=False,
            error=last_error,
            execution_time_ms=execution_time
        )
        
        if self.state:
            self.state.record_execution(tool_result)
        
        self.log.error(
            "tool_failed",
            tool=tool_name,
            error=last_error
        )
        
        return tool_result
    
    async def cleanup(self) -> None:
        """Cleanup agent resources."""
        if self.state:
            await self.state_manager.create_snapshot(
                self.state,
                description="Agent cleanup snapshot",
                recovery_point=True
            )
        self.log.info("agent_cleanup_complete")
