"""MCP (Model Context Protocol) client for IAB OpenDirect server."""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from .base import ToolResult, ToolStatus

logger = logging.getLogger(__name__)


@dataclass
class MCPClientConfig:
    """Configuration for MCP client."""
    
    server_url: str = "https://agentic-direct-server-hwgrypmndq-uk.a.run.app"
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    headers: dict[str, str] = field(default_factory=dict)


class MCPError(Exception):
    """Error from MCP protocol."""
    
    def __init__(self, message: str, code: int | None = None, data: Any = None):
        super().__init__(message)
        self.code = code
        self.data = data


class MCPClient:
    """Client for MCP protocol communication.
    
    Handles communication with the IAB OpenDirect MCP server,
    including tool discovery and execution.
    """
    
    def __init__(self, config: MCPClientConfig | None = None):
        """Initialize MCP client.
        
        Args:
            config: Client configuration (uses defaults if not provided)
        """
        self.config = config or MCPClientConfig()
        self._client: httpx.AsyncClient | None = None
        self._tools_cache: list[dict] | None = None
    
    @property
    def server_url(self) -> str:
        """Get the server URL."""
        return self.config.server_url
    
    async def __aenter__(self) -> "MCPClient":
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, *args) -> None:
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self) -> None:
        """Establish connection to MCP server."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.server_url,
                timeout=self.config.timeout,
                headers=self.config.headers,
            )
            logger.info(f"Connected to MCP server: {self.config.server_url}")
    
    async def disconnect(self) -> None:
        """Close connection to MCP server."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("Disconnected from MCP server")
    
    async def _ensure_connected(self) -> httpx.AsyncClient:
        """Ensure client is connected."""
        if self._client is None:
            await self.connect()
        return self._client  # type: ignore
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs,
    ) -> dict[str, Any]:
        """Make an HTTP request to the MCP server.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Additional request arguments
            
        Returns:
            Response data as dictionary
            
        Raises:
            MCPError: If request fails
        """
        client = await self._ensure_connected()
        
        last_error: Exception | None = None
        
        for attempt in range(self.config.max_retries):
            try:
                response = await client.request(method, endpoint, **kwargs)
                
                if response.status_code == 429:
                    # Rate limited - wait and retry
                    retry_after = float(response.headers.get("Retry-After", self.config.retry_delay))
                    logger.warning(f"Rate limited, waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    continue
                
                response.raise_for_status()
                return response.json()
                
            except httpx.HTTPStatusError as e:
                last_error = MCPError(
                    f"HTTP {e.response.status_code}: {e.response.text}",
                    code=e.response.status_code,
                )
                if e.response.status_code < 500:
                    raise last_error
                # Retry on 5xx errors
                
            except httpx.RequestError as e:
                last_error = MCPError(f"Request failed: {e}")
            
            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))
        
        raise last_error or MCPError("Request failed after retries")
    
    async def list_tools(self) -> list[dict]:
        """List available tools from server.
        
        Returns:
            List of tool definitions
        """
        if self._tools_cache is not None:
            return self._tools_cache
        
        try:
            response = await self._request("GET", "/tools")
            self._tools_cache = response.get("tools", [])
            logger.info(f"Discovered {len(self._tools_cache)} tools from MCP server")
            return self._tools_cache
        except MCPError as e:
            logger.error(f"Failed to list tools: {e}")
            return []
    
    async def call_tool(self, name: str, arguments: dict) -> dict:
        """Call an MCP tool on the IAB server.
        
        Args:
            name: Tool name
            arguments: Tool arguments
            
        Returns:
            Tool execution result
            
        Raises:
            MCPError: If tool call fails
        """
        logger.debug(f"Calling tool '{name}' with args: {arguments}")
        
        payload = {
            "name": name,
            "arguments": arguments,
        }
        
        response = await self._request(
            "POST",
            "/tools/call",
            json=payload,
        )
        
        logger.debug(f"Tool '{name}' response: {response}")
        return response
    
    async def execute_tool(self, name: str, arguments: dict) -> ToolResult:
        """Execute a tool and return a ToolResult.
        
        Args:
            name: Tool name
            arguments: Tool arguments
            
        Returns:
            ToolResult with execution outcome
        """
        try:
            response = await self.call_tool(name, arguments)
            
            # Check if response indicates error
            if "error" in response:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    error=response["error"].get("message", str(response["error"])),
                    metadata={"tool": name, "raw_response": response},
                )
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=response.get("result", response),
                metadata={"tool": name},
            )
            
        except MCPError as e:
            status = ToolStatus.RATE_LIMITED if e.code == 429 else ToolStatus.ERROR
            return ToolResult(
                status=status,
                error=str(e),
                metadata={"tool": name, "error_code": e.code},
            )
        except asyncio.TimeoutError:
            return ToolResult(
                status=ToolStatus.TIMEOUT,
                error=f"Tool '{name}' timed out",
                metadata={"tool": name},
            )
    
    def invalidate_cache(self) -> None:
        """Invalidate the tools cache."""
        self._tools_cache = None
    
    async def health_check(self) -> bool:
        """Check if server is healthy.
        
        Returns:
            True if server is responding
        """
        try:
            await self._request("GET", "/health")
            return True
        except Exception:
            return False
    
    def __repr__(self) -> str:
        connected = "connected" if self._client else "disconnected"
        return f"<MCPClient(url={self.config.server_url}, {connected})>"
