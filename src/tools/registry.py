"""Tool registry for managing MCP tools."""

from typing import TYPE_CHECKING

from .base import Tool, ToolSpec

if TYPE_CHECKING:
    from .mcp_client import MCPClient


class ToolRegistry:
    """Registry of all available MCP tools.
    
    The registry manages tool registration, lookup, and provides
    tool specifications for LLM context injection.
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self._tools: dict[str, Tool] = {}
        self._mcp_client: "MCPClient | None" = None
    
    @property
    def tools(self) -> dict[str, Tool]:
        """Get all registered tools."""
        return self._tools.copy()
    
    def set_mcp_client(self, client: "MCPClient"):
        """Set the MCP client for remote tool calls."""
        self._mcp_client = client
    
    def register(self, tool: Tool) -> None:
        """Register a tool.
        
        Args:
            tool: Tool instance to register
            
        Raises:
            ValueError: If tool with same name already registered
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool
    
    def register_many(self, tools: list[Tool]) -> None:
        """Register multiple tools.
        
        Args:
            tools: List of tool instances to register
        """
        for tool in tools:
            self.register(tool)
    
    def unregister(self, name: str) -> Tool | None:
        """Unregister a tool by name.
        
        Args:
            name: Name of tool to unregister
            
        Returns:
            The unregistered tool, or None if not found
        """
        return self._tools.pop(name, None)
    
    def get(self, name: str) -> Tool | None:
        """Get tool by name.
        
        Args:
            name: Name of the tool
            
        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)
    
    def get_or_raise(self, name: str) -> Tool:
        """Get tool by name, raising if not found.
        
        Args:
            name: Name of the tool
            
        Returns:
            Tool instance
            
        Raises:
            KeyError: If tool not found
        """
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"Tool '{name}' not found in registry")
        return tool
    
    def has(self, name: str) -> bool:
        """Check if tool is registered.
        
        Args:
            name: Name of the tool
            
        Returns:
            True if tool is registered
        """
        return name in self._tools
    
    def list_tools(self) -> list[ToolSpec]:
        """List all available tools (for LLM context).
        
        Returns:
            List of tool specifications
        """
        return [tool.get_spec() for tool in self._tools.values()]
    
    def list_names(self) -> list[str]:
        """List all tool names.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def filter_by_prefix(self, prefix: str) -> list[Tool]:
        """Get tools whose names start with prefix.
        
        Args:
            prefix: Name prefix to filter by
            
        Returns:
            List of matching tools
        """
        return [
            tool for name, tool in self._tools.items()
            if name.startswith(prefix)
        ]
    
    def filter_by_category(self, category: str) -> list[Tool]:
        """Get tools by category (buyer/seller).
        
        Args:
            category: Category to filter by
            
        Returns:
            List of tools in category
        """
        # Category is encoded in tool module path
        return [
            tool for tool in self._tools.values()
            if category in tool.__class__.__module__
        ]
    
    def to_llm_context(self) -> list[dict]:
        """Export all tools in LLM-friendly format.
        
        Returns:
            List of tool specs as dictionaries
        """
        return [spec.to_dict() for spec in self.list_tools()]
    
    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()
    
    def __len__(self) -> int:
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        return name in self._tools
    
    def __iter__(self):
        return iter(self._tools.values())
    
    def __repr__(self) -> str:
        return f"<ToolRegistry(tools={len(self._tools)})>"


# Global registry instance
_default_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """Get the default global registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry()
    return _default_registry


def reset_registry() -> None:
    """Reset the default global registry."""
    global _default_registry
    _default_registry = None
