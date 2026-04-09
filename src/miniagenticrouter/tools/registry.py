"""Tool registry for automatic tool discovery and registration.

This module provides:
- ToolRegistry: Singleton registry for all available tools
- register_tool: Decorator for registering tool classes

Example:
    >>> from miniagenticrouter.tools.registry import register_tool, ToolRegistry
    >>>
    >>> @register_tool("my_tool")
    ... class MyTool(BaseTool):
    ...     name = "my_tool"
    ...     ...
    >>>
    >>> registry = ToolRegistry.get_instance()
    >>> tool = registry.create_tool("my_tool", arg1="value")
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any, Callable, TypeVar

if TYPE_CHECKING:
    from miniagenticrouter.tools.base import BaseTool

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="BaseTool")


class ToolRegistry:
    """Singleton registry for all available tools.

    The registry allows tools to be registered either:
    1. Via the @register_tool decorator
    2. Programmatically via registry.register_class()
    3. By full module path (lazy loading)

    Example:
        >>> registry = ToolRegistry.get_instance()
        >>> registry.list_tools()
        ['bash', 'python', 'search', ...]
        >>> tool = registry.create_tool("search", api_key="...")
    """

    _instance: ToolRegistry | None = None

    def __init__(self) -> None:
        """Initialize the registry. Use get_instance() instead."""
        self._tool_classes: dict[str, type[BaseTool]] = {}
        self._tool_paths: dict[str, str] = {}  # name -> module.path.ClassName
        self._loaded_modules: set[str] = set()

    @classmethod
    def get_instance(cls) -> ToolRegistry:
        """Get the singleton registry instance.

        Returns:
            The global ToolRegistry instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (mainly for testing)."""
        cls._instance = None

    def register(self, name: str | None = None) -> Callable[[type[T]], type[T]]:
        """Decorator to register a tool class.

        Args:
            name: Optional tool name. If not provided, uses class.name attribute
                  or class name in lowercase.

        Returns:
            Decorator function.

        Example:
            >>> @register_tool("search")
            ... class SearchTool(BaseTool):
            ...     ...
        """

        def decorator(cls: type[T]) -> type[T]:
            tool_name = name
            if tool_name is None:
                tool_name = getattr(cls, "name", None) or cls.__name__.lower().replace("tool", "")

            self._tool_classes[tool_name] = cls
            logger.debug(f"Registered tool: {tool_name} -> {cls.__module__}.{cls.__name__}")
            return cls

        return decorator

    def register_class(self, name: str, cls: type[BaseTool]) -> None:
        """Register a tool class programmatically.

        Args:
            name: Tool name to register under.
            cls: Tool class to register.
        """
        self._tool_classes[name] = cls
        logger.debug(f"Registered tool: {name} -> {cls.__module__}.{cls.__name__}")

    def register_path(self, name: str, path: str) -> None:
        """Register a tool by its module path for lazy loading.

        Args:
            name: Tool name to register under.
            path: Full module path (e.g., "miniagenticrouter.tools.builtin.search.WebSearchTool").
        """
        self._tool_paths[name] = path
        logger.debug(f"Registered tool path: {name} -> {path}")

    def get_tool_class(self, name: str) -> type[BaseTool]:
        """Get a tool class by name.

        Args:
            name: Tool name or full module path.

        Returns:
            The tool class.

        Raises:
            ValueError: If the tool is not found.
        """
        # First check already loaded classes
        if name in self._tool_classes:
            return self._tool_classes[name]

        # Check if it's a registered path (lazy load)
        if name in self._tool_paths:
            path = self._tool_paths[name]
            cls = self._load_class_from_path(path)
            self._tool_classes[name] = cls
            return cls

        # Try to interpret as a full module path
        if "." in name:
            try:
                cls = self._load_class_from_path(name)
                return cls
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Could not load tool from path '{name}': {e}") from e

        # Not found
        available = list(self._tool_classes.keys()) + list(self._tool_paths.keys())
        raise ValueError(f"Unknown tool: '{name}'. Available tools: {available}")

    def _load_class_from_path(self, path: str) -> type[BaseTool]:
        """Load a class from its full module path.

        Args:
            path: Full path like "module.submodule.ClassName".

        Returns:
            The loaded class.
        """
        module_name, class_name = path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    def create_tool(self, name: str, **kwargs: Any) -> BaseTool:
        """Create a tool instance by name.

        Args:
            name: Tool name or full module path.
            **kwargs: Arguments passed to tool constructor.

        Returns:
            Instantiated tool.
        """
        cls = self.get_tool_class(name)
        return cls(**kwargs)

    def list_tools(self) -> list[str]:
        """List all registered tool names.

        Returns:
            List of tool names.
        """
        all_names = set(self._tool_classes.keys()) | set(self._tool_paths.keys())
        return sorted(all_names)

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered.

        Args:
            name: Tool name to check.

        Returns:
            True if tool exists.
        """
        return name in self._tool_classes or name in self._tool_paths or "." in name

    def load_builtin_tools(self) -> None:
        """Load all built-in tools by importing the builtin module.

        This triggers @register_tool decorators to execute.
        """
        if "miniagenticrouter.tools.builtin" in self._loaded_modules:
            return

        try:
            import miniagenticrouter.tools.builtin  # noqa: F401

            self._loaded_modules.add("miniagenticrouter.tools.builtin")
            logger.debug("Loaded builtin tools")
        except ImportError as e:
            logger.warning(f"Could not load builtin tools: {e}")

    def get_all_schemas(self, tool_names: list[str] | None = None) -> list[dict[str, Any]]:
        """Get schemas for multiple tools.

        Args:
            tool_names: List of tool names. If None, returns schemas for all tools.

        Returns:
            List of tool schema dicts.
        """
        names = tool_names if tool_names is not None else self.list_tools()
        schemas = []

        for name in names:
            try:
                cls = self.get_tool_class(name)
                # Create a temporary instance to get schema
                # Note: This may fail for tools that require constructor args
                instance = cls.__new__(cls)
                if hasattr(instance, "name"):
                    instance.name = name
                if hasattr(instance, "description"):
                    instance.description = getattr(cls, "description", f"Tool: {name}")

                # Try to get schema without full initialization
                if hasattr(cls, "get_schema"):
                    try:
                        schema = instance.get_schema()
                        schemas.append(schema.to_dict())
                    except Exception:
                        # Fallback: create minimal schema
                        schemas.append(
                            {
                                "name": name,
                                "description": getattr(cls, "description", f"Tool: {name}"),
                                "parameters": {"type": "object", "properties": {}},
                            }
                        )
            except Exception as e:
                logger.warning(f"Could not get schema for tool '{name}': {e}")

        return schemas


# Global registry instance access
_registry = ToolRegistry.get_instance()


def register_tool(name: str | None = None) -> Callable[[type[T]], type[T]]:
    """Convenience decorator for registering tools.

    Args:
        name: Optional tool name. If not provided, uses class.name or class name.

    Returns:
        Decorator function.

    Example:
        >>> @register_tool("search")
        ... class WebSearchTool(BaseTool):
        ...     name = "search"
        ...     description = "Search the web"
        ...     ...
    """
    return _registry.register(name)


def get_registry() -> ToolRegistry:
    """Get the global tool registry.

    Returns:
        The singleton ToolRegistry instance.
    """
    return ToolRegistry.get_instance()
