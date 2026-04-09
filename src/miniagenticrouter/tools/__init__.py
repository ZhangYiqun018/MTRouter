"""Tool abstractions for mini-SWE-agent.

This module provides abstractions for executing actions extracted from model responses.
Tools are responsible for executing actions and returning observations.

For the new multi-tool architecture, see:
- miniagenticrouter.tools.base: BaseTool, ToolResult, ToolSchema
- miniagenticrouter.tools.registry: ToolRegistry, register_tool
- miniagenticrouter.tools.builtin: Built-in tool implementations
- miniagenticrouter.tools.cache: GlobalToolCache, CacheConfig
"""

import copy
import importlib
from typing import Any, Protocol

# Re-export new tool system components
from miniagenticrouter.tools.base import BaseTool, LegacyToolAdapter, ToolResult, ToolSchema
from miniagenticrouter.tools.cache import CacheConfig, GlobalToolCache, compute_cache_key
from miniagenticrouter.tools.registry import ToolRegistry, get_registry, register_tool


class Tool(Protocol):
    """Protocol for tools that execute actions.

    Tools receive parsed actions and execute them in some environment,
    returning observations that can be fed back to the model.

    Note: For new tools, consider inheriting from BaseTool instead.
    """

    name: str

    def execute(self, action: dict) -> dict:
        """Execute an action and return the result.

        Args:
            action: Parsed action dict containing at least:
                - "action": The action string to execute (or None)
                - "action_type": Type of action (e.g., "bash", "text", "none")

        Returns:
            Result dict containing:
            - "output": Output/observation from execution
            - "returncode": Exit code (0 for success)
            - Additional tool-specific fields
        """
        ...

    def get_config(self) -> dict[str, Any]:
        """Return tool configuration for template rendering."""
        ...


# Legacy tool mapping (for backwards compatibility)
_TOOL_MAPPING = {
    # Legacy tools
    "bash": "miniagenticrouter.tools.bash.BashTool",
    "noop": "miniagenticrouter.tools.noop.NoOpTool",
    "text_command": "miniagenticrouter.tools.text_command.TextCommandTool",
    # New builtin tools (also available via registry)
    "search": "miniagenticrouter.tools.builtin.search.WebSearchTool",
    "browse": "miniagenticrouter.tools.builtin.browse.WebBrowseTool",
    "python": "miniagenticrouter.tools.builtin.python.PythonTool",
    "answer": "miniagenticrouter.tools.builtin.answer.AnswerTool",
    "file_read": "miniagenticrouter.tools.builtin.file_read.FileReadTool",
    "bash_v2": "miniagenticrouter.tools.builtin.bash.BashTool",
}


def get_tool_class(spec: str) -> type[Tool]:
    """Get a tool class by name or full path.

    Args:
        spec: Either a short name (e.g., "bash") or full import path
              (e.g., "miniagenticrouter.tools.bash.BashTool")

    Returns:
        The tool class.

    Raises:
        ValueError: If the tool is not found.
    """
    full_path = _TOOL_MAPPING.get(spec, spec)
    try:
        module_name, class_name = full_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ValueError, ImportError, AttributeError):
        msg = f"Unknown tool type: {spec} (resolved to {full_path}, available: {list(_TOOL_MAPPING.keys())})"
        raise ValueError(msg)


def get_tool(config: dict | None = None, *, default_type: str = "bash", **kwargs) -> Tool:
    """Create a tool instance from configuration.

    Args:
        config: Tool configuration dict. Should contain "tool_class" key
                to specify the tool type.
        default_type: Default tool type if not specified in config.
        **kwargs: Additional arguments passed to tool constructor.

    Returns:
        A Tool instance.
    """
    config = copy.deepcopy(config) if config else {}
    tool_class_spec = config.pop("tool_class", default_type)
    tool_class = get_tool_class(tool_class_spec)
    # Merge remaining config with kwargs
    config.update(kwargs)
    return tool_class(**config)
