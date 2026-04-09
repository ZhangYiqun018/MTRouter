"""Base classes for multi-tool architecture.

This module provides:
- ToolResult: Structured result from tool execution
- ToolSchema: JSON Schema describing tool parameters
- BaseTool: Abstract base class for all tools with optional caching support
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .cache import GlobalToolCache


@dataclass
class ToolResult:
    """Structured result from tool execution.

    Attributes:
        output: The main output/observation from the tool.
        returncode: Exit code (0 for success, non-zero for error).
        done: Whether this result signals task completion.
        error: Error message if execution failed.
        metadata: Additional tool-specific data.
    """

    output: str
    returncode: int = 0
    done: bool = False
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for backwards compatibility with legacy tools."""
        return {
            "output": self.output,
            "returncode": self.returncode,
            "done": self.done,
            "error": self.error,
            **self.metadata,
        }


@dataclass
class ToolSchema:
    """Schema describing a tool's parameters using JSON Schema format.

    Attributes:
        name: Tool name identifier.
        description: Human-readable description of what the tool does.
        parameters: JSON Schema object describing the parameters.
        required: List of required parameter names.
        examples: Example invocations for documentation/prompts.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    required: list[str] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "required": self.required,
            "examples": self.examples,
        }

    def to_xml(self) -> str:
        """Convert schema to XML format for prompt rendering.

        Returns XML in format:
        <tool name="tool_name">
          <description>...</description>
          <parameters>
            <param name="x" type="string" required="true">...</param>
          </parameters>
        </tool>
        """
        lines = [f'<tool name="{self.name}">']
        lines.append(f"  <description>{self.description}</description>")
        lines.append("  <parameters>")

        props = self.parameters.get("properties", {})
        for param_name, param_info in props.items():
            required = "true" if param_name in self.required else "false"
            param_type = param_info.get("type", "string")
            desc = param_info.get("description", "")
            lines.append(
                f'    <param name="{param_name}" type="{param_type}" required="{required}">{desc}</param>'
            )

        lines.append("  </parameters>")
        lines.append("</tool>")
        return "\n".join(lines)


class BaseTool(ABC):
    """Abstract base class for tools with structured parameter support.

    All tools should inherit from this class and implement:
    - get_schema(): Return JSON Schema describing parameters
    - execute(**kwargs): Execute the tool with keyword arguments

    Caching Support:
        Tools can opt-in to caching by setting class attributes:
        - cacheable: bool = True  # Enable caching
        - cache_key_fields: list[str] | None = ["field1"]  # Fields for cache key

        Subclasses can also override these methods for custom behavior:
        - compute_cache_key(**kwargs): Customize cache key generation
        - should_cache_result(result, **kwargs): Customize caching conditions

    Example:
        >>> class MyTool(BaseTool):
        ...     name = "my_tool"
        ...     description = "Does something useful"
        ...     cacheable = True  # Enable caching
        ...     cache_key_fields = ["input"]  # Only use 'input' for cache key
        ...
        ...     def get_schema(self) -> ToolSchema:
        ...         return ToolSchema(
        ...             name=self.name,
        ...             description=self.description,
        ...             parameters={
        ...                 "type": "object",
        ...                 "properties": {
        ...                     "input": {"type": "string"}
        ...                 }
        ...             },
        ...             required=["input"]
        ...         )
        ...
        ...     def execute(self, input: str, **kwargs) -> ToolResult:
        ...         return ToolResult(output=f"Processed: {input}")
    """

    name: str = ""
    description: str = ""

    # ========== Cache-related class attributes ==========
    # Subclasses can override these to customize caching behavior
    cacheable: bool = False  # Whether to enable caching for this tool
    cache_key_fields: list[str] | None = None  # Fields to use for cache key (None = all)

    @abstractmethod
    def get_schema(self) -> ToolSchema:
        """Return JSON Schema describing the tool's parameters.

        Returns:
            ToolSchema with name, description, and parameter definitions.
        """
        ...

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with keyword arguments.

        Args:
            **kwargs: Tool-specific arguments matching the schema.

        Returns:
            ToolResult with output and metadata.
        """
        ...

    def validate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalize arguments against schema.

        Args:
            args: Arguments dict to validate.

        Returns:
            Validated and normalized arguments.

        Raises:
            ValueError: If required parameters are missing.
            TypeError: If parameter types don't match schema.
        """
        schema = self.get_schema()

        # Check required parameters
        for req in schema.required:
            if req not in args:
                raise ValueError(f"Missing required parameter '{req}' for tool '{self.name}'")

        # Basic type validation for known types
        properties = schema.parameters.get("properties", {})
        validated = {}

        for key, value in args.items():
            if key in properties:
                prop_schema = properties[key]
                prop_type = prop_schema.get("type")

                # Type coercion for common cases
                if prop_type == "string" and not isinstance(value, str):
                    value = str(value)
                elif prop_type == "integer" and not isinstance(value, int):
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        raise TypeError(f"Parameter '{key}' must be an integer")
                elif prop_type == "number" and not isinstance(value, (int, float)):
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        raise TypeError(f"Parameter '{key}' must be a number")
                elif prop_type == "boolean" and not isinstance(value, bool):
                    if isinstance(value, str):
                        value = value.lower() in ("true", "1", "yes")
                    else:
                        value = bool(value)
                elif prop_type == "array" and not isinstance(value, list):
                    if isinstance(value, str):
                        # Try to handle comma-separated values
                        value = [v.strip() for v in value.split(",")]
                    else:
                        value = [value]

            validated[key] = value

        return validated

    def get_config(self) -> dict[str, Any]:
        """Return tool configuration for template rendering.

        Returns:
            Dict with tool name, description, and schema.
        """
        schema = self.get_schema()
        return {
            "name": self.name,
            "description": self.description,
            "parameters": schema.parameters,
            "required": schema.required,
            "examples": schema.examples,
        }

    # ========== Cache-related methods ==========

    _cache: GlobalToolCache | None = None  # Lazy-initialized cache instance

    @property
    def cache(self) -> GlobalToolCache:
        """Get the global cache instance (lazy initialization)."""
        if self._cache is None:
            from .cache import GlobalToolCache

            self._cache = GlobalToolCache.get_instance()
        return self._cache

    def compute_cache_key(self, **kwargs) -> str:
        """Compute cache key from arguments.

        Subclasses can override this to customize key generation.

        Default behavior:
        - If cache_key_fields is set, only use those fields
        - Otherwise, use all arguments

        Args:
            **kwargs: Tool arguments

        Returns:
            Cache key string in format "tool_name:hash"
        """
        if self.cache_key_fields is not None:
            key_data = {k: kwargs.get(k) for k in self.cache_key_fields}
        else:
            key_data = kwargs

        normalized = json.dumps(key_data, sort_keys=True, default=str)
        hash_val = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        return f"{self.name}:{hash_val}"

    def should_cache_result(self, result: ToolResult, **kwargs) -> bool:
        """Determine if result should be cached.

        Subclasses can override this to customize caching conditions.

        Default behavior: only cache successful results (returncode == 0)
        that don't signal task completion (done == False).

        Args:
            result: The execution result
            **kwargs: Original tool arguments

        Returns:
            True if result should be cached
        """
        return result.returncode == 0 and not result.done

    def execute_with_cache(self, **kwargs) -> ToolResult:
        """Execute tool with caching support.

        This is the main entry point for cached execution. It:
        1. Checks if caching is enabled for this tool
        2. Looks up existing cache entry
        3. On miss, executes the tool and caches the result

        Args:
            **kwargs: Tool arguments

        Returns:
            ToolResult (from cache or fresh execution)
        """
        # If caching is disabled, just execute directly
        if not self.cacheable:
            return self.execute(**kwargs)

        # Compute cache key
        cache_key = self.compute_cache_key(**kwargs)

        # Check cache
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            # Mark as cache hit in metadata
            cached_result.metadata["cache_hit"] = True
            return cached_result

        # Execute tool
        result = self.execute(**kwargs)

        # Cache the result if appropriate
        if self.should_cache_result(result, **kwargs):
            self.cache.set(cache_key, result, tool_name=self.name)

        return result


class LegacyToolAdapter(BaseTool):
    """Adapter to wrap legacy tools into the new BaseTool interface.

    This allows existing tools (BashTool, TextCommandTool, etc.) to be used
    with the MultiToolAgent without modification.

    Example:
        >>> from miniagenticrouter.tools.bash import BashTool
        >>> legacy = BashTool(env=some_env)
        >>> adapted = LegacyToolAdapter(legacy)
        >>> result = adapted.execute(command="ls -la")
    """

    def __init__(self, legacy_tool: Any):
        """Initialize with a legacy tool instance.

        Args:
            legacy_tool: A tool with execute(action: dict) method.
        """
        self._legacy = legacy_tool
        self.name = getattr(legacy_tool, "name", legacy_tool.__class__.__name__)
        self.description = f"Legacy tool: {self.name}"

    def get_schema(self) -> ToolSchema:
        """Return a generic schema for legacy tools."""
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command to execute",
                    }
                },
            },
            required=["command"],
        )

    def execute(self, command: str = "", **kwargs) -> ToolResult:
        """Execute via the legacy tool interface.

        Args:
            command: Command string to execute.
            **kwargs: Additional arguments (ignored).

        Returns:
            ToolResult with output from legacy execution.
        """
        action = {"action": command, "action_type": self.name}
        result = self._legacy.execute(action)

        return ToolResult(
            output=result.get("output", ""),
            returncode=result.get("returncode", 0),
            done=result.get("done", False),
            error=result.get("error"),
            metadata={k: v for k, v in result.items() if k not in ("output", "returncode", "done", "error")},
        )
