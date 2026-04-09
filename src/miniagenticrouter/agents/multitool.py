"""Multi-tool agent for structured tool invocations.

This module provides MultiToolAgent, which extends FlexibleAgent with:
- Multiple tool support via ToolRegistry
- Tool routing based on tool_call.name
- Structured JSON parameters for tools
- Parallel tool execution
- Tool schema injection into prompts
"""

from __future__ import annotations

import concurrent.futures
import logging
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from jinja2 import StrictUndefined, Template

from miniagenticrouter import Environment, Model
from miniagenticrouter.agents.flexible import (
    FlexibleAgent,
    FlexibleAgentConfig,
    NonTerminatingException,
    Submitted,
)
from miniagenticrouter.parsers import get_parser
from miniagenticrouter.parsers.toolcall import ToolCall
from miniagenticrouter.tools.base import BaseTool, LegacyToolAdapter, ToolResult, ToolSchema
from miniagenticrouter.tools.registry import ToolRegistry

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class MultiToolAgentConfig(FlexibleAgentConfig):
    """Configuration for MultiToolAgent.

    Extends FlexibleAgentConfig with multi-tool specific options.

    Attributes:
        enabled_tools: List of tool names to enable.
        tool_configs: Per-tool configuration dicts.
        max_parallel_calls: Maximum number of parallel tool executions.
        answer_tool: Tool name that triggers task completion.
        include_tool_schemas: Whether to inject tool schemas into prompts.
        tool_response_template: Template for formatting tool responses.
        cache_enabled: Whether to enable tool result caching globally.
        cache_db_path: Path to SQLite cache database (None for in-memory).
        cache_max_size: Maximum number of cached entries.
        cache_ttl_seconds: Cache TTL in seconds (None for no expiration).
    """

    # Tool configuration
    enabled_tools: list[str] = field(default_factory=lambda: ["bash"])
    tool_configs: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Execution settings
    max_parallel_calls: int = 5
    answer_tool: str = "answer"

    # Prompt settings
    include_tool_schemas: bool = True
    tool_response_template: str = '<tool_response name="{{tool_name}}">\n{{output}}\n</tool_response>'

    # Cache settings
    cache_enabled: bool = True
    cache_db_path: str | None = "~/.miniagenticrouter/tool_cache.db"
    cache_max_size: int = 10000
    cache_ttl_seconds: int | None = None  # None = never expire

    # Override defaults for multi-tool mode
    action_mode: str = "multitool"


class MultiToolAgent(FlexibleAgent):
    """Agent that supports multiple tools with structured parameters.

    Key features over FlexibleAgent:
    1. Multiple tools managed via ToolRegistry
    2. Tool routing based on tool_call.name
    3. Structured JSON parameters for tools
    4. Tool schema injection into prompts
    5. Parallel tool execution support

    Example:
        >>> from miniagenticrouter.models import get_model
        >>> agent = MultiToolAgent(
        ...     model=get_model("anthropic/claude-sonnet-4-5-20250929"),
        ...     env=None,  # Optional for tools that don't need env
        ...     enabled_tools=["search", "python", "answer"],
        ...     tool_configs={
        ...         "search": {"api_key_env": "SERPER_API_KEY"},
        ...         "python": {"timeout": 60},
        ...     },
        ... )
        >>> exit_status, result = agent.run("What is 2^100?")
    """

    def __init__(
        self,
        model: Model,
        env: Environment | None = None,
        *,
        config_class: type = MultiToolAgentConfig,
        **kwargs: Any,
    ) -> None:
        """Initialize the multi-tool agent.

        Args:
            model: Language model for generating responses.
            env: Environment for tools that need command execution (optional).
            config_class: Configuration dataclass to use.
            **kwargs: Configuration options passed to config_class.
        """
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars: dict[str, Any] = {}

        # Initialize tools
        self._init_tools()

        # Initialize parser for tool calls
        self._init_parser_for_multitool()

        # Keep self.tool for backwards compatibility (first tool or None)
        self.tool = next(iter(self.tools.values())) if self.tools else None

    def _init_tools(self) -> None:
        """Initialize all enabled tools from the registry."""
        # Initialize global cache with config settings
        self._init_cache()

        registry = ToolRegistry.get_instance()
        registry.load_builtin_tools()

        self.tools: dict[str, BaseTool] = {}

        for tool_name in self.config.enabled_tools:
            tool_config = self.config.tool_configs.get(tool_name, {}).copy()

            # Inject environment for tools that need it
            if tool_name in ("bash", "python", "file_read") and self.env is not None:
                tool_config.setdefault("env", self.env)

            try:
                tool = registry.create_tool(tool_name, **tool_config)

                # Apply per-tool cache settings from config
                # tool_configs.search.cacheable = false would disable caching for search
                if "cacheable" in tool_config:
                    tool.cacheable = tool_config["cacheable"]
                if "cache_key_fields" in tool_config:
                    tool.cache_key_fields = tool_config["cache_key_fields"]

                # If global cache is disabled, disable per-tool caching
                if not self.config.cache_enabled:
                    tool.cacheable = False

                self.tools[tool_name] = tool
                logger.debug(f"Initialized tool: {tool_name} (cacheable={tool.cacheable})")
            except Exception as e:
                logger.warning(f"Could not initialize tool '{tool_name}': {e}")

    def _init_cache(self) -> None:
        """Initialize the global tool cache with config settings."""
        import os

        from miniagenticrouter.tools.cache import CacheConfig, GlobalToolCache

        # Expand path if provided
        db_path = None
        if self.config.cache_db_path:
            db_path = os.path.expanduser(self.config.cache_db_path)

        cache_config = CacheConfig(
            enabled=self.config.cache_enabled,
            max_size=self.config.cache_max_size,
            db_path=db_path,
            ttl_seconds=self.config.cache_ttl_seconds,
        )

        # Initialize global cache singleton
        GlobalToolCache.get_instance(cache_config)
        logger.debug(
            f"Initialized tool cache: enabled={cache_config.enabled}, "
            f"db_path={cache_config.db_path}, max_size={cache_config.max_size}"
        )

    def _init_parser_for_multitool(self) -> None:
        """Initialize parser for multi-tool mode."""
        parser_config = self.config.parser_config.copy()

        # Default to toolcall parser
        if not parser_config.get("parser_class"):
            parser_config["parser_class"] = "toolcall"

        # Pass known tool names for XML parsing support
        if "known_tools" not in parser_config:
            parser_config["known_tools"] = list(self.tools.keys())

        self.parser = get_parser(parser_config)

    def get_tool_schemas(self) -> list[ToolSchema]:
        """Get schemas for all enabled tools.

        Returns:
            List of ToolSchema objects for prompt rendering.
            These can be used in templates with methods like to_dict() or to_xml().
        """
        schemas = []
        for name, tool in self.tools.items():
            try:
                schema = tool.get_schema()
                schemas.append(schema)
            except Exception as e:
                logger.warning(f"Could not get schema for tool '{name}': {e}")
                schemas.append(
                    ToolSchema(
                        name=name,
                        description=getattr(tool, "description", f"Tool: {name}"),
                        parameters={"type": "object", "properties": {}},
                    )
                )
        return schemas

    def render_template(self, template: str, **kwargs: Any) -> str:
        """Render a Jinja2 template with agent context.

        Extends parent to inject tool schemas.

        Args:
            template: Jinja2 template string.
            **kwargs: Additional template variables.

        Returns:
            Rendered template string.
        """
        template_vars = asdict(self.config)

        # Add environment vars if available
        if self.env is not None:
            template_vars |= self.env.get_template_vars()

        # Add model vars
        template_vars |= self.model.get_template_vars()

        # Add parser config
        template_vars |= self.parser.get_config()

        # Add tool schemas
        if self.config.include_tool_schemas:
            template_vars["tool_schemas"] = self.get_tool_schemas()
            template_vars["tools"] = list(self.tools.keys())

        # Add extra vars and kwargs
        template_vars |= self.extra_template_vars
        template_vars |= kwargs

        return Template(template, undefined=StrictUndefined).render(**template_vars)

    def execute_action(self, action: dict) -> dict:
        """Execute tool call(s) from parsed action.

        Supports parallel execution of multiple tool calls.

        Args:
            action: Parsed action dict containing "tool_calls" key.

        Returns:
            Combined result dict with outputs from all tools.

        Raises:
            Submitted: When answer tool is called or task is complete.
        """
        tool_calls: list[ToolCall] = action.get("tool_calls", [])

        # Handle no tool calls
        if not tool_calls:
            if action.get("action") is None:
                # No action at all - check if there's actual content
                content = action.get("content", "")
                # If content is empty, ask model to retry instead of submitting
                if not content.strip():
                    raise NonTerminatingException(
                        "Your response was empty. Please provide a tool call to continue working, "
                        "or use the answer tool to submit your final answer."
                    )
                # Non-empty content without tool call - treat as response
                raise Submitted(content)
            return {"output": "", "returncode": 0, "done": False}

        # Limit parallel calls
        calls_to_execute = tool_calls[: self.config.max_parallel_calls]

        # Execute tool calls (parallel if multiple)
        if len(calls_to_execute) == 1:
            results = [self._execute_single_call(calls_to_execute[0])]
        else:
            results = self._execute_parallel(calls_to_execute)

        # Check for answer tool (triggers completion)
        for call, result in zip(calls_to_execute, results):
            if call.name == self.config.answer_tool:
                # If the answer tool call was malformed, do not terminate the run.
                # Treat as a recoverable error so the agent can retry.
                if result.returncode != 0:
                    raise NonTerminatingException(result.output)

                # Extract answer from result
                answer = result.metadata.get("answer", result.output)

                # Check for empty answer
                if isinstance(answer, str) and not answer.strip():
                    raise NonTerminatingException(
                        "Error: Empty answer submitted. Call the answer tool with a non-empty `answer`."
                    )
                raise Submitted(answer)

        # Combine results
        combined_output = self._format_tool_responses(calls_to_execute, results)

        return {
            "output": combined_output,
            "returncode": 0 if all(r.returncode == 0 for r in results) else 1,
            "done": any(r.done for r in results),
            "tool_results": [r.to_dict() for r in results],
        }

    def _execute_single_call(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call.

        Args:
            call: ToolCall to execute.

        Returns:
            ToolResult from execution.
        """
        # Check if tool exists
        if call.name not in self.tools:
            available = list(self.tools.keys())
            return ToolResult(
                output=f"Error: Unknown tool '{call.name}'. Available tools: {available}",
                returncode=1,
                error=f"unknown_tool:{call.name}",
            )

        tool = self.tools[call.name]

        try:
            # Validate and execute (with caching if enabled for this tool)
            validated_args = tool.validate_args(call.arguments)
            result = tool.execute_with_cache(**validated_args)

            # Ensure we have a ToolResult
            if not isinstance(result, ToolResult):
                # Legacy tool returned dict
                result = ToolResult(
                    output=result.get("output", str(result)),
                    returncode=result.get("returncode", 0),
                    done=result.get("done", False),
                )

            return result

        except ValueError as e:
            # Validation error
            return ToolResult(
                output=f"Error: Invalid arguments for '{call.name}': {e}",
                returncode=1,
                error=f"validation_error:{e}",
            )
        except Exception as e:
            # Execution error
            logger.exception(f"Error executing tool '{call.name}'")
            return ToolResult(
                output=f"Error executing {call.name}: {type(e).__name__}: {e}",
                returncode=1,
                error=f"execution_error:{type(e).__name__}",
            )

    def _execute_parallel(self, calls: list[ToolCall]) -> list[ToolResult]:
        """Execute multiple tool calls in parallel.

        Args:
            calls: List of ToolCalls to execute.

        Returns:
            List of ToolResults in same order as input.
        """
        results: list[ToolResult | None] = [None] * len(calls)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_parallel_calls) as executor:
            future_to_idx = {executor.submit(self._execute_single_call, call): i for i, call in enumerate(calls)}

            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = ToolResult(
                        output=f"Error: {e}",
                        returncode=1,
                        error=str(e),
                    )

        # Ensure all results are populated
        return [r if r is not None else ToolResult(output="", returncode=1) for r in results]

    def _format_tool_responses(self, calls: list[ToolCall], results: list[ToolResult]) -> str:
        """Format tool responses for the model.

        Args:
            calls: List of executed tool calls.
            results: List of corresponding results.

        Returns:
            Formatted string with all tool responses.
        """
        parts = []

        for call, result in zip(calls, results):
            formatted = self.render_template(
                self.config.tool_response_template,
                tool_name=call.name,
                output=result.output,
                returncode=result.returncode,
                error=result.error,
            )
            parts.append(formatted)

        return "\n\n".join(parts)

    def add_tool(self, name: str, tool: BaseTool) -> None:
        """Add a tool at runtime.

        Args:
            name: Name to register the tool under.
            tool: Tool instance to add.
        """
        self.tools[name] = tool
        logger.debug(f"Added tool at runtime: {name}")

    def remove_tool(self, name: str) -> bool:
        """Remove a tool at runtime.

        Args:
            name: Name of tool to remove.

        Returns:
            True if tool was removed, False if not found.
        """
        if name in self.tools:
            del self.tools[name]
            logger.debug(f"Removed tool: {name}")
            return True
        return False

    def wrap_legacy_tool(self, legacy_tool: Any, name: str | None = None) -> None:
        """Wrap and add a legacy tool.

        Args:
            legacy_tool: Legacy tool with execute(action: dict) method.
            name: Optional name override.
        """
        adapted = LegacyToolAdapter(legacy_tool)
        tool_name = name or adapted.name
        self.tools[tool_name] = adapted
        logger.debug(f"Added wrapped legacy tool: {tool_name}")
