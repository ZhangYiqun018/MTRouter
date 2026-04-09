"""Tool call parser for structured tool invocations.

This module provides:
- ToolCall: Dataclass representing a single tool call
- ToolCallParser: Parser for extracting tool calls from LLM responses

Supported formats:
1. JSON inside tool_call: <tool_call>{"name":"search","arguments":{...}}</tool_call>
2. XML inside tool_call: <tool_call><search><query>...</query></search></tool_call>
3. Markdown: ```tool\n{"name":"search","arguments":{...}}\n```
"""

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

from miniagenticrouter.parsers.regex import FormatError


@dataclass
class ToolCall:
    """Represents a single tool call extracted from LLM response.

    Attributes:
        name: Tool name to invoke.
        arguments: Dict of arguments to pass to the tool.
        raw: The raw matched string for debugging.
    """

    name: str
    arguments: dict[str, Any]
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "name": self.name,
            "arguments": self.arguments,
        }


@dataclass
class ToolCallParserConfig:
    """Configuration for ToolCallParser.

    Attributes:
        tool_call_regex: Primary regex pattern for tool calls (captures content inside).
        alternative_patterns: Additional patterns to try.
        allow_multiple_calls: Whether to allow multiple tool calls per response.
        allow_no_calls: Whether to allow responses without tool calls.
        strict_json: Whether to require strict JSON parsing.
        known_tools: List of known tool names for XML parsing.
        enable_xml_parsing: Whether to enable XML format parsing.
    """

    # Primary format: <tool_call>...</tool_call> (captures any content)
    tool_call_regex: str = r"<tool_call>\s*(.*?)\s*</tool_call>"

    # Alternative formats (JSON only)
    alternative_patterns: list[str] = field(
        default_factory=lambda: [
            r"```tool\s*\n(\{.*?\})\n```",  # ```tool {...} ```
            r"```json:tool\s*\n(\{.*?\})\n```",  # ```json:tool {...} ```
            r"```tool_call\s*\n(\{.*?\})\n```",  # ```tool_call {...} ```
        ]
    )

    allow_multiple_calls: bool = True
    allow_no_calls: bool = False
    strict_json: bool = False

    # XML parsing support
    known_tools: list[str] = field(default_factory=list)
    enable_xml_parsing: bool = True


class ToolCallParser:
    """Parser for structured tool calls with JSON arguments.

    Extracts tool calls from LLM responses in various formats.
    Supports multiple tool calls per response for parallel execution.

    Example:
        >>> parser = ToolCallParser()
        >>> response = {"content": '<tool_call>{"name":"search","arguments":{"query":"test"}}</tool_call>'}
        >>> result = parser.parse(response)
        >>> result["tool_calls"][0].name
        'search'
        >>> result["tool_calls"][0].arguments
        {'query': 'test'}
    """

    def __init__(self, *, config_class: type = ToolCallParserConfig, **kwargs: Any) -> None:
        """Initialize the parser.

        Args:
            config_class: Configuration dataclass to use.
            **kwargs: Configuration options passed to config_class.
        """
        self.config = config_class(**kwargs)
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile all regex patterns."""
        self._patterns: list[re.Pattern] = [
            re.compile(self.config.tool_call_regex, re.DOTALL),
        ]
        for pattern in self.config.alternative_patterns:
            self._patterns.append(re.compile(pattern, re.DOTALL))

    def parse(self, response: dict) -> dict:
        """Parse tool calls from model response.

        Args:
            response: Model response dict containing "content" key.

        Returns:
            Dict containing:
            - "tool_calls": List of ToolCall objects
            - "action": First tool call (for backwards compatibility)
            - "action_type": "tool_call"
            - All original response fields

        Raises:
            FormatError: When no valid tool calls found and allow_no_calls=False.
        """
        content = response.get("content", "")
        tool_calls = self._extract_tool_calls(content)

        # Validation
        if not tool_calls and not self.config.allow_no_calls:
            raise FormatError(
                "No valid tool calls found. Please use the format:\n"
                '<tool_call>{"name":"tool_name","arguments":{...}}</tool_call>'
            )

        if len(tool_calls) > 1 and not self.config.allow_multiple_calls:
            raise FormatError(
                f"Expected at most 1 tool call, found {len(tool_calls)}. "
                "Please use only one tool call per response."
            )

        # Build result
        result = {
            "tool_calls": tool_calls,
            "action_type": "tool_call",
            **response,
        }

        # Backwards compatibility: set action to first call
        if tool_calls:
            result["action"] = tool_calls[0]
        else:
            result["action"] = None

        return result

    def _extract_tool_calls(self, content: str) -> list[ToolCall]:
        """Extract all tool calls from content.

        Supports both JSON and XML formats inside <tool_call> tags.
        Priority: JSON first, then XML fallback.

        Args:
            content: LLM response content.

        Returns:
            List of parsed ToolCall objects.
        """
        tool_calls: list[ToolCall] = []
        seen_raw: set[str] = set()  # Avoid duplicates

        for pattern in self._patterns:
            matches = pattern.findall(content)
            for match in matches:
                if match in seen_raw:
                    continue
                seen_raw.add(match)

                # 1. Try JSON parsing first
                tool_call = self._parse_json_tool_call(match)
                if tool_call is not None:
                    tool_calls.append(tool_call)
                    continue

                # 2. Try XML parsing if enabled
                if self.config.enable_xml_parsing and self.config.known_tools:
                    tool_call = self._parse_xml_tool_call(match)
                    if tool_call is not None:
                        tool_calls.append(tool_call)

        return tool_calls

    def _parse_json_tool_call(self, json_str: str) -> ToolCall | None:
        """Parse a JSON string into a ToolCall.

        Args:
            json_str: JSON string to parse.

        Returns:
            ToolCall if valid, None otherwise.
        """
        try:
            # Clean up the JSON string
            json_str = json_str.strip()

            # Handle potential nested JSON (double-encoded)
            data = json.loads(json_str)

            # Validate structure
            if not isinstance(data, dict):
                return None

            name = data.get("name")
            if not name or not isinstance(name, str):
                return None

            arguments = data.get("arguments", {})
            if not isinstance(arguments, dict):
                # Try to parse arguments if it's a string
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"value": arguments}
                else:
                    arguments = {}

            return ToolCall(
                name=name,
                arguments=arguments,
                raw=json_str,
            )

        except json.JSONDecodeError:
            if self.config.strict_json:
                return None
            # Try to fix common JSON issues
            return self._parse_lenient_json(json_str)

    def _parse_lenient_json(self, json_str: str) -> ToolCall | None:
        """Attempt to parse malformed JSON with common fixes.

        Args:
            json_str: Potentially malformed JSON string.

        Returns:
            ToolCall if parseable, None otherwise.
        """
        # Fix common issues
        fixed = json_str

        # Fix single quotes -> double quotes (careful with values)
        fixed = re.sub(r"'(\w+)':", r'"\1":', fixed)

        # Fix trailing commas
        fixed = re.sub(r",\s*}", "}", fixed)
        fixed = re.sub(r",\s*]", "]", fixed)

        # Fix unquoted keys
        fixed = re.sub(r"(\{|,)\s*(\w+)\s*:", r'\1"\2":', fixed)

        try:
            data = json.loads(fixed)
            name = data.get("name")
            if name:
                return ToolCall(
                    name=name,
                    arguments=data.get("arguments", {}),
                    raw=json_str,
                )
        except json.JSONDecodeError:
            pass

        return None

    def _parse_xml_tool_call(self, content: str) -> ToolCall | None:
        """Parse XML tool call inside <tool_call> block.

        Looks for pattern like <tool_name><param>value</param></tool_name>
        where tool_name is in known_tools.

        Handles nested tags (e.g., <answer><answer>value</answer></answer>)
        by finding the LAST closing tag to support nested same-name elements.

        Args:
            content: Content inside <tool_call> tags.

        Returns:
            ToolCall if valid XML tool found, None otherwise.
        """
        for tool_name in self.config.known_tools:
            # Find opening tag
            open_tag = f"<{tool_name}>"
            close_tag = f"</{tool_name}>"

            open_idx = content.find(open_tag)
            if open_idx == -1:
                continue

            # Find the LAST closing tag (to handle nested same-name tags)
            close_idx = content.rfind(close_tag)
            if close_idx == -1 or close_idx <= open_idx:
                continue

            # Extract inner content
            inner_start = open_idx + len(open_tag)
            inner_content = content[inner_start:close_idx].strip()

            try:
                arguments = self._parse_xml_arguments(inner_content)
                return ToolCall(
                    name=tool_name,
                    arguments=arguments,
                    raw=f"<tool_call>{content}</tool_call>",
                )
            except Exception:
                continue
        return None

    def _parse_xml_arguments(self, xml_content: str) -> dict[str, Any]:
        """Parse XML content into arguments dict.

        Parses <param_name>value</param_name> into {"param_name": value}.
        Attempts to parse values as JSON for numbers, booleans, arrays.

        Args:
            xml_content: XML content with parameter tags.

        Returns:
            Dict of argument name to value.
        """
        # Wrap in root element for parsing
        xml_str = f"<root>{xml_content}</root>"
        root = ET.fromstring(xml_str)
        arguments: dict[str, Any] = {}

        for child in root:
            tag = child.tag
            # Get text content, handling multiline and whitespace
            text = child.text if child.text else ""

            # Try to parse as JSON value (numbers, booleans, arrays, objects)
            try:
                arguments[tag] = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                # Keep as string, strip only leading/trailing whitespace
                arguments[tag] = text.strip() if text else ""

        return arguments

    def get_config(self) -> dict[str, Any]:
        """Return parser configuration for template rendering.

        Returns:
            Config dict with parser type and settings.
        """
        return {
            "parser_type": "toolcall",
            "parser_class": "toolcall",
            "tool_call_regex": self.config.tool_call_regex,
            "allow_multiple_calls": self.config.allow_multiple_calls,
            "allow_no_calls": self.config.allow_no_calls,
            "known_tools": self.config.known_tools,
            "enable_xml_parsing": self.config.enable_xml_parsing,
        }
