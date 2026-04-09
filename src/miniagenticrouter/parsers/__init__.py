"""Action parsers for mini-SWE-agent.

This module provides abstractions for parsing actions from model responses.
"""

import copy
import importlib
from dataclasses import asdict
from typing import Any, Protocol


class ActionParser(Protocol):
    """Protocol for action parsers.

    Action parsers are responsible for extracting actions from model responses.
    Different parsers can support different formats (regex, JSON, none, etc.).
    """

    def parse(self, response: dict) -> dict:
        """Parse an action from a model response.

        Args:
            response: Model response dict containing at least "content" key.

        Returns:
            Parsed action dict containing:
            - "action": The extracted action string (or None if no action)
            - "action_type": Type of action (e.g., "bash", "text", "none")
            - All original response fields

        Raises:
            FormatError: When the response doesn't match expected format.
        """
        ...

    def get_config(self) -> dict[str, Any]:
        """Return parser configuration for template rendering."""
        ...


_PARSER_MAPPING = {
    "regex": "miniagenticrouter.parsers.regex.RegexActionParser",
    "noaction": "miniagenticrouter.parsers.noaction.NoActionParser",
    "toolcall": "miniagenticrouter.parsers.toolcall.ToolCallParser",
}


def get_parser_class(spec: str) -> type[ActionParser]:
    """Get a parser class by name or full path.

    Args:
        spec: Either a short name (e.g., "regex") or full import path
              (e.g., "miniagenticrouter.parsers.regex.RegexActionParser")

    Returns:
        The parser class.

    Raises:
        ValueError: If the parser is not found.
    """
    full_path = _PARSER_MAPPING.get(spec, spec)
    try:
        module_name, class_name = full_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ValueError, ImportError, AttributeError):
        msg = f"Unknown parser type: {spec} (resolved to {full_path}, available: {list(_PARSER_MAPPING.keys())})"
        raise ValueError(msg)


def get_parser(config: dict | None = None, *, default_type: str = "regex") -> ActionParser:
    """Create a parser instance from configuration.

    Args:
        config: Parser configuration dict. Should contain "parser_class" key
                to specify the parser type.
        default_type: Default parser type if not specified in config.

    Returns:
        An ActionParser instance.
    """
    config = copy.deepcopy(config) if config else {}
    parser_class_spec = config.pop("parser_class", default_type)
    parser_class = get_parser_class(parser_class_spec)
    return parser_class(**config)
