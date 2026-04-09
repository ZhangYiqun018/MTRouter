"""Regex-based action parser for mini-SWE-agent.

This parser extracts actions from model responses using regular expressions.
It is compatible with the original DefaultAgent's parse_action() behavior.
"""

import re
from dataclasses import asdict, dataclass, field
from typing import Any


class FormatError(Exception):
    """Raised when the model's output is not in the expected format.

    This is a non-terminating exception - the agent can recover by
    adding the error message to the conversation and continuing.
    """

    pass


@dataclass
class RegexActionParserConfig:
    """Configuration for RegexActionParser.

    Attributes:
        action_regex: Regular expression pattern to extract actions.
                      Default matches ```bash ... ``` code blocks.
        action_type: Type identifier for extracted actions (e.g., "bash", "text").
        require_single_action: If True, raises FormatError when != 1 action found.
        allow_no_action: If True, returns action=None instead of raising error
                         when no action is found.
        format_error_template: Template for error message when format is invalid.
    """

    action_regex: str = r"```bash\s*\n(.*?)\n```"
    action_type: str = "bash"
    require_single_action: bool = True
    allow_no_action: bool = False
    format_error_template: str = (
        "Please always provide EXACTLY ONE action in triple backticks, found {count} actions."
    )


class RegexActionParser:
    """Regex-based action parser.

    This parser uses regular expressions to extract actions from model responses.
    It is designed to be backward-compatible with the original DefaultAgent's
    parse_action() method.

    Example:
        >>> parser = RegexActionParser(action_regex=r"```bash\\s*\\n(.*?)\\n```")
        >>> response = {"content": "Let me list files\\n\\n```bash\\nls -la\\n```"}
        >>> result = parser.parse(response)
        >>> result["action"]
        'ls -la'
        >>> result["action_type"]
        'bash'
    """

    def __init__(self, *, config_class: type = RegexActionParserConfig, **kwargs):
        """Initialize the parser.

        Args:
            config_class: Configuration dataclass to use.
            **kwargs: Configuration options passed to config_class.
        """
        self.config = config_class(**kwargs)
        self._compiled_regex = re.compile(self.config.action_regex, re.DOTALL)

    def parse(self, response: dict) -> dict:
        """Parse an action from a model response.

        Args:
            response: Model response dict containing "content" key.

        Returns:
            Dict containing:
            - "action": Extracted action string (or None if allow_no_action=True
                        and no action found)
            - "action_type": Type of action (from config)
            - All original response fields

        Raises:
            FormatError: When the response doesn't match expected format
                         (e.g., zero or multiple actions when require_single_action=True).
        """
        content = response.get("content", "")
        actions = self._compiled_regex.findall(content)

        if len(actions) == 1:
            return {
                "action": actions[0].strip(),
                "action_type": self.config.action_type,
                **response,
            }

        if len(actions) == 0 and self.config.allow_no_action:
            return {
                "action": None,
                "action_type": "none",
                **response,
            }

        if self.config.require_single_action:
            error_msg = self.config.format_error_template.format(
                count=len(actions),
                actions=actions,
            )
            raise FormatError(error_msg)

        # Multiple actions found and require_single_action=False
        # Return the first action (or could be extended to return all)
        return {
            "action": actions[0].strip() if actions else None,
            "action_type": self.config.action_type if actions else "none",
            "all_actions": [a.strip() for a in actions],
            **response,
        }

    def get_config(self) -> dict[str, Any]:
        """Return parser configuration for template rendering."""
        return asdict(self.config)
