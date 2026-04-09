"""No-action parser for mini-SWE-agent.

This parser is used for pure conversation mode where no action extraction
is needed. It simply passes through the response without extracting any action.
"""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class NoActionParserConfig:
    """Configuration for NoActionParser.

    Attributes:
        action_type: Type identifier, always "none" for this parser.
    """

    action_type: str = "none"


class NoActionParser:
    """No-action parser for pure conversation mode.

    This parser does not extract any action from the model response.
    It is used when the agent operates in pure conversation mode
    without tool execution.

    Example:
        >>> parser = NoActionParser()
        >>> response = {"content": "Hello, how can I help you?"}
        >>> result = parser.parse(response)
        >>> result["action"]
        None
        >>> result["action_type"]
        'none'
    """

    def __init__(self, *, config_class: type = NoActionParserConfig, **kwargs):
        """Initialize the parser.

        Args:
            config_class: Configuration dataclass to use.
            **kwargs: Configuration options passed to config_class.
        """
        self.config = config_class(**kwargs)

    def parse(self, response: dict) -> dict:
        """Pass through response without extracting action.

        Args:
            response: Model response dict.

        Returns:
            Dict containing:
            - "action": Always None
            - "action_type": Always "none"
            - All original response fields
        """
        return {
            "action": None,
            "action_type": self.config.action_type,
            **response,
        }

    def get_config(self) -> dict[str, Any]:
        """Return parser configuration for template rendering."""
        return asdict(self.config)
