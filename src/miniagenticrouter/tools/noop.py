"""No-op tool for mini-SWE-agent.

This tool is used for pure conversation mode where no action execution is needed.
It simply returns an empty result without executing anything.
"""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class NoOpToolConfig:
    """Configuration for NoOpTool.

    Attributes:
        name: Tool name identifier.
    """

    name: str = "noop"


class NoOpTool:
    """No-operation tool for pure conversation mode.

    This tool does not execute any action. It is used when the agent
    operates in pure conversation mode without tool execution.

    Example:
        >>> tool = NoOpTool()
        >>> result = tool.execute({"action": None, "action_type": "none"})
        >>> result["output"]
        ''
        >>> result["returncode"]
        0
    """

    def __init__(self, *, config_class: type = NoOpToolConfig, **kwargs):
        """Initialize the tool.

        Args:
            config_class: Configuration dataclass to use.
            **kwargs: Configuration options passed to config_class.
        """
        self.config = config_class(**kwargs)

    @property
    def name(self) -> str:
        """Return tool name."""
        return self.config.name

    def execute(self, action: dict) -> dict:
        """Return empty result without executing anything.

        Args:
            action: Parsed action dict (ignored).

        Returns:
            Result dict containing:
            - "output": Empty string
            - "returncode": Always 0
            - "done": Always False
        """
        return {
            "output": "",
            "returncode": 0,
            "done": False,
        }

    def get_config(self) -> dict[str, Any]:
        """Return tool configuration for template rendering."""
        return asdict(self.config)
