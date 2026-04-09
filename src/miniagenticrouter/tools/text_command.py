"""Text command tool for mini-SWE-agent.

This tool is designed for text-based simulation environments like ScienceWorld
where commands are plain text strings sent to a simulator.
"""

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Protocol


class TextEnvironment(Protocol):
    """Protocol for text-based environments.

    These environments receive text commands and return text observations.
    Examples include ScienceWorld, text adventure games, etc.
    """

    def step(self, command: str) -> dict:
        """Execute a command and return the result.

        Args:
            command: Text command to execute.

        Returns:
            Dict containing at least:
            - "observation": Text observation from the environment
            - "done": Whether the episode is finished
            - "reward": (optional) Reward signal
        """
        ...


@dataclass
class TextCommandToolConfig:
    """Configuration for TextCommandTool.

    Attributes:
        name: Tool name identifier.
    """

    name: str = "text_command"


class TextCommandTool:
    """Text command tool for simulation environments.

    This tool sends text commands to a TextEnvironment (like ScienceWorld)
    and returns observations.

    Example:
        >>> # Assuming a ScienceWorld environment
        >>> tool = TextCommandTool(env=scienceworld_env)
        >>> result = tool.execute({"action": "look around", "action_type": "text"})
        >>> result["output"]
        'You are in a kitchen...'
    """

    def __init__(
        self,
        env: TextEnvironment,
        *,
        config_class: type = TextCommandToolConfig,
        **kwargs,
    ):
        """Initialize the tool.

        Args:
            env: Text environment instance for command execution.
            config_class: Configuration dataclass to use.
            **kwargs: Configuration options passed to config_class.
        """
        self.env = env
        self.config = config_class(**kwargs)

    @property
    def name(self) -> str:
        """Return tool name."""
        return self.config.name

    def execute(self, action: dict) -> dict:
        """Execute a text command and return the result.

        Args:
            action: Parsed action dict containing:
                - "action": Text command to execute
                - "action_type": Should be "text"

        Returns:
            Result dict containing:
            - "output": Observation from the environment
            - "returncode": 0 for success
            - "done": Whether the episode is finished
            - "reward": Reward signal (if available)
        """
        command = action.get("action")

        if command is None:
            return {
                "output": "",
                "returncode": 0,
                "done": False,
            }

        # Execute via text environment
        result = self.env.step(command)

        return {
            "output": result.get("observation", ""),
            "returncode": 0,
            "done": result.get("done", False),
            "reward": result.get("reward", 0),
        }

    def get_config(self) -> dict[str, Any]:
        """Return tool configuration for template rendering."""
        return asdict(self.config)
