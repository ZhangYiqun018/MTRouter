"""Bash tool for mini-SWE-agent.

This tool executes bash commands in an Environment and returns observations.
It is designed to be backward-compatible with the original DefaultAgent's
execute_action() behavior.
"""

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from miniagenticrouter import Environment


@dataclass
class BashToolConfig:
    """Configuration for BashTool.

    Attributes:
        name: Tool name identifier.
        cwd: Working directory for command execution (empty = use environment default).
    """

    name: str = "bash"
    cwd: str = ""


class BashTool:
    """Bash command execution tool.

    This tool wraps an Environment to execute bash commands extracted from
    model responses. It is the default tool for SWE-bench tasks.

    Example:
        >>> from miniagenticrouter.environments.local import LocalEnvironment
        >>> tool = BashTool(env=LocalEnvironment())
        >>> result = tool.execute({"action": "echo 'hello'", "action_type": "bash"})
        >>> result["output"]
        'hello\\n'
        >>> result["returncode"]
        0
    """

    def __init__(
        self,
        env: "Environment",
        *,
        config_class: type = BashToolConfig,
        **kwargs,
    ):
        """Initialize the tool.

        Args:
            env: Environment instance for command execution.
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
        """Execute a bash command and return the result.

        Args:
            action: Parsed action dict containing:
                - "action": Bash command to execute
                - "action_type": Should be "bash"

        Returns:
            Result dict containing:
            - "output": Command output (stdout/stderr combined)
            - "returncode": Exit code (0 for success)
            - "done": Whether execution signals task completion
        """
        command = action.get("action")

        if command is None:
            # No action to execute
            return {
                "output": "",
                "returncode": 0,
                "done": False,
            }

        # Execute via environment
        result = self.env.execute(command, cwd=self.config.cwd)

        return {
            "output": result.get("output", ""),
            "returncode": result.get("returncode", 0),
            "done": result.get("done", False),
        }

    def get_config(self) -> dict[str, Any]:
        """Return tool configuration for template rendering."""
        return asdict(self.config)
