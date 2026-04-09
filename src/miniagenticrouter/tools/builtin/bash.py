"""Bash tool for executing shell commands.

This tool executes bash commands in the configured environment.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from miniagenticrouter.tools.base import BaseTool, ToolResult, ToolSchema
from miniagenticrouter.tools.registry import register_tool

if TYPE_CHECKING:
    from miniagenticrouter import Environment


@dataclass
class BashToolConfig:
    """Configuration for BashTool."""

    cwd: str = ""
    timeout: int = 60


@register_tool("bash")
class BashTool(BaseTool):
    """Execute bash commands in the environment.

    Requires an Environment instance for command execution.

    Example:
        >>> from miniagenticrouter.environments.local import LocalEnvironment
        >>> tool = BashTool(env=LocalEnvironment())
        >>> result = tool.execute(command="echo 'Hello, World!'")
        >>> result.output
        'Hello, World!\\n'
    """

    name = "bash"
    description = "Execute shell commands"

    def __init__(
        self,
        env: "Environment | None" = None,
        cwd: str = "",
        timeout: int = 60,
        **kwargs,
    ) -> None:
        """Initialize the bash tool.

        Args:
            env: Environment for command execution.
            cwd: Default working directory.
            timeout: Command timeout in seconds.
        """
        self.env = env
        self.config = BashToolConfig(cwd=cwd, timeout=timeout)

    def get_schema(self) -> ToolSchema:
        """Return the tool schema."""
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Bash command to execute",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory (optional, uses default if not specified)",
                    },
                },
            },
            required=["command"],
            examples=[
                {"command": "ls -la"},
                {"command": "cat file.txt", "cwd": "/tmp"},
                {"command": "grep -r 'pattern' ."},
            ],
        )

    def execute(self, command: str, cwd: str = "", **kwargs) -> ToolResult:
        """Execute a bash command.

        Args:
            command: Bash command to execute.
            cwd: Working directory override.

        Returns:
            ToolResult with command output.
        """
        if self.env is None:
            return ToolResult(
                output="Error: No environment configured for bash execution",
                returncode=1,
                error="no_environment",
            )

        try:
            result = self.env.execute(command, cwd=cwd or self.config.cwd)

            return ToolResult(
                output=result.get("output", ""),
                returncode=result.get("returncode", 0),
                done=result.get("done", False),
                metadata={"command": command, "cwd": cwd or self.config.cwd},
            )
        except Exception as e:
            return ToolResult(
                output=f"Error executing command: {e}",
                returncode=1,
                error=str(e),
            )

    def get_config(self) -> dict[str, Any]:
        """Return tool configuration."""
        return {
            "name": self.name,
            "description": self.description,
            "cwd": self.config.cwd,
            "timeout": self.config.timeout,
        }
