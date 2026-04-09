"""File reading tool for accessing file contents.

This tool reads file contents with optional line range selection.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from miniagenticrouter.tools.base import BaseTool, ToolResult, ToolSchema
from miniagenticrouter.tools.registry import register_tool

if TYPE_CHECKING:
    from miniagenticrouter import Environment


@dataclass
class FileReadConfig:
    """Configuration for FileReadTool."""

    max_lines: int = 500
    max_chars: int = 50000


@register_tool("file_read")
class FileReadTool(BaseTool):
    """Read file contents with optional line range.

    Can execute via Environment or local file system.

    Example:
        >>> tool = FileReadTool()
        >>> result = tool.execute(path="/etc/hostname")
        >>> result.returncode
        0
    """

    name = "file_read"
    description = "Read the contents of a file"

    def __init__(
        self,
        env: "Environment | None" = None,
        max_lines: int = 500,
        max_chars: int = 50000,
        **kwargs,
    ) -> None:
        """Initialize the file read tool.

        Args:
            env: Optional environment for file access.
            max_lines: Maximum lines to read.
            max_chars: Maximum characters to return.
        """
        self.env = env
        self.config = FileReadConfig(max_lines=max_lines, max_chars=max_chars)

    def get_schema(self) -> ToolSchema:
        """Return the tool schema."""
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Starting line number (1-indexed, default: 1)",
                        "minimum": 1,
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Ending line number (inclusive, default: read all)",
                        "minimum": 1,
                    },
                },
            },
            required=["path"],
            examples=[
                {"path": "/etc/passwd"},
                {"path": "src/main.py", "start_line": 1, "end_line": 50},
                {"path": "README.md"},
            ],
        )

    def execute(
        self,
        path: str,
        start_line: int = 1,
        end_line: int | None = None,
        **kwargs,
    ) -> ToolResult:
        """Read file contents.

        Args:
            path: File path to read.
            start_line: Starting line number (1-indexed).
            end_line: Ending line number (inclusive).

        Returns:
            ToolResult with file contents.
        """
        if self.env is not None:
            return self._read_via_env(path, start_line, end_line)
        else:
            return self._read_local(path, start_line, end_line)

    def _read_via_env(
        self,
        path: str,
        start_line: int,
        end_line: int | None,
    ) -> ToolResult:
        """Read file via environment.

        Args:
            path: File path.
            start_line: Starting line.
            end_line: Ending line.

        Returns:
            ToolResult with file contents.
        """
        if end_line is not None:
            # Use sed to extract line range
            cmd = f"sed -n '{start_line},{end_line}p' {path}"
        else:
            # Use head to limit lines
            cmd = f"head -n {self.config.max_lines} {path}"
            if start_line > 1:
                cmd = f"tail -n +{start_line} {path} | head -n {self.config.max_lines}"

        try:
            result = self.env.execute(cmd)
            content = result.get("output", "")

            return ToolResult(
                output=content[: self.config.max_chars],
                returncode=result.get("returncode", 0),
                metadata={
                    "path": path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "via_env": True,
                },
            )
        except Exception as e:
            return ToolResult(
                output=f"Error reading file: {e}",
                returncode=1,
                error=str(e),
            )

    def _read_local(
        self,
        path: str,
        start_line: int,
        end_line: int | None,
    ) -> ToolResult:
        """Read file locally.

        Args:
            path: File path.
            start_line: Starting line.
            end_line: Ending line.

        Returns:
            ToolResult with file contents.
        """
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            # Apply line range
            start_idx = start_line - 1
            if end_line is not None:
                lines = lines[start_idx:end_line]
            else:
                lines = lines[start_idx : start_idx + self.config.max_lines]

            content = "".join(lines)

            return ToolResult(
                output=content[: self.config.max_chars],
                returncode=0,
                metadata={
                    "path": path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "num_lines": len(lines),
                    "local": True,
                },
            )

        except FileNotFoundError:
            return ToolResult(
                output=f"File not found: {path}",
                returncode=1,
                error="file_not_found",
            )
        except PermissionError:
            return ToolResult(
                output=f"Permission denied: {path}",
                returncode=1,
                error="permission_denied",
            )
        except Exception as e:
            return ToolResult(
                output=f"Error reading file: {e}",
                returncode=1,
                error=str(e),
            )

    def get_config(self) -> dict[str, Any]:
        """Return tool configuration."""
        return {
            "name": self.name,
            "description": self.description,
            "max_lines": self.config.max_lines,
            "max_chars": self.config.max_chars,
            "has_env": self.env is not None,
        }
