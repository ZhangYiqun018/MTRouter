"""Python execution tool with sandbox support.

This tool executes Python code either locally or via a remote sandbox service.
"""

import logging
import os
import random
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from miniagenticrouter.tools.base import BaseTool, ToolResult, ToolSchema
from miniagenticrouter.tools.registry import register_tool

if TYPE_CHECKING:
    from miniagenticrouter import Environment

logger = logging.getLogger(__name__)


@dataclass
class PythonToolConfig:
    """Configuration for PythonTool."""

    timeout: int = 60
    max_output_length: int = 50000
    sandbox_endpoints: list[str] = field(default_factory=list)
    max_retries: int = 3
    use_env: bool = True  # Use Environment if available


@register_tool("python")
class PythonTool(BaseTool):
    """Execute Python code in a sandbox environment.

    Supports multiple execution modes:
    1. Remote sandbox (if sandbox_endpoints configured)
    2. Environment execution (if env provided)
    3. Local subprocess (fallback)

    Example:
        >>> tool = PythonTool(timeout=30)
        >>> result = tool.execute(code="print(2 ** 100)")
        >>> "1267650600228229401496703205376" in result.output
        True
    """

    name = "python"
    description = "Execute Python code and return the output"

    def __init__(
        self,
        env: "Environment | None" = None,
        timeout: int = 60,
        max_output_length: int = 50000,
        sandbox_endpoints: list[str] | None = None,
        max_retries: int = 3,
        **kwargs,
    ) -> None:
        """Initialize the Python tool.

        Args:
            env: Optional environment for execution.
            timeout: Execution timeout in seconds.
            max_output_length: Maximum output length to return.
            sandbox_endpoints: List of sandbox service URLs.
            max_retries: Number of retries for sandbox execution.
        """
        self.env = env

        # Get sandbox endpoints from config or environment
        endpoints = sandbox_endpoints or []
        if not endpoints:
            env_endpoints = os.getenv("SANDBOX_ENDPOINTS", "")
            if env_endpoints:
                endpoints = [e.strip() for e in env_endpoints.split(",") if e.strip()]

        self.config = PythonToolConfig(
            timeout=timeout,
            max_output_length=max_output_length,
            sandbox_endpoints=endpoints,
            max_retries=max_retries,
        )

    def get_schema(self) -> ToolSchema:
        """Return the tool schema."""
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute",
                    },
                },
            },
            required=["code"],
            examples=[
                {"code": "print('Hello, World!')"},
                {"code": "import math\nresult = math.sqrt(16)\nprint(f'Square root of 16 is {result}')"},
                {"code": "# Calculate factorial\nimport math\nprint(math.factorial(10))"},
            ],
        )

    def execute(self, code: str, **kwargs) -> ToolResult:
        """Execute Python code.

        Args:
            code: Python code to execute.

        Returns:
            ToolResult with execution output.
        """
        # Try sandbox first if configured
        if self.config.sandbox_endpoints:
            result = self._execute_sandbox(code)
            if result is not None:
                return result

        # Try environment execution
        if self.env is not None and self.config.use_env:
            return self._execute_via_env(code)

        # Fall back to local execution
        return self._execute_local(code)

    def _execute_sandbox(self, code: str) -> ToolResult | None:
        """Execute code via remote sandbox service.

        Args:
            code: Python code to execute.

        Returns:
            ToolResult if successful, None if all retries failed.
        """
        import requests

        endpoints = self.config.sandbox_endpoints.copy()
        random.shuffle(endpoints)

        for attempt in range(self.config.max_retries):
            if not endpoints:
                break

            endpoint = endpoints[attempt % len(endpoints)]

            try:
                response = requests.post(
                    f"{endpoint}/execute",
                    json={
                        "code": code,
                        "language": "python",
                        "timeout": self.config.timeout,
                    },
                    timeout=self.config.timeout + 10,
                )
                response.raise_for_status()
                data = response.json()

                output = data.get("stdout", "") + data.get("stderr", "")
                if data.get("timeout"):
                    output += "\n[Execution timed out]"

                return ToolResult(
                    output=output[: self.config.max_output_length],
                    returncode=data.get("returncode", 0),
                    metadata={"endpoint": endpoint, "sandbox": True},
                )

            except requests.RequestException as e:
                logger.warning(f"Sandbox request failed (attempt {attempt + 1}): {e}")
                continue

        return None

    def _execute_via_env(self, code: str) -> ToolResult:
        """Execute code via the configured Environment.

        Args:
            code: Python code to execute.

        Returns:
            ToolResult with execution output.
        """
        # Escape the code for shell
        import shlex

        escaped_code = shlex.quote(code)

        # Use python3 -c with the code
        command = f"python3 -c {escaped_code}"

        try:
            result = self.env.execute(command)
            return ToolResult(
                output=result.get("output", "")[: self.config.max_output_length],
                returncode=result.get("returncode", 0),
                metadata={"via_env": True},
            )
        except Exception as e:
            return ToolResult(
                output=f"Error executing via environment: {e}",
                returncode=1,
                error=str(e),
            )

    def _execute_local(self, code: str) -> ToolResult:
        """Execute code locally via subprocess.

        Args:
            code: Python code to execute.

        Returns:
            ToolResult with execution output.
        """
        try:
            result = subprocess.run(
                ["python3", "-c", code],
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
            )

            output = result.stdout
            if result.stderr:
                output += "\nSTDERR:\n" + result.stderr

            return ToolResult(
                output=output[: self.config.max_output_length],
                returncode=result.returncode,
                metadata={"local": True},
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                output=f"Execution timed out after {self.config.timeout} seconds",
                returncode=1,
                error="timeout",
            )
        except Exception as e:
            return ToolResult(
                output=f"Error executing Python code: {e}",
                returncode=1,
                error=str(e),
            )

    def get_config(self) -> dict[str, Any]:
        """Return tool configuration."""
        return {
            "name": self.name,
            "description": self.description,
            "timeout": self.config.timeout,
            "has_sandbox": bool(self.config.sandbox_endpoints),
            "has_env": self.env is not None,
        }
