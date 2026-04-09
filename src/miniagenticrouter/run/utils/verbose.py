"""Unified verbose output utilities for benchmark runners."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from rich.console import Console
from rich.panel import Panel

if TYPE_CHECKING:
    from miniagenticrouter.agents.flexible import FlexibleAgent

# Shared color configuration for different message roles
ROLE_COLORS = {
    "system": "bright_blue",
    "user": "green",
    "assistant": "yellow",
    "tool": "cyan",
}


@dataclass
class VerboseConfig:
    """Configuration for VerboseRunner.

    Attributes:
        benchmark_name: Name displayed in the header (e.g., "ScienceWorld", "HLE")
        role_colors: Color mapping for different message roles
        observation_role: Role name for observation/tool response messages ("user" or "tool")
        show_model_name: Whether to display model name in assistant message titles
        max_content_len: Maximum content length before truncation
        on_finish: Optional callback for custom finish message, receives (env, exception) and returns string
    """

    benchmark_name: str = "Task"
    role_colors: dict = field(default_factory=lambda: ROLE_COLORS.copy())
    observation_role: str = "user"  # "user" or "tool"
    show_model_name: bool = True
    max_content_len: int = 500
    on_finish: Callable[[Any, Exception], str | None] | None = None


class VerboseRunner:
    """Unified verbose runner for all benchmarks.

    Provides consistent verbose output across different benchmark runners,
    with configurable behavior for benchmark-specific differences.

    Example:
        runner = VerboseRunner(VerboseConfig(
            benchmark_name="ScienceWorld",
            on_finish=lambda env, e: f"Score: {env.get_score():.2f}" if env else None,
        ))
        exit_status, result = runner.run(agent, task_description, env=env)
    """

    def __init__(self, config: VerboseConfig | None = None):
        """Initialize VerboseRunner with optional configuration.

        Args:
            config: VerboseConfig instance, uses defaults if None
        """
        self.config = config or VerboseConfig()
        self.console = Console()

    def truncate_content(self, content: str) -> str:
        """Truncate long content showing head and tail.

        Args:
            content: The content string to truncate

        Returns:
            Truncated content with "... (N chars omitted) ..." in the middle
        """
        max_len = self.config.max_content_len
        if len(content) <= max_len:
            return content
        head_len = max_len // 2
        tail_len = max_len // 2
        return f"{content[:head_len]}\n\n... ({len(content) - max_len} chars omitted) ...\n\n{content[-tail_len:]}"

    def print_message(
        self,
        role: str,
        content: str,
        step: int | None = None,
        model_name: str | None = None,
    ):
        """Print a message with color coding by role.

        Args:
            role: Message role (system, user, assistant, tool)
            content: Message content
            step: Optional step number
            model_name: Optional model name to display (for router mode)
        """
        color = self.config.role_colors.get(role, "white")

        # Build title with optional step and model info
        parts = [f"[bold]{role.upper()}[/bold]"]
        if step is not None:
            parts.append(f"Step {step}")
        if model_name is not None and self.config.show_model_name:
            parts.append(f"Model: {model_name}")

        title = " | ".join(parts) if len(parts) > 1 else parts[0]
        display_content = self.truncate_content(content)
        self.console.print(Panel(display_content, title=title, border_style=color))

    def get_model_name(self, agent: FlexibleAgent, response: dict) -> str | None:
        """Extract model name from router response or agent config.

        Args:
            agent: The agent instance
            response: Response dict from model query

        Returns:
            Model name string or None if not available
        """
        # Router responses include model_name directly
        model_name = response.get("model_name")
        # Fall back to agent's model config
        if model_name is None and hasattr(agent.model, "config"):
            model_name = getattr(agent.model.config, "model_name", None)
        return model_name

    def run(self, agent: FlexibleAgent, task: str, **kwargs) -> tuple[str, str]:
        """Run agent with verbose output, printing each step.

        Args:
            agent: The agent instance (FlexibleAgent or MultiToolAgent)
            task: Task description string
            **kwargs: Additional arguments, notably 'env' for environment access

        Returns:
            Tuple of (exit_status, result)
        """
        from miniagenticrouter.agents.flexible import (
            NonTerminatingException,
            TerminatingException,
        )

        # Initialize messages like agent.run() does
        agent.extra_template_vars["task"] = task
        agent.messages = []
        agent.add_message("system", agent.render_template(agent.config.system_template))
        agent.add_message("user", agent.render_template(agent.config.instance_template))

        # Print header and initial messages
        self.console.print(
            f"\n[bold cyan]═══ Starting {self.config.benchmark_name} Task ═══[/bold cyan]\n"
        )
        self.print_message("system", agent.messages[0]["content"])
        self.print_message("user", agent.messages[1]["content"])

        step = 0
        while True:
            step += 1
            self.console.print(f"\n[bold magenta]─── Step {step} ───[/bold magenta]")

            try:
                # Query model
                response = agent.query()
                model_name = self.get_model_name(agent, response)
                self.print_message(
                    "assistant", response["content"], step, model_name=model_name
                )

                # Get observation
                agent.get_observation(response)

                # Print observation (the last user message added by get_observation)
                if agent.messages and agent.messages[-1]["role"] == "user":
                    self.print_message(
                        self.config.observation_role,
                        agent.messages[-1]["content"],
                        step,
                    )

            except NonTerminatingException as e:
                # Format errors, timeouts etc - add to messages and continue
                agent.add_message("user", str(e))
                self.print_message("user", str(e), step)

            except TerminatingException as e:
                # Task finished - show summary
                self.console.print(
                    f"\n[bold green]═══ Task Finished ═══[/bold green]"
                )
                self.console.print(f"[green]Exit status: {type(e).__name__}[/green]")

                # Call custom finish handler for benchmark-specific info
                if self.config.on_finish:
                    extra_info = self.config.on_finish(kwargs.get("env"), e)
                    if extra_info:
                        self.console.print(extra_info)

                return type(e).__name__, str(e)
