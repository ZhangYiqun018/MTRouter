"""Flexible agent that supports multiple action modes.

This agent extends DefaultAgent with pluggable parsers and tools,
enabling support for different action formats beyond bash commands.
"""

import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from jinja2 import StrictUndefined, Template

from miniagenticrouter import Environment, Model
from miniagenticrouter.parsers import ActionParser, get_parser
from miniagenticrouter.parsers.regex import FormatError
from miniagenticrouter.tools import Tool, get_tool
from miniagenticrouter.tools.bash import BashTool
from miniagenticrouter.tools.noop import NoOpTool


@dataclass
class FlexibleAgentConfig:
    """Configuration for FlexibleAgent.

    Attributes:
        system_template: Template for system message.
        instance_template: Template for instance/task message.
        timeout_template: Template for timeout error messages.
        format_error_template: Template for format error messages.
        action_observation_template: Template for observations.
        step_limit: Maximum number of model calls (0 = unlimited).
        cost_limit: Maximum cost in dollars (0 = unlimited).
        action_mode: Action mode - "bash", "text", or "none".
        finish_markers: List of strings that signal task completion.
        finish_on_action: Optional action string that triggers finish.
        parser_config: Configuration dict for the parser.
        tool_config: Configuration dict for the tool.
    """

    system_template: str = "You are a helpful assistant that can do anything."
    instance_template: str = (
        "Your task: {{task}}. Please reply with a single shell command in triple backticks. "
        "To finish, the first line of the output of the shell command must be 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'."
    )
    timeout_template: str = (
        "The last command <command>{{action['action']}}</command> timed out and has been killed.\n"
        "The output of the command was:\n"
        "{% if output | length < 10000 -%}\n"
        "<output>\n{{output}}\n</output>\n"
        "{%- else -%}\n"
        "<warning>Output was too long and has been truncated.</warning>\n"
        "<output_head>\n{{ output[:5000] }}\n</output_head>\n"
        "<elided_chars>{{ output | length - 10000 }} characters elided</elided_chars>\n"
        "<output_tail>\n{{ output[-5000:] }}\n</output_tail>\n"
        "{%- endif %}\n"
        "Please try another command and make sure to avoid those requiring interactive input."
    )
    format_error_template: str = "Please always provide EXACTLY ONE action in triple backticks."
    action_observation_template: str = "Observation: {{output}}"
    step_limit: int = 0
    cost_limit: float = 3.0

    # FlexibleAgent-specific fields
    action_mode: str = "bash"  # "bash", "text", or "none"
    finish_markers: list[str] = field(default_factory=lambda: [
        "MINI_SWE_AGENT_FINAL_OUTPUT",
        "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
    ])
    finish_on_action: str | None = None  # Optional action that triggers finish

    # Parser and tool configuration
    parser_config: dict[str, Any] = field(default_factory=dict)
    tool_config: dict[str, Any] = field(default_factory=dict)


class NonTerminatingException(Exception):
    """Raised for conditions that can be handled by the agent."""


class ExecutionTimeoutError(NonTerminatingException):
    """Raised when the action execution timed out."""


class TerminatingException(Exception):
    """Raised for conditions that terminate the agent."""


class Submitted(TerminatingException):
    """Raised when the LM declares that the agent has finished its task."""


class LimitsExceeded(TerminatingException):
    """Raised when the agent has reached its cost or step limit."""


# Mode presets for common configurations
MODE_PRESETS = {
    "bash": {
        "parser_config": {
            "parser_class": "regex",
            "action_regex": r"```bash\s*\n(.*?)\n```",
            "action_type": "bash",
        },
        "tool_class": "bash",
    },
    "text": {
        "parser_config": {
            "parser_class": "regex",
            "action_regex": r"```text\s*\n(.*?)\n```",
            "action_type": "text",
        },
        "tool_class": "text_command",  # Use TextCommandTool for text environments
    },
    "none": {
        "parser_config": {
            "parser_class": "noaction",
        },
        "tool_class": "noop",
    },
}


class FlexibleAgent:
    """Flexible agent supporting multiple action modes.

    This agent extends the DefaultAgent concept with:
    - Pluggable parsers for action extraction
    - Pluggable tools for action execution
    - Configurable finish markers
    - Support for different action modes (bash, text, none)

    Example:
        >>> from miniagenticrouter.models.test_models import DeterministicModel
        >>> from miniagenticrouter.environments.local import LocalEnvironment
        >>> agent = FlexibleAgent(
        ...     model=DeterministicModel(outputs=["```bash\\necho 'done'\\n```"]),
        ...     env=LocalEnvironment(),
        ...     action_mode="bash",
        ... )
        >>> exit_status, result = agent.run("Say done")
    """

    def __init__(
        self,
        model: Model,
        env: Environment | None = None,
        *,
        parser: ActionParser | None = None,
        tool: Tool | None = None,
        config_class: type = FlexibleAgentConfig,
        **kwargs,
    ):
        """Initialize the agent.

        Args:
            model: Language model for generating responses.
            env: Environment for executing commands (optional for "none" mode).
            parser: Custom parser instance (overrides action_mode).
            tool: Custom tool instance (overrides action_mode).
            config_class: Configuration dataclass to use.
            **kwargs: Configuration options passed to config_class.
        """
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}

        # Initialize parser and tool based on mode or custom instances
        self._init_parser(parser)
        self._init_tool(tool)

    def _init_parser(self, parser: ActionParser | None):
        """Initialize the parser based on configuration."""
        if parser is not None:
            self.parser = parser
            return

        # Get preset for the action mode
        mode = self.config.action_mode
        preset = MODE_PRESETS.get(mode, {})

        # Merge preset config with user config
        parser_config = {**preset.get("parser_config", {}), **self.config.parser_config}
        self.parser = get_parser(parser_config)

    def _init_tool(self, tool: Tool | None):
        """Initialize the tool based on configuration."""
        if tool is not None:
            self.tool = tool
            return

        # Get preset for the action mode
        mode = self.config.action_mode
        preset = MODE_PRESETS.get(mode, {})

        # Determine tool class
        tool_class = self.config.tool_config.get("tool_class", preset.get("tool_class", "bash"))

        # Tools that require an environment
        env_required_tools = [
            "bash",
            "miniagenticrouter.tools.bash.BashTool",
            "text_command",
            "miniagenticrouter.tools.text_command.TextCommandTool",
        ]

        # Create tool instance
        if tool_class in env_required_tools:
            if self.env is None:
                raise ValueError(f"Environment required for action_mode='{mode}' with {tool_class} tool")
            self.tool = get_tool(self.config.tool_config, default_type=tool_class, env=self.env)
        else:
            self.tool = get_tool(self.config.tool_config, default_type=tool_class)

    def render_template(self, template: str, **kwargs) -> str:
        """Render a Jinja2 template with agent context."""
        template_vars = asdict(self.config)
        if self.env is not None:
            template_vars |= self.env.get_template_vars()
        template_vars |= self.model.get_template_vars()
        template_vars |= self.parser.get_config()
        template_vars |= self.tool.get_config()
        return Template(template, undefined=StrictUndefined).render(
            **kwargs, **template_vars, **self.extra_template_vars
        )

    def add_message(self, role: str, content: str, **kwargs):
        """Add a message to the conversation history.

        Note: Assistant message content is stripped to avoid trailing whitespace
        issues with providers like AWS Bedrock that enforce strict validation.
        Empty assistant messages are skipped entirely.
        """
        if role == "assistant":
            content = content.rstrip()
            # Skip empty assistant messages (Bedrock doesn't allow them)
            if not content:
                return
        self.messages.append({"role": role, "content": content, "timestamp": time.time(), **kwargs})

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run step() until agent is finished. Return exit status & message."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template))

        while True:
            try:
                self.step()
            except NonTerminatingException as e:
                # Avoid consecutive user messages (some APIs don't support them)
                error_msg = str(e)
                if self.messages and self.messages[-1]["role"] == "user":
                    # Merge with previous user message
                    self.messages[-1]["content"] += f"\n\n{error_msg}"
                else:
                    self.add_message("user", error_msg)
            except TerminatingException as e:
                self.add_message("user", str(e))
                return type(e).__name__, str(e)

    def step(self) -> dict:
        """Query the LM, execute the action, return the observation."""
        return self.get_observation(self.query())

    def query(self) -> dict:
        """Query the model and return the response."""
        if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
            raise LimitsExceeded()
        response = self.model.query(self.messages)
        self.add_message("assistant", **response)
        return response

    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the observation."""
        action = self.parse_action(response)
        output = self.execute_action(action)

        # Only add observation if there's actual output
        if output.get("output"):
            observation = self.render_template(self.config.action_observation_template, output=output["output"])
            self.add_message("user", observation)

        return output

    def parse_action(self, response: dict) -> dict:
        """Parse the action from the response using the parser."""
        try:
            return self.parser.parse(response)
        except FormatError as e:
            # Re-raise as NonTerminatingException with rendered template
            raise NonTerminatingException(
                self.render_template(self.config.format_error_template, error=str(e))
            ) from e

    def execute_action(self, action: dict) -> dict:
        """Execute the action using the tool."""
        # Check for finish trigger on specific action
        if self.config.finish_on_action and action.get("action") == self.config.finish_on_action:
            raise Submitted(action.get("action", ""))

        # Check if action itself is a finish marker (before sending to environment)
        # This is useful for text environments like ScienceWorld where "task completed"
        # is not a valid environment command but signals task completion
        action_text = action.get("action")
        if action_text and action_text.strip() in self.config.finish_markers:
            raise Submitted(action_text)

        # Handle no action case (pure conversation mode)
        # In "none" mode, finish immediately after the model responds
        if action.get("action") is None:
            content = action.get("content", "")
            # If content is empty, ask model to retry instead of submitting
            if not content.strip():
                raise NonTerminatingException(
                    "Your response was empty. Please provide a valid action or response."
                )
            raise Submitted(content)

        try:
            output = self.tool.execute(action)
        except (TimeoutError, subprocess.TimeoutExpired) as e:
            output_text = e.output.decode("utf-8", errors="replace") if getattr(e, "output", None) else ""
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output=output_text)
            )

        # Check for finish markers in output
        self.has_finished(output)

        return {**output, "action": action.get("action")}

    def has_finished(self, output: dict[str, str]):
        """Raises Submitted exception with final output if the agent has finished its task."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() in self.config.finish_markers:
            raise Submitted("".join(lines[1:]))
