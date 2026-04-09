"""LLM-based router for model selection.

This module provides LLMRouter, which uses an LLM policy to decide
which model to use for each query. Unlike multi-round routing approaches,
this router only performs single-turn model selection.

The LLM outputs:
- <think>reasoning about which model to use</think>
- <select>model_name</select>

Example:
    >>> router = LLMRouter(
    ...     model_kwargs=[
    ...         {"model_name": "deepseek/deepseek-v3.2"},
    ...         {"model_name": "openai/gpt-5"},
    ...     ],
    ...     policy_model_name="openai/gpt-5",
    ... )
    >>> response = router.query(messages)
    >>> print(response["router_info"]["selected_model"])
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from miniagenticrouter.models import get_model
from miniagenticrouter.routers.base import BaseRouter

if TYPE_CHECKING:
    from miniagenticrouter import Model


def parse_select_tag(text: str) -> str | None:
    """Parse <select>model_name</select> from LLM output.

    Args:
        text: LLM output text.

    Returns:
        Model name if found, None otherwise.
    """
    match = re.search(r"<select>\s*([^<]+?)\s*</select>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def parse_think_tag(text: str) -> str | None:
    """Parse <think>reasoning</think> from LLM output.

    Args:
        text: LLM output text.

    Returns:
        Think content if found, None otherwise.
    """
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return match.group(1).strip() if match else None


@dataclass
class LLMRouterConfig:
    """Configuration for LLMRouter.

    Attributes:
        model_kwargs: List of model configuration dicts for target models.
        policy_model_name: Name of the LLM to use for routing decisions.
        model_descriptors: Optional descriptions of each model's strengths.
        history_turns: Number of conversation turns to show (0 = all).
        model_name: Name identifier for this router instance.
    """

    model_kwargs: list[dict]
    policy_model_name: str
    model_descriptors: dict[str, str] = field(default_factory=dict)
    history_turns: int = 3
    model_name: str = "llmrouter"


class LLMRouter(BaseRouter):
    """Router that uses an LLM policy for model selection.

    This router is behaviorally consistent with RouletteRouter/LearnedRouter:
    - select_model() decides which model to use
    - The target model's response is the final answer

    The difference is in how select_model() works:
    - RouletteRouter: random.choice(models)
    - LearnedRouter: argmax(Q-function(state))
    - LLMRouter: parse(policy_llm.query(prompt))

    Example:
        >>> router = LLMRouter(
        ...     model_kwargs=[
        ...         {"model_name": "deepseek/deepseek-v3.2"},
        ...         {"model_name": "openai/gpt-5"},
        ...     ],
        ...     policy_model_name="openai/gpt-5",
        ...     model_descriptors={
        ...         "deepseek/deepseek-v3.2": "Low cost, good for simple tasks",
        ...         "openai/gpt-5": "High quality, good for complex reasoning",
        ...     },
        ... )
        >>> response = router.query(messages)
        >>> print(response["router_info"]["think"])  # Reasoning
        >>> print(response["router_info"]["selected_model"])  # Which model was chosen
    """

    def __init__(
        self,
        *,
        policy_model_name: str,
        model_descriptors: dict[str, str] | None = None,
        history_turns: int = 3,
        fallback_model_name: str | None = None,
        config_class: type = LLMRouterConfig,
        **kwargs,
    ):
        """Initialize LLMRouter.

        Args:
            policy_model_name: Name of the LLM to use for routing decisions.
            model_descriptors: Optional dict mapping model names to descriptions.
            history_turns: Number of conversation turns to show (0 = all).
            fallback_model_name: Model name for context window fallback.
            config_class: Configuration dataclass.
            **kwargs: Configuration parameters including model_kwargs.
        """
        super().__init__(**kwargs, fallback_model_name=fallback_model_name)

        self.config = config_class(
            model_kwargs=kwargs.get("model_kwargs", []),
            policy_model_name=policy_model_name,
            model_descriptors=model_descriptors or {},
            history_turns=history_turns,
        )

        # Initialize policy LLM
        self._policy_model = get_model(config={"model_name": policy_model_name})
        self._model_descriptors = model_descriptors or {}

        # Model name mapping
        self._model_names = [m.config.model_name for m in self.models]
        self._model_name_to_idx = {name: i for i, name in enumerate(self._model_names)}

        # Context (set before select_model via query)
        self._current_messages: list[dict] = []

        # Track last selection info
        self._last_think: str = ""
        self._last_selected_name: str = ""
        self._last_policy_output: str = ""

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the policy LLM.

        Returns:
            System prompt string with model descriptions and instructions.
        """
        model_list = "\n".join(
            f"- {name}: {self._model_descriptors.get(name, 'General purpose LLM')}"
            for name in self._model_names
        )
        return f"""You are a model routing assistant. Your job is to select the best model for the given task.

Available models:
{model_list}

Instructions:
1. Analyze the task in <think>...</think> tags
2. Select the best model using <select>model_name</select>

Example:
<think>This is a simple math question. A cheaper model would suffice.</think>
<select>deepseek/deepseek-v3.2</select>

Rules:
- You MUST output exactly one <select> tag
- The model name must match exactly from the available list
- Consider: task complexity, model strengths, cost-effectiveness
"""

    def _build_task_prompt(self, messages: list[dict]) -> str:
        """Build task description from conversation history.

        Shows the last N turns of conversation based on config.history_turns.
        A turn is defined as a user-assistant message pair.

        Args:
            messages: Conversation messages.

        Returns:
            Task prompt string with conversation history.
        """
        if not messages:
            return "No messages provided."

        history_turns = self.config.history_turns
        total_messages = len(messages)

        # Determine which messages to show
        if history_turns == 0:
            # Show all messages
            selected_messages = messages
        else:
            # Show last N turns (approximately 2*N messages for user-assistant pairs)
            # But ensure we at least show the last message
            n_messages = max(1, history_turns * 2)
            selected_messages = messages[-n_messages:] if total_messages > n_messages else messages

        # Build formatted history
        lines = []
        lines.append(f"Current step: {total_messages // 2 + 1}")
        lines.append(f"Showing last {len(selected_messages)} messages (history_turns={history_turns}):")
        lines.append("")

        for msg in selected_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            # Format role
            if role == "user":
                role_label = "[User]"
            elif role == "assistant":
                role_label = "[Assistant]"
            elif role == "system":
                role_label = "[System]"
            else:
                role_label = f"[{role}]"

            # Format content (handle tool calls if present)
            if isinstance(content, str):
                display_content = content
            else:
                display_content = str(content)

            lines.append(f"{role_label}")
            lines.append(display_content)
            lines.append("")

        return "\n".join(lines)

    def select_model(self) -> Model:
        """Select a model using LLM policy.

        Queries the policy LLM to analyze the task and select the best model.

        Returns:
            Selected model instance.
        """
        # Build prompt
        system_prompt = self._build_system_prompt()
        task_prompt = self._build_task_prompt(self._current_messages)

        policy_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt},
        ]

        # Query policy LLM
        response = self._policy_model.query(policy_messages)
        output = response["content"]
        self._last_policy_output = output

        # Parse output
        self._last_think = parse_think_tag(output) or ""
        selected_name = parse_select_tag(output)

        # Validate and select model
        if selected_name and selected_name in self._model_name_to_idx:
            self._last_selected_name = selected_name
            return self.models[self._model_name_to_idx[selected_name]]

        # Parse failed: fallback to first model
        self._last_selected_name = self._model_names[0] if self._model_names else ""
        return self.models[0]

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        """Execute query with automatic context setting.

        Overrides BaseRouter.query() to set context before selection.

        Args:
            messages: Conversation messages.
            **kwargs: Additional query parameters.

        Returns:
            Response dict with LLMRouter-specific metadata.
        """
        self._current_messages = messages

        # Call parent query (which calls select_model)
        response = super().query(messages, **kwargs)

        # Add LLMRouter-specific metadata
        response["router_info"] = {
            "router_type": "LLMRouter",
            "policy_model": self._policy_model.config.model_name,
            "selected_model": self._last_selected_name,
            "think": self._last_think,
            "model_pool": self._model_names,
            "policy_cost": self._policy_model.cost,
            "policy_output": self._last_policy_output,
        }

        return response

    @property
    def cost(self) -> float:
        """Total cost including policy LLM cost.

        Returns:
            Sum of policy model cost and all target model costs.
        """
        return self._policy_model.cost + sum(m.cost for m in self.models)

    def get_template_vars(self) -> dict[str, Any]:
        """Get template variables.

        Returns:
            Dict with router configuration and stats.
        """
        return {
            **asdict(self.config),
            **super().get_template_vars(),
            "policy_model": self._policy_model.config.model_name,
            "policy_cost": self._policy_model.cost,
        }
