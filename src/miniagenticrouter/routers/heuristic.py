"""Heuristic router for rule-based model selection without training.

This router implements simple rules for model selection:
1. First step -> strong model
2. Tool error in previous turn -> strong model
3. Late phase (step >= 2/3 * max_steps) -> value model
4. Otherwise -> random from cheap pool

Usage:
    >>> from miniagenticrouter.routers.heuristic import HeuristicRouter
    >>> router = HeuristicRouter(
    ...     model_kwargs=[
    ...         {"model_name": "openai/gpt-5"},
    ...         {"model_name": "deepseek/deepseek-v3.2"},
    ...         {"model_name": "minimax/minimax-m2"},
    ...     ],
    ...     task_type="hle",
    ...     max_steps=30,
    ...     strong_model="openai/gpt-5",
    ...     value_model="deepseek/deepseek-v3.2",
    ...     cheap_pool=["minimax/minimax-m2"],
    ... )
    >>> router.set_context(messages, step=1)
    >>> response = router.query(messages)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from miniagenticrouter.routers.base import BaseRouter

if TYPE_CHECKING:
    from miniagenticrouter import Model


@dataclass
class HeuristicRouterConfig:
    """Configuration for HeuristicRouter.

    Attributes:
        model_kwargs: List of model configuration dicts.
        task_type: Task type ("hle" or "scienceworld").
        max_steps: Maximum steps for the task.
        strong_model: Name of the strongest model.
        value_model: Name of the best value model.
        cheap_pool: List of cheap model names.
        late_phase_fraction: Fraction of max_steps to trigger late phase.
        error_patterns: Patterns that indicate tool errors.
        check_returncode: Whether to check returncode for errors.
        model_name: Router identifier.
    """

    model_kwargs: list[dict]
    task_type: str = "hle"
    max_steps: int = 30
    strong_model: str = "openai/gpt-5"
    value_model: str = "deepseek/deepseek-v3.2"
    cheap_pool: list[str] = field(
        default_factory=lambda: [
            "minimax/minimax-m2",
            "moonshotai/kimi-k2-0905",
            "x-ai/grok-4.1-fast",
        ]
    )
    late_phase_fraction: float = 0.667
    error_patterns: list[str] = field(
        default_factory=lambda: [
            "unknown_tool:",
            "validation_error:",
            "execution_error:",
        ]
    )
    check_returncode: bool = True
    model_name: str = "heuristic"


class HeuristicRouter(BaseRouter):
    """Rule-based router without training.

    Selection strategy (in priority order):
    1. If step == 1 -> strong_model
    2. If tool_error_prev == True -> strong_model
    3. If step >= ceil(late_phase_fraction * max_steps) -> value_model
    4. Otherwise -> random from cheap_pool

    The router automatically detects tool errors from the message history
    and infers the current step if not explicitly provided.

    Attributes:
        config: Router configuration.
        models: List of managed model instances.

    Example:
        >>> router = HeuristicRouter(
        ...     model_kwargs=[
        ...         {"model_name": "openai/gpt-5"},
        ...         {"model_name": "deepseek/deepseek-v3.2"},
        ...         {"model_name": "minimax/minimax-m2"},
        ...     ],
        ...     task_type="hle",
        ...     max_steps=30,
        ... )
        >>> router.set_context(messages, step=1)
        >>> response = router.query(messages)
    """

    def __init__(
        self,
        *,
        config_class: type = HeuristicRouterConfig,
        fallback_model_name: str | None = None,
        **kwargs,
    ):
        """Initialize HeuristicRouter.

        Args:
            config_class: Configuration dataclass type.
            fallback_model_name: Model name for context window fallback.
            **kwargs: Configuration parameters passed to config_class.
        """
        self.config = config_class(**kwargs)
        super().__init__(
            model_kwargs=self.config.model_kwargs,
            fallback_model_name=fallback_model_name,
        )

        # Build model name -> Model mapping
        self._model_map: dict[str, Model] = {
            m.config.model_name: m for m in self.models
        }
        self._model_names = list(self._model_map.keys())

        # Validate configuration
        self._validate_config()

        # Context state (set before each selection)
        self._current_step: int = 1
        self._tool_error_prev: bool = False
        self._current_messages: list[dict] = []

        # Last selection info (for response metadata)
        self._last_selected_model: Model | None = None
        self._last_selection_reason: str = ""
        self._last_propensity: float = 0.0

    def _validate_config(self) -> None:
        """Validate that configured models exist in model pool.

        Raises:
            ValueError: If any configured model is not in the model pool.
        """
        missing = []
        if self.config.strong_model not in self._model_map:
            missing.append(f"strong_model: {self.config.strong_model}")
        if self.config.value_model not in self._model_map:
            missing.append(f"value_model: {self.config.value_model}")
        for cheap in self.config.cheap_pool:
            if cheap not in self._model_map:
                missing.append(f"cheap_pool: {cheap}")

        if missing:
            raise ValueError(
                f"Missing models in pool: {missing}\n"
                f"Available: {self._model_names}"
            )

    def set_context(
        self,
        messages: list[dict],
        step: int | None = None,
        tool_error_prev: bool | None = None,
    ) -> None:
        """Set context for model selection.

        This method should be called before select_model() or query()
        to provide the current state information.

        Args:
            messages: Current conversation messages.
            step: Current step number (1-indexed). If None, inferred from messages.
            tool_error_prev: Whether previous turn had tool error. If None, detected.
        """
        self._current_messages = messages

        # Infer step from messages if not provided
        if step is not None:
            self._current_step = step
        else:
            self._current_step = self._infer_step(messages)

        # Detect tool error if not provided
        if tool_error_prev is not None:
            self._tool_error_prev = tool_error_prev
        else:
            self._tool_error_prev = self._detect_tool_error(messages)

    def _infer_step(self, messages: list[dict]) -> int:
        """Infer current step from message count.

        Step = number of assistant messages + 1 (for the upcoming response).

        Args:
            messages: Conversation messages.

        Returns:
            Inferred step number (1-indexed).
        """
        assistant_count = sum(1 for m in messages if m.get("role") == "assistant")
        return assistant_count + 1

    def _detect_tool_error(self, messages: list[dict]) -> bool:
        """Detect if the previous turn had a tool error.

        Checks the last user message for:
        1. Error patterns in content (from tool_response)
        2. returncode != 0 markers

        Args:
            messages: Conversation messages.

        Returns:
            True if tool error detected, False otherwise.
        """
        if len(messages) < 2:
            return False

        # Find the last user message (which contains tool response)
        last_user = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = msg
                break

        if last_user is None:
            return False

        content = last_user.get("content", "")

        # Check for error patterns
        for pattern in self.config.error_patterns:
            if pattern in content:
                return True

        # Check for returncode indicators in tool_response
        if self.config.check_returncode:
            # Look for common error indicators
            if 'returncode="1"' in content or "returncode=1" in content:
                return True
            # Check for Error: prefix in tool response
            if "<tool_response" in content and "Error:" in content:
                return True

        return False

    def _compute_late_phase_threshold(self) -> int:
        """Compute the step threshold for late phase.

        Returns:
            Step number at which late phase begins.
        """
        return math.ceil(self.config.late_phase_fraction * self.config.max_steps)

    def select_model(self) -> Model:
        """Select model based on heuristic rules.

        Priority:
        1. step == 1 -> strong_model
        2. tool_error_prev -> strong_model
        3. step >= late_phase_threshold -> value_model
        4. Otherwise -> random cheap model

        Returns:
            Selected model instance.
        """
        step = self._current_step
        late_threshold = self._compute_late_phase_threshold()

        # Rule 1: First step -> strong model
        if step == 1:
            selected_name = self.config.strong_model
            reason = "first_step"
            propensity = 1.0

        # Rule 2: Tool error -> strong model
        elif self._tool_error_prev:
            selected_name = self.config.strong_model
            reason = "tool_error_recovery"
            propensity = 1.0

        # Rule 3: Late phase -> value model
        elif step >= late_threshold:
            selected_name = self.config.value_model
            reason = f"late_phase (step {step} >= {late_threshold})"
            propensity = 1.0

        # Rule 4: Otherwise -> random cheap model
        else:
            selected_name = random.choice(self.config.cheap_pool)
            reason = "cheap_exploration"
            propensity = 1.0 / len(self.config.cheap_pool)

        self._last_selected_model = self._model_map[selected_name]
        self._last_selection_reason = reason
        self._last_propensity = propensity

        return self._last_selected_model

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        """Execute query with automatic context detection.

        If step/tool_error not explicitly set via set_context(),
        they will be inferred from messages.

        Args:
            messages: Conversation messages.
            **kwargs: Additional query parameters.

        Returns:
            Response dict with router metadata including:
                - model_name: Selected model name
                - propensity: Selection probability
                - router_info: Detailed routing information
        """
        # Auto-set context if not already set for this messages
        if self._current_messages != messages:
            self.set_context(messages)

        # Call parent (which calls select_model)
        response = super().query(messages, **kwargs)

        # Add heuristic router info
        response["propensity"] = self._last_propensity
        response["selected_model"] = (
            self._last_selected_model.config.model_name
            if self._last_selected_model
            else ""
        )

        response["router_info"] = {
            "router_type": self.__class__.__name__,
            "selection_reason": self._last_selection_reason,
            "step": self._current_step,
            "max_steps": self.config.max_steps,
            "tool_error_prev": self._tool_error_prev,
            "late_phase_threshold": self._compute_late_phase_threshold(),
            "propensity": self._last_propensity,
            "model_pool": self._model_names,
            "task_type": self.config.task_type,
        }

        return response

    def reset(self) -> None:
        """Reset context for new episode."""
        self._current_step = 1
        self._tool_error_prev = False
        self._current_messages = []
        self._last_selected_model = None
        self._last_selection_reason = ""
        self._last_propensity = 0.0

    def get_template_vars(self) -> dict[str, Any]:
        """Get template variables for Jinja2 rendering.

        Returns:
            Dict containing base router vars plus heuristic-specific vars.
        """
        base_vars = super().get_template_vars()
        return {
            **base_vars,
            "task_type": self.config.task_type,
            "max_steps": self.config.max_steps,
            "strong_model": self.config.strong_model,
            "value_model": self.config.value_model,
            "cheap_pool": self.config.cheap_pool,
            "late_phase_threshold": self._compute_late_phase_threshold(),
        }
