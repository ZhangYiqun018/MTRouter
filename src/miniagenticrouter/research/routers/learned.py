"""Learned router that uses Q-function for model selection.

This module provides LearnedRouter, which extends BaseRouter to select
models based on learned Q-values Q(x, a) where x is the state (history)
and a is the model choice.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from miniagenticrouter.routers.base import BaseRouter
from miniagenticrouter.research.routers.policies import (
    GreedyPolicy,
    SelectionPolicy,
)

if TYPE_CHECKING:
    from miniagenticrouter import Model


def validate_model_names(
    checkpoint_model_names: list[str],
    expected_model_names: list[str] | None = None,
    *,
    source: str = "checkpoint",
) -> None:
    """Validate model names match between checkpoint and expected.

    This ensures that the model index mapping is consistent between
    training and inference, preventing silent routing errors.

    Args:
        checkpoint_model_names: Model names from checkpoint.
        expected_model_names: Expected model names (if None, skip validation).
        source: Source description for error message.

    Raises:
        ValueError: If model names don't match (including order).
    """
    if expected_model_names is None:
        return

    if checkpoint_model_names != expected_model_names:
        raise ValueError(
            f"Model names mismatch between {source} and expected.\n"
            f"  {source}: {checkpoint_model_names}\n"
            f"  expected: {expected_model_names}\n"
            "These must match (including order) for correct routing."
        )


class QFunction(ABC):
    """Abstract interface for Q-function estimators.

    A Q-function estimates the value Q(x, a) of selecting model a
    given state x (conversation history).

    Implementations can use:
    - Neural networks (MLP, Transformer)
    - Linear models
    - Simple heuristics for baselines
    """

    @abstractmethod
    def predict(
        self,
        history: str | list[dict],
        available_models: list[str],
    ) -> np.ndarray:
        """Predict Q-values for each available model.

        Args:
            history: Conversation history (text or messages).
            available_models: List of model names.

        Returns:
            Q-value array, shape=(len(available_models),)
        """
        ...

    def predict_with_features(
        self,
        features: dict[str, Any],
        available_models: list[str],
    ) -> np.ndarray:
        """Predict Q-values from pre-computed features.

        Override this for more efficient inference when features
        are already computed.

        Args:
            features: Pre-computed state features.
            available_models: List of model names.

        Returns:
            Q-value array.
        """
        # Default: extract history and use predict()
        history = features.get("history_text", "")
        return self.predict(history, available_models)

    def load(self, path: Path | str) -> None:
        """Load model weights from path.

        Args:
            path: Path to weights file.
        """
        raise NotImplementedError("Subclass must implement load()")

    def save(self, path: Path | str) -> None:
        """Save model weights to path.

        Args:
            path: Path to save weights.
        """
        raise NotImplementedError("Subclass must implement save()")


class RandomQFunction(QFunction):
    """Random Q-function for baseline comparison.

    Returns uniform random Q-values, equivalent to random selection.
    """

    def predict(
        self,
        history: str | list[dict],
        available_models: list[str],
    ) -> np.ndarray:
        """Return random Q-values."""
        return np.random.uniform(0, 1, size=len(available_models))


class ConstantQFunction(QFunction):
    """Constant Q-function that always prefers a specific model.

    Useful for testing and as a baseline.

    Args:
        preferred_model_idx: Index of the preferred model.
        preference_strength: How much to prefer the model.
    """

    def __init__(
        self,
        preferred_model_idx: int = 0,
        preference_strength: float = 1.0,
    ):
        self.preferred_model_idx = preferred_model_idx
        self.preference_strength = preference_strength

    def predict(
        self,
        history: str | list[dict],
        available_models: list[str],
    ) -> np.ndarray:
        """Return Q-values with preference for one model."""
        q_values = np.zeros(len(available_models))
        if self.preferred_model_idx < len(available_models):
            q_values[self.preferred_model_idx] = self.preference_strength
        return q_values


class CostAwareQFunction(QFunction):
    """Q-function that penalizes expensive models.

    Simple heuristic baseline that assigns Q-values inversely
    proportional to model cost.

    Args:
        model_costs: Dict mapping model names to costs.
        lambda_: Cost penalty coefficient.
    """

    def __init__(
        self,
        model_costs: dict[str, float],
        lambda_: float = 1.0,
    ):
        self.model_costs = model_costs
        self.lambda_ = lambda_

    def predict(
        self,
        history: str | list[dict],
        available_models: list[str],
    ) -> np.ndarray:
        """Return Q-values inversely proportional to cost."""
        q_values = []
        max_cost = max(self.model_costs.values()) if self.model_costs else 1.0

        for model in available_models:
            cost = self.model_costs.get(model, max_cost)
            # Higher Q for lower cost
            q = 1.0 - self.lambda_ * (cost / max_cost)
            q_values.append(q)

        return np.array(q_values)


@dataclass
class LearnedRouterConfig:
    """Configuration for LearnedRouter.

    Attributes:
        model_kwargs: List of model configuration dicts.
        q_function_path: Path to Q-function weights (optional).
        policy: Selection policy name.
        policy_kwargs: Policy configuration.
        model_name: Router identifier.
    """

    model_kwargs: list[dict]
    q_function_path: Path | str | None = None
    policy: str = "greedy"
    policy_kwargs: dict[str, Any] = field(default_factory=dict)
    model_name: str = "learned"


class LearnedRouter(BaseRouter):
    """Router that uses a learned Q-function for model selection.

    This router estimates the value of selecting each model given the
    current conversation state, and uses a selection policy to choose.

    Compatible with existing BaseRouter interface:
    - Implements select_model()
    - Aggregates cost/n_calls across models
    - Works with FlexibleAgent

    The Q-function can be:
    - A trained neural network
    - A simple heuristic (for baselines)
    - None (defaults to random selection)

    Example:
        >>> from miniagenticrouter.research.routers.learned import (
        ...     LearnedRouter, CostAwareQFunction
        ... )
        >>> q_func = CostAwareQFunction(
        ...     model_costs={"haiku": 0.25, "sonnet": 3.0, "opus": 15.0}
        ... )
        >>> router = LearnedRouter(
        ...     model_kwargs=[
        ...         {"model_name": "claude-3-5-haiku-latest"},
        ...         {"model_name": "claude-sonnet-4-5-20250929"},
        ...     ],
        ...     q_function=q_func,
        ...     policy="greedy",
        ... )
        >>> response = router.query(messages)
    """

    def __init__(
        self,
        *,
        config_class: type = LearnedRouterConfig,
        q_function: QFunction | None = None,
        policy: SelectionPolicy | None = None,
        fallback_model_name: str | None = None,
        **kwargs,
    ):
        """Initialize LearnedRouter.

        Args:
            config_class: Configuration dataclass.
            q_function: Q-function for value estimation.
            policy: Selection policy (uses config.policy if None).
            fallback_model_name: Model name for context window fallback.
            **kwargs: Configuration parameters.
        """
        self.config = config_class(**kwargs)
        super().__init__(
            model_kwargs=self.config.model_kwargs,
            fallback_model_name=fallback_model_name,
        )

        # Initialize Q-function
        self.q_function = q_function
        if self.q_function is None and self.config.q_function_path:
            self._load_q_function()

        # Initialize selection policy
        if policy is not None:
            self.policy = policy
        else:
            self.policy = self._create_policy()

        # Model name mapping
        self._model_names = [m.config.model_name for m in self.models]

        # Current context (set before select_model)
        self._current_messages: list[dict] = []
        self._last_q_values: np.ndarray | None = None

    def _load_q_function(self) -> None:
        """Load Q-function from config path.

        Override this method to support different Q-function types.
        """
        # Default: do nothing (subclass should implement)
        pass

    def _create_policy(self) -> SelectionPolicy:
        """Create selection policy from config."""
        from miniagenticrouter.research.routers.policies import (
            EpsilonGreedyPolicy,
            GreedyPolicy,
            SoftmaxPolicy,
            ThompsonSamplingPolicy,
            UCBPolicy,
        )

        policy_map = {
            "greedy": GreedyPolicy,
            "epsilon_greedy": EpsilonGreedyPolicy,
            "softmax": SoftmaxPolicy,
            "ucb": UCBPolicy,
            "thompson": ThompsonSamplingPolicy,
        }

        policy_name = self.config.policy.lower()
        if policy_name not in policy_map:
            raise ValueError(f"Unknown policy: {policy_name}")

        policy_cls = policy_map[policy_name]
        return policy_cls(**self.config.policy_kwargs)

    def set_context(self, messages: list[dict]) -> None:
        """Set the current conversation context.

        Call this before select_model() to provide the state
        for Q-value computation.

        Args:
            messages: Current conversation messages.
        """
        self._current_messages = messages

    def select_model(self) -> Model:
        """Select a model based on Q-values and policy.

        Uses the current context (set via set_context or query)
        to compute Q-values and select a model.

        Returns:
            Selected model instance.
        """
        if self.q_function is None:
            # Fallback: random selection
            import random

            return random.choice(self.models)

        # Compute Q-values
        q_values = self.q_function.predict(
            history=self._current_messages,
            available_models=self._model_names,
        )
        self._last_q_values = q_values

        # Select using policy
        model_indices = list(range(len(self.models)))
        selected_idx = self.policy.select(q_values, model_indices)

        return self.models[selected_idx]

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        """Execute query with automatic context setting.

        Overrides BaseRouter.query() to set context before selection.

        Args:
            messages: Conversation messages.
            **kwargs: Additional query parameters.

        Returns:
            Response dict with Q-value metadata.
        """
        # Set context for select_model()
        self.set_context(messages)

        # Call parent (which calls select_model)
        response = super().query(messages, **kwargs)

        # Add Q-value information
        if self._last_q_values is not None:
            response["q_values"] = self._last_q_values.tolist()
            # Only add selected_q if model is in pool (not a fallback model)
            if response["model_name"] in self._model_names:
                response["selected_q"] = float(
                    self._last_q_values[self._model_names.index(response["model_name"])]
                )

        response["router_info"] = {
            "router_type": self.__class__.__name__,
            "policy_type": self.policy.__class__.__name__,
            "q_values": (
                self._last_q_values.tolist() if self._last_q_values is not None else []
            ),
            "model_pool": self._model_names,
        }

        return response

    def get_q_values(self, messages: list[dict]) -> np.ndarray:
        """Get Q-values for given messages without selecting.

        Useful for analysis and debugging.

        Args:
            messages: Conversation messages.

        Returns:
            Q-value array for each model.
        """
        if self.q_function is None:
            return np.zeros(len(self.models))

        return self.q_function.predict(
            history=messages,
            available_models=self._model_names,
        )

    def get_model_ranking(self, messages: list[dict]) -> list[tuple[str, float]]:
        """Get models ranked by Q-value.

        Args:
            messages: Conversation messages.

        Returns:
            List of (model_name, q_value) tuples, sorted by Q-value.
        """
        q_values = self.get_q_values(messages)
        ranking = list(zip(self._model_names, q_values))
        return sorted(ranking, key=lambda x: x[1], reverse=True)

    def get_template_vars(self) -> dict[str, Any]:
        """Get template variables."""
        base_vars = super().get_template_vars()
        return {
            **base_vars,
            "policy_type": self.policy.__class__.__name__,
            "has_q_function": self.q_function is not None,
        }

    def reset_policy(self) -> None:
        """Reset policy state (e.g., for new episode)."""
        self.policy.reset()

    def set_lambda(self, lambda_: float) -> None:
        """Set the cost penalty coefficient for Q-value computation.

        This allows dynamic adjustment of the score-cost tradeoff
        at inference time without retraining.

        Args:
            lambda_: New cost penalty coefficient.
        """
        if self.q_function is not None and hasattr(self.q_function, 'set_lambda'):
            self.q_function.set_lambda(lambda_)
