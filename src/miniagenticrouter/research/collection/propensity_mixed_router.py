"""Propensity-tracking mixed router for Stage B data collection.

This module provides PropensityMixedRouter, which combines a learned Q-function
policy with uniform random selection (roulette) at a configurable mixing ratio.
The selection propensity is correctly computed for offline policy evaluation.

Mixed Strategy:
    With probability beta: use learned policy (greedy on Q-values)
    With probability (1-beta): use uniform random (roulette)

Propensity Calculation:
    For model a with n total models:
    - If a is the greedy choice: propensity = beta + (1-beta)/n
    - If a is not the greedy choice: propensity = (1-beta)/n
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from miniagenticrouter.routers.base import BaseRouter

if TYPE_CHECKING:
    from miniagenticrouter import Model


@dataclass
class PropensityMixedRouterConfig:
    """Configuration for PropensityMixedRouter.

    Attributes:
        model_kwargs: List of model configuration dicts.
        model_name: Name identifier for this router instance.
        beta: Probability of using learned policy (0-1).
        record_selection_history: Whether to keep a history of all selections.
    """

    model_kwargs: list[dict]
    model_name: str = "propensity_mixed"
    beta: float = 0.5
    record_selection_history: bool = False


class PropensityMixedRouter(BaseRouter):
    """Mixed router that combines learned and random selection with propensity tracking.

    This router implements the β-mixed strategy for Stage B data collection:
    - With probability β: select model with highest Q-value (learned policy)
    - With probability (1-β): select uniformly at random (roulette)

    The propensity is computed as:
        propensity(a) = β * I[a = greedy_choice] + (1-β) / n

    This ensures correct importance weights for offline evaluation.

    Example:
        >>> from miniagenticrouter.research.training import NeuralQFunction
        >>> q_func = NeuralQFunction(model_names=["model_a", "model_b"])
        >>> q_func.load("checkpoint.pt")
        >>> router = PropensityMixedRouter(
        ...     model_kwargs=[{"model_name": "model_a"}, {"model_name": "model_b"}],
        ...     q_function=q_func,
        ...     beta=0.5,
        ... )
        >>> response = router.query(messages)
        >>> print(response["propensity"])  # Correctly computed propensity
    """

    def __init__(
        self,
        *,
        q_function: Any,  # NeuralQFunction or compatible
        config_class: type = PropensityMixedRouterConfig,
        fallback_model_name: str | None = None,
        **kwargs,
    ):
        """Initialize PropensityMixedRouter.

        Args:
            q_function: Trained Q-function for value estimation.
            config_class: Configuration dataclass.
            fallback_model_name: Model name for context window fallback.
            **kwargs: Configuration parameters including:
                - model_kwargs: List of model configuration dicts.
                - beta: Mixing ratio (probability of using learned policy).
        """
        self.config = config_class(**kwargs)
        super().__init__(
            model_kwargs=self.config.model_kwargs,
            fallback_model_name=fallback_model_name,
        )

        self.q_function = q_function
        self.beta = self.config.beta

        # Build model name to index mapping
        self._model_names = [m.config.model_name for m in self.models]
        self._model_name_to_idx = {name: i for i, name in enumerate(self._model_names)}
        self._n_models = len(self.models)

        # Track last selection for query() to use
        self._last_selected_model: Model | None = None
        self._last_propensity: float = 0.0
        self._last_greedy_idx: int = -1
        self._last_was_greedy: bool = False

        # Optional selection history
        self._selection_history: list[dict[str, Any]] = []

        # Current messages for Q-value computation
        self._current_messages: list[dict] = []

    def _get_greedy_idx(self, messages: list[dict]) -> int:
        """Get the index of the greedy (highest Q-value) model.

        Args:
            messages: Current conversation history.

        Returns:
            Index of the model with highest Q-value.
        """
        q_values = self.q_function.predict(messages, self._model_names)
        return int(np.argmax(q_values))

    def select_model(self) -> Model:
        """Select a model using mixed strategy and record propensity.

        The selection follows:
        - With prob β: select greedy (highest Q-value)
        - With prob (1-β): select uniformly at random

        Returns:
            Selected model instance.
        """
        # Get greedy choice based on current messages
        greedy_idx = self._get_greedy_idx(self._current_messages)
        self._last_greedy_idx = greedy_idx

        # Mixed selection
        if random.random() < self.beta:
            # Learned policy: select greedy
            selected_idx = greedy_idx
            self._last_was_greedy = True
        else:
            # Roulette: uniform random
            selected_idx = random.randint(0, self._n_models - 1)
            self._last_was_greedy = False

        selected = self.models[selected_idx]
        self._last_selected_model = selected

        # Compute propensity
        # p(a) = β * I[a=greedy] + (1-β) / n
        if selected_idx == greedy_idx:
            self._last_propensity = self.beta + (1 - self.beta) / self._n_models
        else:
            self._last_propensity = (1 - self.beta) / self._n_models

        if self.config.record_selection_history:
            self._selection_history.append(
                {
                    "step": self.n_calls,
                    "selected_model": selected.config.model_name,
                    "selected_idx": selected_idx,
                    "greedy_idx": greedy_idx,
                    "greedy_model": self._model_names[greedy_idx],
                    "was_greedy": self._last_was_greedy,
                    "propensity": self._last_propensity,
                    "beta": self.beta,
                    "n_models": self._n_models,
                }
            )

        return selected

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        """Execute query and add propensity to response.

        Args:
            messages: List of message dicts.
            **kwargs: Additional query parameters.

        Returns:
            Response dict with additional fields:
            - propensity: Selection probability under mixed strategy
            - selected_model: Name of the selected model
            - router_info: Dict with router metadata
        """
        # Store messages for select_model() to use
        self._current_messages = messages

        # Call parent query (which calls select_model)
        response = super().query(messages, **kwargs)

        # Add propensity information
        response["propensity"] = self._last_propensity
        response["selected_model"] = (
            self._last_selected_model.config.model_name
            if self._last_selected_model
            else ""
        )

        # Add detailed router info for trajectory logging
        response["router_info"] = {
            "router_type": self.__class__.__name__,
            "propensity": self._last_propensity,
            "selected_model": response["selected_model"],
            "greedy_model": self._model_names[self._last_greedy_idx],
            "was_greedy": self._last_was_greedy,
            "beta": self.beta,
            "n_models": self._n_models,
            "model_pool": self._model_names,
        }

        return response

    def get_selection_history(self) -> list[dict[str, Any]]:
        """Get the selection history.

        Returns:
            List of selection records (empty if not recording).
        """
        return self._selection_history.copy()

    def get_template_vars(self) -> dict:
        """Get template variables.

        Returns:
            Dict with router stats and configuration.
        """
        return asdict(self.config) | super().get_template_vars()

    def reset_history(self) -> None:
        """Clear the selection history."""
        self._selection_history.clear()
