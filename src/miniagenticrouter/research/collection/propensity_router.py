"""Propensity-tracking roulette router for offline evaluation.

This module provides PropensityRouletteRouter, which extends RouletteRouter
to record the selection probability (propensity) for each model selection.
This is essential for offline policy evaluation and importance sampling.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from miniagenticrouter.routers.base import BaseRouter

if TYPE_CHECKING:
    from miniagenticrouter import Model


@dataclass
class PropensityRouletteRouterConfig:
    """Configuration for PropensityRouletteRouter.

    Attributes:
        model_kwargs: List of model configuration dicts.
        model_name: Name identifier for this router instance.
        record_selection_history: Whether to keep a history of all selections.
    """

    model_kwargs: list[dict]
    model_name: str = "propensity_roulette"
    record_selection_history: bool = False


class PropensityRouletteRouter(BaseRouter):
    """Roulette router that records selection propensity.

    This router extends the basic roulette selection with:
    - Recording the propensity (selection probability) for each query
    - Adding propensity to the query response for trajectory logging
    - Optional selection history tracking

    The propensity is uniform (1/n) for all models in the pool.

    Example:
        >>> router = PropensityRouletteRouter(model_kwargs=[
        ...     {"model_name": "claude-3-5-haiku-latest"},
        ...     {"model_name": "claude-sonnet-4-5-20250929"},
        ...     {"model_name": "claude-opus-4-20250514"},
        ... ])
        >>> response = router.query(messages)
        >>> print(response["propensity"])  # 0.333...
        >>> print(response["selected_model"])  # Name of selected model
    """

    def __init__(
        self,
        *,
        config_class: type = PropensityRouletteRouterConfig,
        fallback_model_name: str | None = None,
        **kwargs,
    ):
        """Initialize PropensityRouletteRouter.

        Args:
            config_class: Configuration dataclass.
            fallback_model_name: Model name for context window fallback.
            **kwargs: Configuration parameters.
        """
        self.config = config_class(**kwargs)
        super().__init__(
            model_kwargs=self.config.model_kwargs,
            fallback_model_name=fallback_model_name,
        )

        # Track last selection for query() to use
        self._last_selected_model: Model | None = None
        self._last_propensity: float = 0.0

        # Optional selection history
        self._selection_history: list[dict[str, Any]] = []

    @property
    def propensity(self) -> float:
        """Get the uniform propensity (1/n) for model selection."""
        return 1.0 / len(self.models) if self.models else 0.0

    def select_model(self) -> Model:
        """Randomly select a model and record propensity.

        Returns:
            Randomly chosen model instance.
        """
        selected = random.choice(self.models)
        self._last_selected_model = selected
        self._last_propensity = self.propensity

        if self.config.record_selection_history:
            self._selection_history.append(
                {
                    "step": self.n_calls,
                    "selected_model": selected.config.model_name,
                    "propensity": self._last_propensity,
                    "n_models": len(self.models),
                }
            )

        return selected

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        """Execute query and add propensity to response.

        Extends BaseRouter.query() to include propensity information
        in the response, which will be saved in the trajectory.

        Args:
            messages: List of message dicts.
            **kwargs: Additional query parameters.

        Returns:
            Response dict with additional fields:
            - propensity: Selection probability
            - selected_model: Name of the selected model
            - router_info: Dict with router metadata
        """
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
            "n_models": len(self.models),
            "model_pool": [m.config.model_name for m in self.models],
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


# Convenience factory function
def create_propensity_router(
    model_names: list[str],
    **model_kwargs: Any,
) -> PropensityRouletteRouter:
    """Create a PropensityRouletteRouter from model names.

    Args:
        model_names: List of model name strings.
        **model_kwargs: Additional kwargs to pass to each model config.

    Returns:
        PropensityRouletteRouter instance.

    Example:
        >>> router = create_propensity_router([
        ...     "claude-3-5-haiku-latest",
        ...     "claude-sonnet-4-5-20250929",
        ... ])
    """
    configs = []
    for name in model_names:
        config = {"model_name": name}
        config.update(model_kwargs)
        configs.append(config)

    return PropensityRouletteRouter(model_kwargs=configs)
