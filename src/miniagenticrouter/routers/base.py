"""Base router class providing common functionality for all routers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import litellm

from miniagenticrouter.models import get_model

if TYPE_CHECKING:
    from miniagenticrouter import Model

logger = logging.getLogger(__name__)


class BaseRouter(ABC):
    """Abstract base class for model routers.

    Provides common functionality for managing multiple models and
    aggregating statistics. Subclasses must implement `select_model()`
    to define the selection strategy.

    Attributes:
        models: List of managed model instances
        config: Router configuration object

    Example:
        class MyRouter(BaseRouter):
            def select_model(self) -> Model:
                # Custom selection logic
                return self.models[0]
    """

    def __init__(
        self,
        *,
        model_kwargs: list[dict],
        fallback_model_name: str | None = None,
        **kwargs,
    ):
        """Initialize the router with a list of model configurations.

        Args:
            model_kwargs: List of configuration dicts for each model.
                Each dict is passed to `get_model()` to create a model instance.
            fallback_model_name: Model name to use when ContextWindowExceededError
                occurs. If None, the error is raised without fallback.
            **kwargs: Additional configuration parameters stored in _extra_config
        """
        self.models: list[Model] = [get_model(config=cfg) for cfg in model_kwargs]
        self._extra_config = kwargs
        self._fallback_model_name = fallback_model_name
        self._fallback_model: Model | None = None  # Lazy loaded
        self._fallback_count = 0

    def _get_fallback_model(self) -> Model | None:
        """Get or create the fallback model (lazy loading).

        Returns:
            Fallback model instance, or None if not configured.
        """
        if self._fallback_model_name is None:
            return None
        if self._fallback_model is None:
            self._fallback_model = get_model(
                config={"model_name": self._fallback_model_name}
            )
        return self._fallback_model

    @property
    def cost(self) -> float:
        """Total cost across all managed models including fallback."""
        total = sum(model.cost for model in self.models)
        if self._fallback_model is not None:
            total += self._fallback_model.cost
        return total

    @property
    def n_calls(self) -> int:
        """Total number of calls across all managed models including fallback."""
        total = sum(model.n_calls for model in self.models)
        if self._fallback_model is not None:
            total += self._fallback_model.n_calls
        return total

    @abstractmethod
    def select_model(self) -> Model:
        """Select the next model to use.

        Subclasses must implement this method to define the routing strategy.

        Returns:
            The selected model instance
        """
        ...

    def query(self, *args, **kwargs) -> dict:
        """Execute a query using the selected model.

        Selects a model using `select_model()`, executes the query,
        and adds metadata about the selection to the response.

        If ContextWindowExceededError occurs and fallback_model is configured,
        retries with the fallback model.

        Args:
            *args: Positional arguments passed to the model's query method
            **kwargs: Keyword arguments passed to the model's query method

        Returns:
            The model's response dict with additional router metadata:
                - model_name: Name of the model that handled the query
                - router_type: Type of router that made the selection
                - fallback_from: (optional) Original model if fallback occurred
        """
        model = self.select_model()
        original_model_name = model.config.model_name
        used_fallback = False

        try:
            response = model.query(*args, **kwargs)
        except litellm.exceptions.ContextWindowExceededError:
            fallback = self._get_fallback_model()
            if fallback is None:
                raise  # No fallback configured

            logger.warning(
                f"ContextWindowExceeded for {original_model_name}, "
                f"falling back to {self._fallback_model_name}"
            )
            response = fallback.query(*args, **kwargs)
            used_fallback = True
            self._fallback_count += 1
            model = fallback

        response["model_name"] = model.config.model_name
        response["router_type"] = self.__class__.__name__
        if used_fallback:
            response["fallback_from"] = original_model_name

        return response

    def get_template_vars(self) -> dict[str, Any]:
        """Get template variables for Jinja2 rendering.

        Returns:
            Dict containing:
                - n_model_calls: Total calls across all models
                - model_cost: Total cost across all models
                - router_type: Name of the router class
                - num_models: Number of managed models
                - fallback_count: Number of times fallback was used
        """
        return {
            "n_model_calls": self.n_calls,
            "model_cost": self.cost,
            "router_type": self.__class__.__name__,
            "num_models": len(self.models),
            "fallback_count": self._fallback_count,
        }
