"""Roulette router - randomly selects a model for each query."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from miniagenticrouter.routers.base import BaseRouter

if TYPE_CHECKING:
    from miniagenticrouter import Model


@dataclass
class RouletteRouterConfig:
    """Configuration for RouletteRouter.

    Attributes:
        model_kwargs: List of model configuration dicts
        model_name: Name identifier for this router instance (for compatibility)
    """

    model_kwargs: list[dict]
    model_name: str = "roulette"


class RouletteRouter(BaseRouter):
    """Router that randomly selects a model for each query.

    This router implements a simple random selection strategy where
    each query is handled by a randomly chosen model from the pool.

    Useful for:
        - A/B testing with equal distribution
        - Load balancing across equivalent models
        - Exploring different model behaviors

    Example:
        router = RouletteRouter(model_kwargs=[
            {"model_name": "anthropic/claude-sonnet-4-5-20250929"},
            {"model_name": "openai/gpt-4o"},
        ])

        # Each call randomly selects one of the models
        response = router.query(messages)
    """

    def __init__(
        self,
        *,
        config_class: type = RouletteRouterConfig,
        fallback_model_name: str | None = None,
        **kwargs,
    ):
        """Initialize the RouletteRouter.

        Args:
            config_class: Configuration dataclass (default: RouletteRouterConfig)
            fallback_model_name: Model name for context window fallback.
            **kwargs: Configuration parameters passed to config_class
        """
        self.config = config_class(**kwargs)
        super().__init__(
            model_kwargs=self.config.model_kwargs,
            fallback_model_name=fallback_model_name,
        )

    def select_model(self) -> Model:
        """Randomly select a model from the pool.

        Returns:
            A randomly chosen model instance
        """
        return random.choice(self.models)

    def get_template_vars(self) -> dict:
        """Get template variables including config details.

        Returns:
            Dict with router stats and configuration
        """
        return asdict(self.config) | super().get_template_vars()


# Backward compatibility aliases
RouletteModel = RouletteRouter
RouletteModelConfig = RouletteRouterConfig
