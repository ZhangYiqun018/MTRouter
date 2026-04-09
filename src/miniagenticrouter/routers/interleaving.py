"""Interleaving router - alternates between models in sequence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from miniagenticrouter.routers.base import BaseRouter

if TYPE_CHECKING:
    from miniagenticrouter import Model


@dataclass
class InterleavingRouterConfig:
    """Configuration for InterleavingRouter.

    Attributes:
        model_kwargs: List of model configuration dicts
        sequence: Custom sequence of model indices. If None, uses round-robin.
            Example: [0, 0, 1] means model 0 twice, then model 1, then repeat.
        model_name: Name identifier for this router instance (for compatibility)
    """

    model_kwargs: list[dict]
    sequence: list[int] | None = None
    model_name: str = "interleaving"


class InterleavingRouter(BaseRouter):
    """Router that alternates between models in a defined sequence.

    This router implements sequential model selection, either round-robin
    or following a custom sequence pattern.

    Modes:
        - Round-robin (sequence=None): Cycles through models in order
        - Custom sequence: Follows the specified index pattern

    Useful for:
        - Controlled A/B testing with specific ratios
        - Alternating between models for comparison
        - Implementing custom rotation patterns

    Example:
        # Round-robin: model0, model1, model0, model1, ...
        router = InterleavingRouter(model_kwargs=[
            {"model_name": "anthropic/claude-sonnet-4-5-20250929"},
            {"model_name": "openai/gpt-4o"},
        ])

        # Custom sequence: model0, model0, model1, model0, model0, model1, ...
        router = InterleavingRouter(
            model_kwargs=[...],
            sequence=[0, 0, 1]
        )
    """

    def __init__(
        self,
        *,
        config_class: type = InterleavingRouterConfig,
        fallback_model_name: str | None = None,
        **kwargs,
    ):
        """Initialize the InterleavingRouter.

        Args:
            config_class: Configuration dataclass (default: InterleavingRouterConfig)
            fallback_model_name: Model name for context window fallback.
            **kwargs: Configuration parameters passed to config_class
        """
        self.config = config_class(**kwargs)
        super().__init__(
            model_kwargs=self.config.model_kwargs,
            fallback_model_name=fallback_model_name,
        )

    def select_model(self) -> Model:
        """Select the next model in the sequence.

        Uses round-robin selection if no custom sequence is defined,
        otherwise follows the configured sequence pattern.

        Returns:
            The next model in the sequence
        """
        if self.config.sequence is None:
            # Round-robin: cycle through models in order
            idx = self.n_calls % len(self.models)
        else:
            # Custom sequence: follow the defined pattern
            idx = self.config.sequence[self.n_calls % len(self.config.sequence)]
        return self.models[idx]

    def get_template_vars(self) -> dict:
        """Get template variables including config details.

        Returns:
            Dict with router stats and configuration
        """
        return asdict(self.config) | super().get_template_vars()


# Backward compatibility aliases
InterleavingModel = InterleavingRouter
InterleavingModelConfig = InterleavingRouterConfig
