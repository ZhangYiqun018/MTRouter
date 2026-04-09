"""Model routers for multi-model selection strategies.

This module provides router classes that manage multiple models and implement
different selection strategies (random, round-robin, weighted, etc.).

Usage:
    from miniagenticrouter.routers import get_router, RouletteRouter, InterleavingRouter

    # Using factory function
    router = get_router({
        "router_class": "roulette",
        "model_kwargs": [
            {"model_name": "anthropic/claude-sonnet-4-5-20250929"},
            {"model_name": "openai/gpt-4o"},
        ]
    })

    # Direct instantiation
    router = RouletteRouter(model_kwargs=[...])
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from miniagenticrouter import Model


class Router(Protocol):
    """Protocol for model routers.

    A router manages multiple models and implements a selection strategy
    for choosing which model to use for each query.
    """

    config: Any
    models: list[Model]

    @property
    def cost(self) -> float:
        """Total cost across all managed models."""
        ...

    @property
    def n_calls(self) -> int:
        """Total number of calls across all managed models."""
        ...

    def select_model(self) -> Model:
        """Select the next model to use based on the routing strategy."""
        ...

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        """Execute a query using the selected model."""
        ...

    def get_template_vars(self) -> dict[str, Any]:
        """Get template variables for Jinja2 rendering."""
        ...


# Router class mapping for factory function
_ROUTER_CLASS_MAPPING = {
    "roulette": "miniagenticrouter.routers.roulette.RouletteRouter",
    "interleaving": "miniagenticrouter.routers.interleaving.InterleavingRouter",
    "heuristic": "miniagenticrouter.routers.heuristic.HeuristicRouter",
}


def get_router_class(router_class: str) -> type:
    """Get a router class by name or full path.

    Args:
        router_class: Short name (e.g., "roulette") or full path
            (e.g., "miniagenticrouter.routers.roulette.RouletteRouter")

    Returns:
        The router class

    Raises:
        ValueError: If the router class cannot be found
    """
    full_path = _ROUTER_CLASS_MAPPING.get(router_class, router_class)
    try:
        module_name, class_name = full_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as e:
        msg = f"Unknown router class: {router_class} (resolved to {full_path}, available: {list(_ROUTER_CLASS_MAPPING.keys())})"
        raise ValueError(msg) from e


def get_router(config: dict) -> Router:
    """Factory function to create a router instance.

    Args:
        config: Router configuration dict containing:
            - router_class: Router type (e.g., "roulette", "interleaving")
            - model_kwargs: List of model configuration dicts
            - Other router-specific parameters

    Returns:
        A router instance

    Example:
        router = get_router({
            "router_class": "roulette",
            "model_kwargs": [
                {"model_name": "anthropic/claude-sonnet-4-5-20250929"},
                {"model_name": "openai/gpt-4o"},
            ]
        })
    """
    config = config.copy()
    router_class_name = config.pop("router_class", "roulette")
    router_class = get_router_class(router_class_name)
    return router_class(**config)


# Convenience imports
from miniagenticrouter.routers.heuristic import HeuristicRouter, HeuristicRouterConfig
from miniagenticrouter.routers.interleaving import InterleavingRouter, InterleavingRouterConfig
from miniagenticrouter.routers.roulette import RouletteRouter, RouletteRouterConfig

__all__ = [
    "Router",
    "get_router",
    "get_router_class",
    "RouletteRouter",
    "RouletteRouterConfig",
    "InterleavingRouter",
    "InterleavingRouterConfig",
    "HeuristicRouter",
    "HeuristicRouterConfig",
]
