"""Shared router creation utilities for benchmark runners."""

from __future__ import annotations

from typing import Literal

from miniagenticrouter.models import get_model

# Router type for CLI options
RouterType = Literal["none", "roulette", "interleaving"]


def create_model_or_router(
    model_names: str | None,
    router_type: RouterType,
    config: dict,
):
    """Create a model or router based on CLI options.

    Args:
        model_names: Comma-separated model names (e.g., "model1,model2")
        router_type: Type of router to use ("none", "roulette", "interleaving")
        config: Model config from YAML file

    Returns:
        Model or Router instance

    Raises:
        ValueError: If router_type is not "none" but fewer than 2 models provided

    Example:
        # Single model mode
        model = create_model_or_router("openai/gpt-4o", "none", config)

        # Router mode
        router = create_model_or_router(
            "openai/gpt-4o,anthropic/claude-sonnet-4-5-20250929",
            "roulette",
            config
        )
    """
    if router_type == "none" or not model_names:
        # Single model mode
        return get_model(model_names, config)

    # Router mode - parse comma-separated model names
    models = [m.strip() for m in model_names.split(",") if m.strip()]

    if len(models) < 2:
        raise ValueError(f"Router requires at least 2 models, got: {models}")

    # Build model_kwargs list for router
    model_kwargs = []
    for model_name in models:
        model_config = {"model_name": model_name}
        # Apply custom_llm_provider if model has prefix like "openai/"
        if "/" in model_name:
            provider = model_name.split("/")[0]
            model_config["model_kwargs"] = {"custom_llm_provider": provider}
        model_kwargs.append(model_config)

    # Create router
    if router_type == "roulette":
        from miniagenticrouter.routers import RouletteRouter

        return RouletteRouter(model_kwargs=model_kwargs)
    elif router_type == "interleaving":
        from miniagenticrouter.routers import InterleavingRouter

        return InterleavingRouter(model_kwargs=model_kwargs)
    else:
        raise ValueError(f"Unknown router type: {router_type}")
