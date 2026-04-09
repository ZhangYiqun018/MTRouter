"""Backward compatibility module for router classes.

This module re-exports the router classes from their new location
in `miniagenticrouter.routers` to maintain backward compatibility with
existing code and configurations.

New code should import directly from `miniagenticrouter.routers`:
    from miniagenticrouter.routers import RouletteRouter, InterleavingRouter

Deprecated imports (still work but not recommended):
    from miniagenticrouter.models.extra.roulette import RouletteModel, InterleavingModel
"""

# Re-export from new location for backward compatibility
from miniagenticrouter.routers.interleaving import (
    InterleavingRouter as InterleavingModel,
    InterleavingRouterConfig as InterleavingModelConfig,
)
from miniagenticrouter.routers.roulette import (
    RouletteRouter as RouletteModel,
    RouletteRouterConfig as RouletteModelConfig,
)

__all__ = [
    "RouletteModel",
    "RouletteModelConfig",
    "InterleavingModel",
    "InterleavingModelConfig",
]
