"""Research module for multi-turn agentic model router.

This module provides tools for:
- Data split management (train/val/test)
- Trajectory collection and parsing
- Learned router implementation

Install with: pip install mtrouter[research]
"""

from miniagenticrouter.research.data.split import DataSplit
from miniagenticrouter.research.trajectory.parser import (
    StepFeature,
    TrajectoryFeature,
    TrajectoryParser,
)
from miniagenticrouter.research.collection.modes import (
    BaselineMode,
    ClusterMode,
    CollectionMode,
    HeuristicMode,
    LearnedMode,
    LLMRouterMode,
    MixedMode,
    PerModelRidgeMode,
    PerModelXGBoostMode,
    RouletteMode,
)
from miniagenticrouter.research.collection.collector import (
    CollectionConfig,
    CollectionResult,
    Collector,
)
from miniagenticrouter.research.routers.policies import (
    SelectionPolicy,
    GreedyPolicy,
    EpsilonGreedyPolicy,
    SoftmaxPolicy,
    UCBPolicy,
    ThompsonSamplingPolicy,
)
from miniagenticrouter.research.routers.learned import (
    QFunction,
    LearnedRouter,
)

__all__ = [
    # Data
    "DataSplit",
    # Trajectory
    "StepFeature",
    "TrajectoryFeature",
    "TrajectoryParser",
    # Collection
    "CollectionMode",
    "BaselineMode",
    "ClusterMode",
    "HeuristicMode",
    "LearnedMode",
    "LLMRouterMode",
    "MixedMode",
    "PerModelRidgeMode",
    "PerModelXGBoostMode",
    "RouletteMode",
    "CollectionConfig",
    "CollectionResult",
    "Collector",
    # Policies
    "SelectionPolicy",
    "GreedyPolicy",
    "EpsilonGreedyPolicy",
    "SoftmaxPolicy",
    "UCBPolicy",
    "ThompsonSamplingPolicy",
    # Router
    "QFunction",
    "LearnedRouter",
]
