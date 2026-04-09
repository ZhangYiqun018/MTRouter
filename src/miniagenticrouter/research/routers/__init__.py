"""Learned router implementations."""

from miniagenticrouter.research.routers.policies import (
    EpsilonGreedyPolicy,
    GreedyPolicy,
    SelectionPolicy,
    SoftmaxPolicy,
    ThompsonSamplingPolicy,
    UCBPolicy,
)
from miniagenticrouter.research.routers.learned import (
    LearnedRouter,
    LearnedRouterConfig,
    QFunction,
)
from miniagenticrouter.research.routers.per_model_ridge_q import (
    BatchedPerModelRidgeQFunction,
    PerModelRidgeQFunction,
)
from miniagenticrouter.research.routers.cluster_q import (
    BatchedClusterQFunction,
    ClusterQFunction,
)
from miniagenticrouter.research.routers.per_model_xgboost_q import (
    BatchedPerModelXGBoostQFunction,
    PerModelXGBoostQFunction,
)

__all__ = [
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
    "LearnedRouterConfig",
    # Per-model Ridge Q-functions
    "PerModelRidgeQFunction",
    "BatchedPerModelRidgeQFunction",
    # Cluster Q-functions
    "ClusterQFunction",
    "BatchedClusterQFunction",
    # Per-model XGBoost Q-functions
    "PerModelXGBoostQFunction",
    "BatchedPerModelXGBoostQFunction",
]
