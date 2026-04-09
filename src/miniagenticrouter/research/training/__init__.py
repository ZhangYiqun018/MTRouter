"""Training utilities for Q-function learning.

This module provides encoders and training utilities for learned routers.

This module requires additional dependencies:
    pip install -e '.[research]'
"""

from .dataset import TrajectoryDataset, collate_fn, collate_fn_precomputed
from .encoders import HistoryEncoder, HistoryEncoderConfig, PrecomputeConfig
from .model_encoder import ModelEncoder, ModelEncoderConfig
from .q_network import (
    NeuralQFunction,
    NeuralQFunctionConfig,
    QNetwork,
    QNetworkConfig,
)

__all__ = [
    "collate_fn",
    "collate_fn_precomputed",
    "HistoryEncoder",
    "HistoryEncoderConfig",
    "ModelEncoder",
    "ModelEncoderConfig",
    "NeuralQFunction",
    "NeuralQFunctionConfig",
    "PrecomputeConfig",
    "QNetwork",
    "QNetworkConfig",
    "TrajectoryDataset",
]
