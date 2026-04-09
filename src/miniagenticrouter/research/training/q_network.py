"""Q-Network and NeuralQFunction implementation.

This module provides the Q-function scorer that combines HistoryEncoder
and ModelEncoder to estimate Q(x, a) for learned routing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .encoders import HistoryEncoder, HistoryEncoderConfig
from .model_encoder import ModelEncoder, ModelEncoderConfig
from miniagenticrouter.research.routers.learned import QFunction, validate_model_names


@dataclass
class QNetworkConfig:
    """Configuration for QNetwork.

    Attributes:
        state_dim: HistoryEncoder output dimension (auto-detected if None).
        model_dim: ModelEncoder output dimension (auto-detected if None).
        hidden_dims: Optional per-layer hidden dimensions.
        dropout: Dropout probability applied after ReLU (0 disables).
        hidden_dim: Hidden layer dimension (used if hidden_dims is None).
        n_layers: Number of hidden layers (used if hidden_dims is None).
    """

    state_dim: int | None = None  # Auto-detect from HistoryEncoder
    model_dim: int | None = None  # Auto-detect from ModelEncoder
    hidden_dims: list[int] | None = None  # If set, overrides hidden_dim/n_layers
    dropout: float = 0.0
    hidden_dim: int = 256  # Default hidden layer dim
    n_layers: int = 2  # Default number of hidden layers


class QNetwork(nn.Module):
    """Single-head Q-value scoring network.

    Takes concatenated state and model embeddings and outputs Q-value directly.

    Architecture:
        concat(z_x, z_m) -> [Linear -> ReLU] * n_layers -> backbone_out
        backbone_out -> head -> Q-value

    Example:
        >>> net = QNetwork(state_dim=1024, model_dim=64)
        >>> z_x = torch.randn(4, 1024)
        >>> z_m = torch.randn(4, 64)
        >>> q_values = net(z_x, z_m)
        >>> q_values.shape
        torch.Size([4, 1])
    """

    def __init__(
        self,
        state_dim: int,
        model_dim: int,
        config: QNetworkConfig | None = None,
    ):
        """Initialize QNetwork.

        Args:
            state_dim: HistoryEncoder output dimension.
            model_dim: ModelEncoder output dimension.
            config: Network configuration. Uses defaults if None.
        """
        super().__init__()
        self.config = config or QNetworkConfig()
        self.state_dim = state_dim
        self.model_dim = model_dim

        # Build shared backbone
        backbone_layers: list[nn.Module] = []
        input_dim = state_dim + model_dim

        hidden_dims = self.config.hidden_dims
        if hidden_dims is None:
            hidden_dims = [self.config.hidden_dim] * self.config.n_layers

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            backbone_layers.append(nn.Linear(prev_dim, hidden_dim))
            backbone_layers.append(nn.ReLU())
            if self.config.dropout and self.config.dropout > 0:
                backbone_layers.append(nn.Dropout(self.config.dropout))
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*backbone_layers)
        self.backbone_dim = prev_dim

        # Single head: Q-value
        self.head = nn.Linear(prev_dim, 1)

    def forward(
        self, state_embed: torch.Tensor, model_embed: torch.Tensor
    ) -> torch.Tensor:
        """Compute Q-value from state and model embeddings.

        Args:
            state_embed: State embedding, shape (batch, state_dim) or (state_dim,).
            model_embed: Model embedding, shape (batch, model_dim) or (model_dim,).

        Returns:
            Q-value tensor, shape (batch, 1) or (1,).
        """
        combined = torch.cat([state_embed, model_embed], dim=-1)
        backbone_out = self.backbone(combined)
        q_value = self.head(backbone_out)
        return q_value


@dataclass
class NeuralQFunctionConfig:
    """Configuration for NeuralQFunction.

    Attributes:
        history_encoder_config: HistoryEncoder configuration.
        model_encoder_config: ModelEncoder configuration.
        q_network_config: QNetwork configuration.
        freeze_hf_encoder: Whether to freeze HF backbone.
    """

    history_encoder_config: HistoryEncoderConfig | None = None
    model_encoder_config: ModelEncoderConfig | None = None
    q_network_config: QNetworkConfig | None = None
    freeze_hf_encoder: bool = True  # Freeze HF backbone


class NeuralQFunction(QFunction, nn.Module):
    """Single-head neural network based Q-function.

    Combines HistoryEncoder, ModelEncoder, and QNetwork to estimate Q-values directly.

    Architecture:
        messages -> HistoryEncoder -> z_x (encoder_dim)
        model_idx -> ModelEncoder -> z_m (64)
        concat(z_x, z_m) -> QNetwork -> Q-value

    The HistoryEncoder combines task and context into a single formatted
    string "[Task]\n{task}\n\n[Context]\n{context}" before encoding,
    resulting in a unified embedding (no projector needed).

    Freezing strategy:
        - HF Encoder: Frozen (pretrained)
        - ModelEncoder: Trainable
        - QNetwork: Trainable

    Example:
        >>> model_names = ["openai/gpt-5", "deepseek/deepseek-v3.2"]
        >>> q_func = NeuralQFunction(model_names=model_names)
        >>> messages = [{"role": "user", "content": "Hello"}]
        >>> q_values = q_func.predict(messages, model_names)
        >>> q_values.shape
        (2,)
    """

    def __init__(
        self,
        model_names: list[str],
        config: NeuralQFunctionConfig | None = None,
        history_encoder: HistoryEncoder | None = None,
        model_encoder: ModelEncoder | None = None,
        q_network: QNetwork | None = None,
        lambda_: float = 1.0,
    ):
        """Initialize NeuralQFunction.

        Args:
            model_names: List of model names (must match custom_models.yaml keys).
            config: Configuration. Uses defaults if None.
            history_encoder: Pre-built HistoryEncoder (optional).
            model_encoder: Pre-built ModelEncoder (optional).
            q_network: Pre-built QNetwork (optional).
            lambda_: Cost penalty coefficient for Q = score - lambda * cost.
        """
        QFunction.__init__(self)
        nn.Module.__init__(self)

        self.config = config or NeuralQFunctionConfig()
        self.model_names = list(model_names)
        self.name_to_idx = {name: i for i, name in enumerate(model_names)}
        self.lambda_ = lambda_

        # Initialize encoders first to get their output dimensions
        self.history_encoder = history_encoder or HistoryEncoder(
            self.config.history_encoder_config
        )
        self.model_encoder = model_encoder or ModelEncoder(
            model_names=model_names,
            config=self.config.model_encoder_config,
        )

        # Auto-detect dimensions from encoders
        state_dim = self.history_encoder.output_dim
        model_dim = self.model_encoder.output_dim

        # Initialize Q-network with auto-detected dimensions
        self.q_network = q_network or QNetwork(
            state_dim=state_dim,
            model_dim=model_dim,
            config=self.config.q_network_config,
        )

        # Freeze HF backbone (only when using HF backend)
        if self.config.freeze_hf_encoder and self.history_encoder.encoder is not None:
            for param in self.history_encoder.encoder.parameters():
                param.requires_grad = False

    def set_lambda(self, lambda_: float) -> None:
        """Set the cost penalty coefficient.

        This allows dynamic adjustment of the score-cost tradeoff
        at inference time without retraining.

        Args:
            lambda_: New cost penalty coefficient.
        """
        self.lambda_ = lambda_

    def predict(
        self,
        history: str | list[dict],
        available_models: list[str],
    ) -> np.ndarray:
        """Predict Q-values for each available model.

        Implements QFunction interface for use with LearnedRouter.
        Uses vectorized computation for efficiency.

        Args:
            history: Conversation history (messages list).
            available_models: List of model names to score.

        Returns:
            Q-value array, shape=(len(available_models),).

        Raises:
            ValueError: If history is a string (not supported).
        """
        self.eval()
        with torch.no_grad():
            if isinstance(history, str):
                raise ValueError("Pass messages list, not string")

            # Encode history once
            z_x = self.history_encoder(messages=history)

            # Ensure z_x is on the same device as model (vLLM returns CPU tensors)
            device = self._get_device()
            z_x = z_x.to(device)

            # Collect valid model indices and their positions
            indices = []
            valid_positions = []
            for pos, model_name in enumerate(available_models):
                idx = self.name_to_idx.get(model_name, -1)
                if idx >= 0:
                    indices.append(idx)
                    valid_positions.append(pos)

            # Initialize result with -inf for invalid models
            q_values = np.full(len(available_models), float("-inf"))

            if not indices:
                return q_values

            # Vectorized computation for all valid models
            z_m = self.model_encoder.forward_batch(indices)
            z_x_expanded = z_x.unsqueeze(0).expand(len(indices), -1)
            q_valid = self.q_network(z_x_expanded, z_m).squeeze(-1)

            # Fill valid positions
            q_values[valid_positions] = q_valid.cpu().numpy()
            return q_values

    def predict_decomposed(
        self,
        history: str | list[dict],
        available_models: list[str],
    ) -> dict[str, np.ndarray]:
        """Predict Q-values with decomposed output format.

        Note: With single-head architecture, cost is always 0 and score equals q_values.
        This method is kept for API compatibility.

        Args:
            history: Conversation history (messages list).
            available_models: List of model names to score.

        Returns:
            Dict with 'score', 'cost', 'q_values' arrays.
            cost is always 0, score equals q_values.
        """
        self.eval()
        with torch.no_grad():
            if isinstance(history, str):
                raise ValueError("Pass messages list, not string")

            z_x = self.history_encoder(messages=history)

            # Ensure z_x is on the same device as model (vLLM returns CPU tensors)
            device = self._get_device()
            z_x = z_x.to(device)

            indices = []
            valid_positions = []
            for pos, model_name in enumerate(available_models):
                idx = self.name_to_idx.get(model_name, -1)
                if idx >= 0:
                    indices.append(idx)
                    valid_positions.append(pos)

            n_models = len(available_models)
            q_values = np.full(n_models, float("-inf"))

            if not indices:
                return {"score": q_values, "cost": np.zeros(n_models), "q_values": q_values}

            z_m = self.model_encoder.forward_batch(indices)
            z_x_expanded = z_x.unsqueeze(0).expand(len(indices), -1)
            q_valid = self.q_network(z_x_expanded, z_m).squeeze(-1)

            q_values[valid_positions] = q_valid.cpu().numpy()

            # score equals q_values, cost is always 0
            return {"score": q_values.copy(), "cost": np.zeros(n_models), "q_values": q_values}

    def forward(
        self,
        messages_batch: list[list[dict]],
        model_indices: list[int],
    ) -> torch.Tensor:
        """Batch forward for training (online mode).

        Args:
            messages_batch: List of message lists.
            model_indices: List of model indices.

        Returns:
            Q-values tensor, shape (batch_size,).
        """
        z_x = self.history_encoder.forward_batch(messages_batch)
        z_m = self.model_encoder.forward_batch(model_indices)
        q_values = self.q_network(z_x, z_m)
        return q_values.squeeze(-1)

    def forward_precomputed(
        self,
        embeddings: torch.Tensor,
        model_indices: list[int],
    ) -> torch.Tensor:
        """Batch forward for training with precomputed embeddings.

        This method skips the HuggingFace encoder and only runs:
        1. ModelEncoder (trainable)
        2. QNetwork (trainable)

        Since task+context are now combined before encoding, embeddings
        are already unified (no projector needed).

        Args:
            embeddings: Precomputed unified embeddings, shape (B, encoder_dim).
            model_indices: List of model indices.

        Returns:
            Q-values tensor, shape (batch_size,).
        """
        # Move embeddings to device if needed
        device = self._get_device()
        z_x = embeddings.to(device)

        # Model encoding and Q-network (trainable)
        z_m = self.model_encoder.forward_batch(model_indices)  # (B, model_dim)
        q_values = self.q_network(z_x, z_m)  # (B, 1)

        return q_values.squeeze(-1)  # (B,)

    def _get_device(self) -> torch.device:
        """Get the device where the model is located."""
        return next(self.model_encoder.parameters()).device

    def forward_all_models(self, messages: list[dict]) -> torch.Tensor:
        """Compute Q-values for all models given a state.

        Args:
            messages: Single message list.

        Returns:
            Q-values for all models, shape (n_models,).
        """
        z_x = self.history_encoder(messages=messages)

        # Ensure z_x is on the same device as model (vLLM returns CPU tensors)
        device = self._get_device()
        z_x = z_x.to(device)

        z_m_all = self.model_encoder.forward_all()

        n_models = len(self.model_names)
        z_x_expanded = z_x.unsqueeze(0).expand(n_models, -1)

        q_values = self.q_network(z_x_expanded, z_m_all).squeeze(-1)
        return q_values

    def forward_all_models_decomposed(
        self, messages: list[dict]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Q-values for all models with decomposed output format.

        Note: With single-head architecture, cost is always 0 and score equals q_values.
        This method is kept for API compatibility.

        Args:
            messages: Single message list.

        Returns:
            Tuple of (scores, costs), each shape (n_models,).
            cost is always 0, scores equals q_values.
        """
        z_x = self.history_encoder(messages=messages)

        # Ensure z_x is on the same device as model (vLLM returns CPU tensors)
        device = self._get_device()
        z_x = z_x.to(device)

        z_m_all = self.model_encoder.forward_all()

        n_models = len(self.model_names)
        z_x_expanded = z_x.unsqueeze(0).expand(n_models, -1)

        q_values = self.q_network(z_x_expanded, z_m_all).squeeze(-1)
        zeros = torch.zeros_like(q_values)
        return q_values, zeros  # scores=q_values, costs=0

    def get_regularization_loss(self) -> torch.Tensor:
        """Get total regularization loss.

        Returns:
            Scalar tensor with regularization loss.
        """
        return self.model_encoder.get_residual_l2_loss()

    def save(self, path: Path | str) -> None:
        """Save model weights and configuration.

        Only saves trainable parameters (model_encoder, q_network).
        Frozen HF encoder weights are NOT saved to reduce checkpoint size
        and avoid meta tensor issues when loading.

        Also saves configuration (including max_tokens, etc.) for consistent
        inference loading.

        Args:
            path: Path to save weights file.
        """
        # Only save trainable parameters to avoid meta tensor issues
        trainable_state = {
            k: v for k, v in self.state_dict().items()
            if not k.startswith("history_encoder.encoder.")
        }

        # Serialize config (handle None sub-configs)
        config_dict = {
            "history_encoder_config": asdict(self.config.history_encoder_config)
                if self.config.history_encoder_config else None,
            "model_encoder_config": asdict(self.config.model_encoder_config)
                if self.config.model_encoder_config else None,
            "q_network_config": asdict(self.config.q_network_config)
                if self.config.q_network_config else None,
            "freeze_hf_encoder": self.config.freeze_hf_encoder,
        }

        torch.save({
            "state_dict": trainable_state,
            "model_names": self.model_names,
            "config": config_dict,
            "lambda_": self.lambda_,
        }, path)

    def load(self, path: Path | str) -> None:
        """Load model weights and lambda.

        Supports both old format (full state_dict) and new format (trainable only).

        Args:
            path: Path to weights file.
        """
        def _load_state_dict(state_dict: dict[str, Any], *, strict: bool) -> None:
            # Some environments may initialize modules on the 'meta' device
            # (e.g., via lazy loading). In that case, a normal load_state_dict()
            # is a no-op; `assign=True` replaces parameters/buffers instead.
            try:
                self.load_state_dict(state_dict, strict=strict, assign=True)
            except TypeError:
                # Older PyTorch without `assign` argument.
                self.load_state_dict(state_dict, strict=strict)

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")

        # Handle both old format (just state_dict) and new format (dict with metadata)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

            # Check for old dual-head architecture (incompatible with single-head)
            has_dual_heads = any("q_network.score_head." in k or "q_network.cost_head." in k for k in state_dict.keys())
            has_single_head = any("q_network.head." in k for k in state_dict.keys())
            if has_dual_heads and not has_single_head:
                raise ValueError(
                    f"Checkpoint '{path}' uses old dual-head architecture which is incompatible "
                    "with the current single-head architecture. Please retrain the model using the "
                    "updated training script."
                )

            # New format: only trainable params, use strict=False
            if "model_names" in checkpoint:
                validate_model_names(
                    checkpoint["model_names"],
                    self.model_names,
                    source="checkpoint",
                )
            # Load lambda_ if present
            if "lambda_" in checkpoint:
                self.lambda_ = checkpoint["lambda_"]
            _load_state_dict(state_dict, strict=False)
        else:
            # Old format: full state_dict - reject as incompatible
            raise ValueError(
                f"Checkpoint '{path}' uses legacy format which is incompatible "
                "with the current dual-head architecture. Please retrain the model using the "
                "updated training script."
            )

    @classmethod
    def load_config_from_checkpoint(
        cls,
        path: Path | str,
    ) -> tuple[list[str], NeuralQFunctionConfig, float]:
        """Load model_names, config, and lambda from checkpoint without loading weights.

        This is useful for creating a NeuralQFunction with the same configuration
        as the checkpoint before loading weights.

        Args:
            path: Path to checkpoint file.

        Returns:
            Tuple of (model_names, config, lambda_).
            If checkpoint doesn't have config (old format), returns default config.

        Example:
            >>> model_names, config, lambda_ = NeuralQFunction.load_config_from_checkpoint("model.pt")
            >>> q_func = NeuralQFunction(model_names=model_names, config=config, lambda_=lambda_)
            >>> q_func.load("model.pt")
        """
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")

        # Extract model_names
        if isinstance(checkpoint, dict) and "model_names" in checkpoint:
            model_names = checkpoint["model_names"]
        else:
            raise ValueError("Checkpoint does not contain model_names")

        # Extract lambda_
        lambda_ = checkpoint.get("lambda_", 1.0) if isinstance(checkpoint, dict) else 1.0

        # Extract config (with backward compatibility for old checkpoints)
        if isinstance(checkpoint, dict) and "config" in checkpoint:
            config_dict = checkpoint["config"]

            # Reconstruct nested dataclass configs
            history_config = None
            if config_dict.get("history_encoder_config"):
                history_config = HistoryEncoderConfig(**config_dict["history_encoder_config"])

            model_config = None
            if config_dict.get("model_encoder_config"):
                model_config = ModelEncoderConfig(**config_dict["model_encoder_config"])

            q_net_config = None
            if config_dict.get("q_network_config"):
                q_net_config = QNetworkConfig(**config_dict["q_network_config"])

            config = NeuralQFunctionConfig(
                history_encoder_config=history_config,
                model_encoder_config=model_config,
                q_network_config=q_net_config,
                freeze_hf_encoder=config_dict.get("freeze_hf_encoder", True),
            )
        else:
            # Old checkpoint without config, use defaults
            import warnings
            warnings.warn(
                f"Checkpoint {path} does not contain config, using defaults. "
                "Re-save the checkpoint to include config."
            )
            config = NeuralQFunctionConfig()

        return model_names, config, lambda_
