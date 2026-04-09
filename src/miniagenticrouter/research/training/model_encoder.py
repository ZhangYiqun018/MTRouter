"""Model encoder for Q-function.

This module provides encoders for converting model attributes
into fixed-dimensional vectors for learned router training.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml


@dataclass
class ModelEncoderConfig:
    """Configuration for ModelEncoder.

    Attributes:
        meta_dim: Number of meta features (5 raw + 3 derived).
        meta_embed_dim: Output dimension of attribute MLP.
        residual_embed_dim: Dimension of per-model residual embedding.
        output_dim: Final output embedding dimension.
        residual_l2_weight: L2 regularization weight for residual embeddings.
        knowledge_cutoff_base: Base date for computing cutoff days.
        custom_models_path: Path to custom_models.yaml (optional).
    """

    # Dimension settings
    meta_dim: int = 8  # 5 raw + 3 derived
    meta_embed_dim: int = 32  # Attribute MLP output dim
    residual_embed_dim: int = 16  # Residual embedding dim
    output_dim: int = 64  # Final output dim

    # Regularization
    residual_l2_weight: float = 0.01  # L2 weight for residual embeddings

    # Date baseline
    knowledge_cutoff_base: str = "2026-01-01"  # Base date for cutoff calculation

    # Model config path
    custom_models_path: str | None = None  # Default: search standard locations


class ModelEncoder(nn.Module):
    """Model encoder with attribute encoding + learnable residual embedding.

    This encoder uses a two-tower design:
    1. Attribute tower: Encodes model metadata (tokens, cost, etc.) via MLP
    2. Residual tower: Per-model learnable embedding (initialized to 0)

    Architecture:
        meta (8) -> log -> z-score -> MLP -> meta_emb (32)
        u_m (16) <- per-model residual embedding (init=0)
        concat(meta_emb, u_m) -> Linear -> LayerNorm -> z_m (64)

    Example:
        >>> encoder = ModelEncoder(model_names=["openai/gpt-4o", "deepseek/deepseek-v3.2"])
        >>> z = encoder(model_name="openai/gpt-4o")
        >>> z.shape
        torch.Size([64])
    """

    # Meta fields used from custom_models.yaml
    META_FIELDS = [
        "max_input_tokens",
        "max_output_tokens",
        "input_cost_per_million",
        "output_cost_per_million",
        "knowledge_cutoff",
    ]

    # Default values for missing fields
    META_DEFAULTS = {
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
        "input_cost_per_million": 1.0,
        "output_cost_per_million": 5.0,
        "knowledge_cutoff": "2024-01-01",
    }

    def __init__(
        self,
        model_names: list[str],
        config: ModelEncoderConfig | None = None,
        custom_models_config: dict[str, Any] | None = None,
    ):
        """Initialize ModelEncoder.

        Args:
            model_names: List of model names to support.
            config: Encoder configuration. Uses defaults if None.
            custom_models_config: Pre-loaded custom_models.yaml dict (optional).
        """
        super().__init__()
        self.config = config or ModelEncoderConfig()
        self.model_names = list(model_names)
        self.n_models = len(model_names)

        # Build name -> index mapping
        self.name_to_idx = {name: i for i, name in enumerate(model_names)}

        # Load model attributes from config
        if custom_models_config is None:
            custom_models_config = self._load_custom_models()
        self.models_config = custom_models_config

        # Extract and preprocess meta features
        self._init_meta_features()

        # Attribute MLP: meta (9) -> meta_emb (32)
        self.meta_mlp = nn.Sequential(
            nn.Linear(self.config.meta_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.config.meta_embed_dim),
        )

        # Residual embedding: per-model, init to 0
        self.residual_embed = nn.Embedding(
            self.n_models,
            self.config.residual_embed_dim,
        )
        nn.init.zeros_(self.residual_embed.weight)

        # Output projection
        combined_dim = self.config.meta_embed_dim + self.config.residual_embed_dim
        self.output_proj = nn.Sequential(
            nn.Linear(combined_dim, self.config.output_dim),
            nn.LayerNorm(self.config.output_dim),
        )

        # Expose output dimension for downstream modules
        self.output_dim = self.config.output_dim

    def _load_custom_models(self) -> dict[str, Any]:
        """Load custom_models.yaml from standard locations.

        Returns:
            Dict of model configs from custom_models.yaml.

        Raises:
            FileNotFoundError: If custom_models.yaml not found.
        """
        # Try standard locations
        search_paths = [
            Path(__file__).parent.parent.parent
            / "config"
            / "models"
            / "custom_models.yaml",
            Path.home() / ".config" / "mini-agentic-router" / "custom_models.yaml",
        ]

        if self.config.custom_models_path:
            search_paths.insert(0, Path(self.config.custom_models_path))

        for path in search_paths:
            if path.exists():
                with open(path) as f:
                    return yaml.safe_load(f).get("models", {})

        raise FileNotFoundError(f"custom_models.yaml not found in {search_paths}")

    def _parse_cutoff_days(self, cutoff_str: str) -> float:
        """Convert knowledge_cutoff date string to days from base date.

        Args:
            cutoff_str: Date string in "YYYY-MM-DD" format.

        Returns:
            Days difference from base date (negative if earlier).
        """
        base = datetime.strptime(self.config.knowledge_cutoff_base, "%Y-%m-%d")
        cutoff = datetime.strptime(cutoff_str, "%Y-%m-%d")
        return float((cutoff - base).days)

    def _extract_raw_meta(self, model_name: str) -> list[float]:
        """Extract raw meta features for a model.

        Args:
            model_name: Model name (e.g., "openai/gpt-4o").

        Returns:
            List of 8 meta features (5 raw + 3 derived).
        """
        cfg = self.models_config.get(model_name, {})

        # Get values with defaults
        max_input = float(
            cfg.get("max_input_tokens", self.META_DEFAULTS["max_input_tokens"])
        )
        max_output = float(
            cfg.get("max_output_tokens", self.META_DEFAULTS["max_output_tokens"])
        )
        input_cost = float(
            cfg.get("input_cost_per_million", self.META_DEFAULTS["input_cost_per_million"])
        )
        output_cost = float(
            cfg.get(
                "output_cost_per_million", self.META_DEFAULTS["output_cost_per_million"]
            )
        )

        # Handle knowledge_cutoff (date -> days)
        cutoff_str = cfg.get("knowledge_cutoff", self.META_DEFAULTS["knowledge_cutoff"])
        cutoff_days = self._parse_cutoff_days(cutoff_str)

        # Raw features (5)
        raw = [max_input, max_output, input_cost, output_cost, cutoff_days]

        # Derived features (3)
        context_total = max_input + max_output
        cost_ratio = output_cost / (input_cost + 1e-6)
        cost_total = input_cost + output_cost

        return raw + [context_total, cost_ratio, cost_total]

    def _init_meta_features(self):
        """Initialize meta features and compute normalization statistics.

        This method:
        1. Extracts raw features for all models
        2. Applies log transform to positive features
        3. Computes mean/std for z-score normalization
        4. Registers statistics as buffers (saved with model weights)
        """
        # Extract all metas
        all_metas = []
        for name in self.model_names:
            meta = self._extract_raw_meta(name)
            all_metas.append(meta)

        all_metas = np.array(all_metas, dtype=np.float32)

        # Log transform (only for non-negative features)
        # Features 0-3: max_input, max_output, input_cost, output_cost (positive)
        # Feature 4: cutoff_days (can be negative, skip log)
        # Features 5-7: derived (positive)
        all_metas_transformed = all_metas.copy()
        positive_cols = [0, 1, 2, 3, 5, 6, 7]
        all_metas_transformed[:, positive_cols] = np.log1p(
            all_metas[:, positive_cols]
        )
        # cutoff_days (col 4) stays as-is

        # Compute mean/std for z-score
        meta_mean = all_metas_transformed.mean(axis=0)
        meta_std = all_metas_transformed.std(axis=0) + 1e-6

        # Register as buffers (saved with model weights)
        self.register_buffer("meta_mean", torch.tensor(meta_mean))
        self.register_buffer("meta_std", torch.tensor(meta_std))

        # Pre-compute normalized metas for all models
        metas_norm = (all_metas_transformed - meta_mean) / meta_std
        self.register_buffer("all_metas", torch.tensor(metas_norm))

    def get_model_idx(self, model_name: str) -> int:
        """Get model index from name.

        Args:
            model_name: Model name.

        Returns:
            Model index, or -1 if not found.
        """
        return self.name_to_idx.get(model_name, -1)

    def forward(
        self,
        model_name: str | None = None,
        model_idx: int | None = None,
    ) -> torch.Tensor:
        """Encode a single model to vector.

        Args:
            model_name: Model name (e.g., "openai/gpt-4o").
            model_idx: Model index (alternative to model_name).

        Returns:
            Model embedding of shape (output_dim,).

        Raises:
            ValueError: If neither model_name nor model_idx provided,
                       or if model_name is unknown.
        """
        if model_idx is None:
            if model_name is None:
                raise ValueError("Either model_name or model_idx required")
            model_idx = self.get_model_idx(model_name)
            if model_idx < 0:
                raise ValueError(f"Unknown model: {model_name}")

        # Get pre-computed normalized meta
        meta = self.all_metas[model_idx]  # (meta_dim,)

        # Attribute encoding
        meta_emb = self.meta_mlp(meta)  # (meta_embed_dim,)

        # Residual embedding
        idx_tensor = torch.tensor([model_idx], device=meta.device)
        u_m = self.residual_embed(idx_tensor).squeeze(0)  # (residual_embed_dim,)

        # Concat and project
        combined = torch.cat([meta_emb, u_m], dim=-1)
        z_m = self.output_proj(combined)

        return z_m

    def forward_batch(self, model_indices: list[int]) -> torch.Tensor:
        """Batch encoding for multiple models.

        Args:
            model_indices: List of model indices.

        Returns:
            Embeddings of shape (batch_size, output_dim).
        """
        indices = torch.tensor(model_indices, device=self.all_metas.device)

        # Get metas for all models in batch
        metas = self.all_metas[indices]  # (batch, meta_dim)

        # Attribute encoding
        meta_embs = self.meta_mlp(metas)  # (batch, meta_embed_dim)

        # Residual embeddings
        u_ms = self.residual_embed(indices)  # (batch, residual_embed_dim)

        # Concat and project
        combined = torch.cat([meta_embs, u_ms], dim=-1)
        z_ms = self.output_proj(combined)

        return z_ms

    def forward_all(self) -> torch.Tensor:
        """Encode all models.

        Returns:
            Embeddings of shape (n_models, output_dim).
        """
        return self.forward_batch(list(range(self.n_models)))

    def get_residual_l2_loss(self) -> torch.Tensor:
        """Get L2 regularization loss for residual embeddings.

        Returns:
            Scalar tensor with L2 loss.
        """
        return self.config.residual_l2_weight * self.residual_embed.weight.pow(2).sum()
