"""Dataset for Q-function training from trajectory files.

This module provides PyTorch Dataset classes for loading trajectory data
and preparing training samples for the Q-function.

Supports two modes:
1. Online mode (default): Stores raw messages, encodes during forward pass
2. Precomputed mode: Precomputes embeddings with HuggingFace during init for faster training
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from miniagenticrouter.research.trajectory.parser import (
    TrajectoryParser,
    find_trajectory_files,
)

if TYPE_CHECKING:
    from .encoders import PrecomputeConfig
    from .error_detector import AECConfig, ErrorDetector


class TrajectoryDataset(Dataset):
    """Dataset for Q-function training from trajectory files.

    Loads trajectory JSON files and extracts training samples with:
    - messages: Raw message list for HistoryEncoder
    - model_idx: Model index for ModelEncoder
    - target_value: Q-learning target (final_score - lambda * remaining_cost)
    - propensity: Selection probability for importance weighting

    Supports loading from multiple directories for joint training (Stage B).

    Supports two modes:
    1. Online mode (default): Stores raw messages, encodes during forward pass
    2. Precomputed mode: Precomputes embeddings with HuggingFace during init

    Example:
        >>> # Single directory (backward compatible)
        >>> dataset = TrajectoryDataset(
        ...     trajectory_dir="trajectories/roulette_propensity",
        ...     model_names=["openai/gpt-5", "deepseek/deepseek-v3.2"],
        ... )

        >>> # Multiple directories (Stage B joint training)
        >>> dataset = TrajectoryDataset(
        ...     trajectory_dir=[
        ...         "trajectories/roulette_propensity",
        ...         "trajectories/mixed_b1",
        ...     ],
        ...     model_names=["openai/gpt-5", "deepseek/deepseek-v3.2"],
        ... )

        >>> # With precomputation (faster training)
        >>> from miniagenticrouter.research.training.encoders import PrecomputeConfig
        >>> config = PrecomputeConfig(enabled=True, batch_size=256)
        >>> dataset = TrajectoryDataset(
        ...     trajectory_dir="trajectories/roulette_propensity",
        ...     model_names=["openai/gpt-5", "deepseek/deepseek-v3.2"],
        ...     precompute_config=config,
        ... )
    """

    def __init__(
        self,
        trajectory_dir: Path | str | list[Path] | list[str],
        model_names: list[str],
        lambda_: float = 0.0,
        max_samples: int | None = None,
        skip_invalid_models: bool = True,
        precompute_config: PrecomputeConfig | None = None,
        precomputed_path: Path | str | None = None,
        expected_backend: str | None = None,
        error_detector: ErrorDetector | None = None,
        aec_config: AECConfig | None = None,
    ):
        """Initialize TrajectoryDataset.

        Args:
            trajectory_dir: Directory or list of directories containing trajectory files.
            model_names: List of model names for index mapping.
            lambda_: Cost penalty coefficient for target calculation.
            max_samples: Maximum number of samples to load (for debugging).
            skip_invalid_models: Whether to skip samples with unknown model_idx.
            precompute_config: Configuration for embedding precomputation.
                If enabled, embeddings are precomputed during init and model is
                unloaded to free GPU memory before training.
            precomputed_path: Path to precomputed embeddings file (.pt).
                If provided, embeddings are loaded from this file instead of
                computing them. This is faster for repeated training runs.
            expected_backend: Expected embedding backend ("hf" or "vllm").
                If provided, checks that precomputed embeddings use this backend.
                Raises ValueError if there's a mismatch.
            error_detector: Optional error detector for analyzing observation errors.
                If provided, each sample will include error_detected, error_types,
                and error_penalty fields.
            aec_config: Optional AEC configuration for time-dependent error penalties.
                If provided and enabled, each sample's final_score will be adjusted
                based on cumulative error penalties. Requires error_detector.
        """
        # Normalize to list of Paths
        if isinstance(trajectory_dir, (str, Path)):
            self.trajectory_dirs = [Path(trajectory_dir)]
        else:
            self.trajectory_dirs = [Path(d) for d in trajectory_dir]

        self.model_names = model_names
        self.lambda_ = lambda_
        self.skip_invalid_models = skip_invalid_models
        self.error_detector = error_detector
        self.aec_config = aec_config

        self.parser = TrajectoryParser(model_names=model_names)
        self.samples: list[dict[str, Any]] = []

        # Precomputation state
        self.precomputed = False
        self.embeddings: torch.Tensor | None = None  # Single unified embedding
        self.encoder_dim: int | None = None

        # Load trajectories
        self._load_trajectories(max_samples)

        # Determine expected backend (from parameter or precompute_config)
        self.expected_backend = expected_backend
        if self.expected_backend is None and precompute_config is not None:
            self.expected_backend = precompute_config.backend

        # Load precomputed embeddings OR compute them
        if precomputed_path is not None:
            self._load_from_precomputed(Path(precomputed_path))
        elif precompute_config is not None and precompute_config.enabled:
            self._precompute_embeddings(precompute_config)

    def _load_trajectories(self, max_samples: int | None = None) -> None:
        """Load all trajectories and extract samples with cost normalization.

        Two-pass loading:
        1. First pass: Load all trajectories, collect raw costs
        2. Compute per-benchmark mean costs
        3. Second pass: Apply cost normalization

        Args:
            max_samples: Maximum number of samples to load.
        """
        # Collect trajectory files from all directories
        traj_files = []
        for traj_dir in self.trajectory_dirs:
            traj_files.extend(find_trajectory_files(traj_dir))

        # First pass: Load all trajectories
        desc = f"Loading from {len(self.trajectory_dirs)} dir(s)"
        pbar = tqdm(traj_files, desc=desc, dynamic_ncols=True)

        # Track skipped samples
        skipped_count = 0
        unknown_models: dict[str, int] = {}

        # Models to silently skip (not in model_pool but expected, e.g., for test-only)
        silent_skip_models = {"openrouter/auto"}

        for traj_file in pbar:
            try:
                traj = self.parser.parse(traj_file)
                samples = traj.to_training_samples(
                    lambda_=self.lambda_,
                    error_detector=self.error_detector,
                    aec_config=self.aec_config,
                )

                for sample in samples:
                    # Skip samples with unknown model
                    if self.skip_invalid_models and sample["model_idx"] < 0:
                        model_name = sample.get("model_name", "unknown")
                        # Only count as "unknown" if not in silent skip list
                        if model_name not in silent_skip_models:
                            skipped_count += 1
                            unknown_models[model_name] = unknown_models.get(model_name, 0) + 1
                        continue
                    self.samples.append(sample)

                    if max_samples and len(self.samples) >= max_samples:
                        pbar.close()
                        break

            except Exception:
                # Skip invalid trajectory files
                continue

            # Update progress bar with sample count
            pbar.set_postfix({"samples": len(self.samples)})

            if max_samples and len(self.samples) >= max_samples:
                break

        # Warn about skipped samples
        if skipped_count > 0:
            total = len(self.samples) + skipped_count
            pct = 100 * skipped_count / total
            print(f"\n  Skipped {skipped_count}/{total} samples ({pct:.1f}%) with unknown models:")
            for model_name, count in sorted(unknown_models.items(), key=lambda x: -x[1])[:5]:
                print(f"    - {model_name}: {count}")
            if len(unknown_models) > 5:
                print(f"    - ... and {len(unknown_models) - 5} more models")

        # Print per-model sample statistics
        if self.samples:
            model_counts: dict[int, int] = {}
            for sample in self.samples:
                model_idx = sample["model_idx"]
                model_counts[model_idx] = model_counts.get(model_idx, 0) + 1

            print(f"\n  Loaded {len(self.samples)} samples by model:")
            for model_idx in sorted(model_counts.keys()):
                count = model_counts[model_idx]
                pct = 100 * count / len(self.samples)
                model_name = self.model_names[model_idx] if model_idx < len(self.model_names) else f"idx={model_idx}"
                print(f"    [{model_idx}] {model_name}: {count} ({pct:.1f}%)")

        # Compute per-benchmark cost statistics and apply normalization
        if self.samples:
            self._apply_cost_normalization()

    def _compute_cost_stats(self) -> dict[str, float]:
        """Compute mean episode cost for each benchmark.

        Statistics are computed at the episode level (not step level).
        For each episode, we use the remaining_cost at step_idx=0,
        which represents the total episode cost.

        Returns:
            Dict mapping benchmark name to mean episode cost in USD.
        """
        from collections import defaultdict

        # Collect episode costs per benchmark
        episode_costs: dict[str, list[float]] = defaultdict(list)
        seen_episodes: set[tuple[str, str]] = set()

        for sample in self.samples:
            benchmark = sample.get("benchmark", "scienceworld")
            episode_id = sample.get("episode_id", "")
            key = (benchmark, episode_id)

            # Only count step_idx=0 to get total episode cost
            if key not in seen_episodes and sample.get("step_idx", 0) == 0:
                seen_episodes.add(key)
                episode_costs[benchmark].append(sample.get("remaining_cost", 0.0))

        # Compute mean cost per benchmark
        stats = {}
        for benchmark, costs in episode_costs.items():
            if costs:
                mean_cost = sum(costs) / len(costs)
                # Avoid division by zero
                stats[benchmark] = max(mean_cost, 1e-6)
            else:
                stats[benchmark] = 1.0  # Fallback

        return stats

    def _apply_cost_normalization(self) -> None:
        """Apply per-benchmark cost normalization to all samples.

        Normalizes remaining_cost by dividing by the benchmark's mean episode cost.
        This makes costs comparable across benchmarks with different cost scales.
        """
        # Compute statistics
        cost_stats = self._compute_cost_stats()

        # Log statistics
        print("\n  Cost normalization statistics:")
        for benchmark, mean_cost in sorted(cost_stats.items()):
            n_episodes = sum(
                1 for s in self.samples
                if s.get("benchmark") == benchmark and s.get("step_idx", 0) == 0
            )
            print(f"    {benchmark}: mean=${mean_cost:.4f}, episodes={n_episodes}")

        # Apply normalization
        for sample in self.samples:
            benchmark = sample.get("benchmark", "scienceworld")
            mean_cost = cost_stats.get(benchmark, 1.0)
            raw_cost = sample.get("remaining_cost", 0.0)
            sample["normalized_cost"] = raw_cost / mean_cost

    def _precompute_embeddings(self, config: PrecomputeConfig) -> None:
        """Precompute embeddings using configured backend.

        Uses the backend specified in config (HF or vLLM) to ensure
        training/inference consistency.

        This method:
        1. Initializes the appropriate precomputer based on config.backend
        2. Segments and encodes all samples (task+context combined)
        3. Stores embeddings as tensors
        4. Unloads model to free GPU memory
        5. Removes raw messages from samples to save memory

        Args:
            config: Precomputation configuration.
        """
        from .encoders import create_precomputer

        backend_name = config.backend.upper() if hasattr(config, 'backend') else "HF"
        print(f"\n{'='*60}")
        print(f"PRECOMPUTING EMBEDDINGS WITH {backend_name}")
        print(f"{'='*60}")

        # Initialize precomputer based on configured backend
        precomputer = create_precomputer(config)

        # Precompute embeddings (single unified embedding per sample)
        self.embeddings = precomputer.precompute(
            self.samples, show_progress=True
        )
        self.encoder_dim = precomputer.encoder_dim

        # Cleanup to free GPU memory
        precomputer.cleanup()

        # Remove raw messages from samples to save memory
        for sample in self.samples:
            if "messages" in sample:
                del sample["messages"]

        self.precomputed = True
        print(f"{'='*60}")
        print(f"Precomputation complete ({backend_name}). Resources freed.")
        print(f"Embeddings stored: {len(self.samples)} x {self.encoder_dim}")
        print(f"{'='*60}\n")

    def _load_from_precomputed(self, path: Path) -> None:
        """Load embeddings from precomputed file.

        This method:
        1. Loads precomputed embeddings from disk
        2. Matches embeddings to current samples by (episode_id, step_idx)
        3. Raises error if any samples are not found (strict mode)
        4. Removes raw messages from samples to save memory

        Args:
            path: Path to precomputed embeddings file (.pt).

        Raises:
            FileNotFoundError: If the precomputed file doesn't exist.
            ValueError: If any samples are not found in the precomputed file.
        """
        from .encoders import HFPrecomputer

        print(f"\n{'='*60}")
        print("LOADING PRECOMPUTED EMBEDDINGS")
        print(f"{'='*60}")

        # 1. Load precomputed data (requires format_version >= 2)
        data = HFPrecomputer.load(path)
        precomputed_embeddings = data["embeddings"]
        precomputed_ids = data["sample_ids"]  # [(episode_id, step_idx), ...]

        # Check backend consistency if expected_backend is specified
        precomputed_backend = data["config"].get("backend", "hf")
        if self.expected_backend is not None and precomputed_backend != self.expected_backend:
            raise ValueError(
                f"Backend mismatch: precomputed embeddings use '{precomputed_backend}', "
                f"but expected '{self.expected_backend}'.\n"
                f"Precomputation and inference must use the same backend.\n"
                f"Either regenerate embeddings with --backend={self.expected_backend}, "
                f"or update the inference configuration to use backend='{precomputed_backend}'."
            )

        if precomputed_embeddings.ndim != 2:
            raise ValueError(
                f"Expected precomputed embeddings of shape (N, D), got {tuple(precomputed_embeddings.shape)}"
            )
        if len(precomputed_ids) != precomputed_embeddings.shape[0]:
            raise ValueError(
                f"Mismatch between precomputed_ids ({len(precomputed_ids)}) and embeddings rows ({precomputed_embeddings.shape[0]})"
            )
        if int(data["config"].get("encoder_dim", precomputed_embeddings.shape[1])) != precomputed_embeddings.shape[1]:
            raise ValueError(
                f"encoder_dim mismatch: config has {data['config'].get('encoder_dim')}, "
                f"but embeddings have dim {precomputed_embeddings.shape[1]}"
            )

        # 2. Build lookup table: (episode_id, step_idx) -> embedding_idx
        id_to_idx: dict[tuple[str, int], int] = {}
        duplicate_keys: list[tuple[str, int]] = []
        for idx, sample_id in enumerate(precomputed_ids):
            key = (sample_id[0], int(sample_id[1]))
            if key in id_to_idx:
                duplicate_keys.append(key)
                continue
            id_to_idx[key] = idx

        if duplicate_keys:
            raise ValueError(
                "Duplicate (episode_id, step_idx) keys found in precomputed embeddings file.\n"
                f"First 5 duplicates: {duplicate_keys[:5]}\n"
            )

        # 3. Match current samples by (episode_id, step_idx)
        matched_indices: list[int] = []
        unmatched_keys = []

        for sample in self.samples:
            key = (sample["episode_id"], sample["step_idx"])
            if key in id_to_idx:
                matched_indices.append(id_to_idx[key])
            else:
                unmatched_keys.append(key)

        if unmatched_keys:
            raise ValueError(
                f"{len(unmatched_keys)} samples not found in precomputed file.\n"
                f"First 5 missing: {unmatched_keys[:5]}\n"
                f"Please re-run precompute_embeddings.py with updated data."
            )

        # 4. Update state
        index = torch.tensor(matched_indices, dtype=torch.long)
        self.embeddings = precomputed_embeddings.index_select(0, index)
        self.encoder_dim = int(data["config"]["encoder_dim"])
        self.precomputed = True

        # 5. Remove messages to save memory
        for sample in self.samples:
            if "messages" in sample:
                del sample["messages"]

        print(f"{'='*60}")
        print(f"Loaded {len(self.samples)} matched samples from precomputed file")
        print(f"Embeddings shape: {self.embeddings.shape}")
        print(f"Backend: {precomputed_backend}")
        print(f"{'='*60}\n")

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dict with either:
            - Online mode: messages, model_idx, score_target, cost_target, propensity, benchmark
            - Precomputed mode: embedding, model_idx, score_target, cost_target, propensity, benchmark
        """
        sample = self.samples[idx]

        if self.precomputed:
            # Precomputed mode: return single unified embedding
            return {
                "embedding": self.embeddings[idx],
                "model_idx": sample["model_idx"],
                "score_target": sample["final_score"],  # AEC-adjusted score
                "original_score": sample.get("original_final_score", sample["final_score"]),
                "cost_target": sample["normalized_cost"],
                "propensity": sample.get("propensity") or 1.0,
                "task_id": sample["task_id"],
                "step_idx": sample["step_idx"],
                "benchmark": sample.get("benchmark", "scienceworld"),
                # Error detection fields
                "error_detected": sample.get("error_detected", False),
                "error_types": sample.get("error_types", []),
                "error_penalty": sample.get("error_penalty", 0.0),
                # AEC fields
                "aec_penalty": sample.get("aec_penalty", 0.0),
                "step_aec_penalty": sample.get("step_aec_penalty", 0.0),
            }
        else:
            # Online mode: return raw messages
            return {
                "messages": sample["messages"],
                "model_idx": sample["model_idx"],
                "score_target": sample["final_score"],  # AEC-adjusted score
                "original_score": sample.get("original_final_score", sample["final_score"]),
                "cost_target": sample["normalized_cost"],
                "propensity": sample.get("propensity") or 1.0,
                "task_id": sample["task_id"],
                "step_idx": sample["step_idx"],
                "benchmark": sample.get("benchmark", "scienceworld"),
                # Error detection fields
                "error_detected": sample.get("error_detected", False),
                "error_types": sample.get("error_types", []),
                "error_penalty": sample.get("error_penalty", 0.0),
                # AEC fields
                "aec_penalty": sample.get("aec_penalty", 0.0),
                "step_aec_penalty": sample.get("step_aec_penalty", 0.0),
            }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collate function for variable-length messages (online mode).

    Args:
        batch: List of samples from TrajectoryDataset in online mode.

    Returns:
        Collated batch with:
        - messages: list of message lists (not tensorized)
        - model_indices: list of ints
        - score_targets: torch.Tensor (AEC-adjusted)
        - original_scores: torch.Tensor (before AEC)
        - cost_targets: torch.Tensor
        - propensities: torch.Tensor
        - error_detected: list of bools
        - error_types: list of error type lists
        - error_penalties: torch.Tensor
        - aec_penalties: torch.Tensor
        - task_ids: list of task ID strings (for AWR/Margin grouping)
        - step_indices: list of step indices (for AWR/Margin grouping)
        - benchmarks: list of benchmark names
    """
    return {
        "messages": [b["messages"] for b in batch],
        "model_indices": [b["model_idx"] for b in batch],
        "score_targets": torch.tensor(
            [b["score_target"] for b in batch], dtype=torch.float32
        ),
        "original_scores": torch.tensor(
            [b["original_score"] for b in batch], dtype=torch.float32
        ),
        "cost_targets": torch.tensor(
            [b["cost_target"] for b in batch], dtype=torch.float32
        ),
        "propensities": torch.tensor(
            [b["propensity"] for b in batch], dtype=torch.float32
        ),
        # Error detection fields
        "error_detected": [b["error_detected"] for b in batch],
        "error_types": [b["error_types"] for b in batch],
        "error_penalties": torch.tensor(
            [b["error_penalty"] for b in batch], dtype=torch.float32
        ),
        # AEC fields
        "aec_penalties": torch.tensor(
            [b["aec_penalty"] for b in batch], dtype=torch.float32
        ),
        # Grouping fields for AWR/Margin loss
        "task_ids": [b["task_id"] for b in batch],
        "step_indices": [b["step_idx"] for b in batch],
        "benchmarks": [b["benchmark"] for b in batch],
    }


def collate_fn_precomputed(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collate function for precomputed embeddings mode.

    Args:
        batch: List of samples from TrajectoryDataset in precomputed mode.

    Returns:
        Collated batch with:
        - embeddings: torch.Tensor of shape (B, encoder_dim)
        - model_indices: list of ints
        - score_targets: torch.Tensor (AEC-adjusted)
        - original_scores: torch.Tensor (before AEC)
        - cost_targets: torch.Tensor
        - propensities: torch.Tensor
        - error_detected: list of bools
        - error_types: list of error type lists
        - error_penalties: torch.Tensor
        - aec_penalties: torch.Tensor
        - task_ids: list of task ID strings (for AWR/Margin grouping)
        - step_indices: list of step indices (for AWR/Margin grouping)
        - benchmarks: list of benchmark names
    """
    return {
        "embeddings": torch.stack([b["embedding"] for b in batch]),
        "model_indices": [b["model_idx"] for b in batch],
        "score_targets": torch.tensor(
            [b["score_target"] for b in batch], dtype=torch.float32
        ),
        "original_scores": torch.tensor(
            [b["original_score"] for b in batch], dtype=torch.float32
        ),
        "cost_targets": torch.tensor(
            [b["cost_target"] for b in batch], dtype=torch.float32
        ),
        "propensities": torch.tensor(
            [b["propensity"] for b in batch], dtype=torch.float32
        ),
        # Error detection fields
        "error_detected": [b["error_detected"] for b in batch],
        "error_types": [b["error_types"] for b in batch],
        "error_penalties": torch.tensor(
            [b["error_penalty"] for b in batch], dtype=torch.float32
        ),
        # AEC fields
        "aec_penalties": torch.tensor(
            [b["aec_penalty"] for b in batch], dtype=torch.float32
        ),
        # Grouping fields for AWR/Margin loss
        "task_ids": [b["task_id"] for b in batch],
        "step_indices": [b["step_idx"] for b in batch],
        "benchmarks": [b["benchmark"] for b in batch],
    }
