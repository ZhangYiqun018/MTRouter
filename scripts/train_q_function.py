#!/usr/bin/env python
"""Training script for NeuralQFunction.

This script trains a Q-function for learned model routing using
trajectory data collected from roulette propensity sampling.

Usage:
    # Quick test (3 epochs)
    python scripts/train_q_function.py --epochs 3 --batch-size 16

    # Full training with the paper default single outcome target
    python scripts/train_q_function.py --epochs 50 --batch-size 64 --lambda_ 0.0

    # With custom paths
    python scripts/train_q_function.py \
        --data-dir trajectories/roulette_propensity \
        --output-dir outputs/q_function \
        --epochs 50

    # With multiple data directories (roulette + baseline)
    python scripts/train_q_function.py \
        --data-dir trajectories/roulette_propensity \
        --data-dir trajectories/baseline_openai_gpt-5 \
        --data-dir trajectories/baseline_deepseek_deepseek-v3.2 \
        --output-dir outputs/q_function \
        --epochs 50

    # With AEC (Annealed Error Cost) penalty enabled
    python scripts/train_q_function.py \
        --data-dir trajectories/roulette_propensity \
        --aec-enabled \
        --epochs 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# ============================================================================
# Logging Utilities
# ============================================================================


@dataclass
class EpochLog:
    """Single epoch training log entry."""

    epoch: int
    train_loss: float
    train_score_loss: float
    train_cost_loss: float
    val_loss: float | None
    val_score_loss: float | None
    val_cost_loss: float | None
    learning_rate: float
    epoch_time: float
    is_best: bool
    pred_score_mean: float
    pred_score_std: float
    pred_cost_mean: float
    pred_cost_std: float
    timestamp: str


class TrainingLogger:
    """Logger that writes epoch logs to JSONL file."""

    def __init__(self, output_dir: Path):
        self.log_path = output_dir / "training_log.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Clear existing log if present
        self.log_path.write_text("")

    def log_epoch(self, log: EpochLog) -> None:
        """Append epoch log entry."""
        with open(self.log_path, "a") as f:
            f.write(json.dumps(asdict(log)) + "\n")


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'


def print_banner() -> None:
    """Print training script banner."""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}╔══════════════════════════════════════════════════════════════════╗
║                   Q-Function Training Pipeline                    ║
║                      Learned Model Routing                        ║
╚══════════════════════════════════════════════════════════════════╝{Colors.RESET}
"""
    print(banner)


def print_section(title: str) -> None:
    """Print section header."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'─' * 60}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}  {title}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'─' * 60}{Colors.RESET}")


def print_config(args: argparse.Namespace) -> None:
    """Print configuration in a nice table format."""
    print_section("Configuration")

    data_dirs_str = ", ".join(str(d) for d in args.data_dirs)
    configs = [
        ("Data", [
            ("Data directories", data_dirs_str),
            ("Output directory", args.output_dir),
            ("Max samples", args.max_samples or "All"),
            ("Validation ratio", f"{args.val_ratio:.0%}"),
        ]),
        ("Training", [
            ("Epochs", args.epochs),
            ("Batch size", args.batch_size),
            ("Lambda (legacy target tradeoff)", args.lambda_),
            ("Cost loss weight", f"{args.cost_loss_weight} (legacy; single-head default uses 0)"),
        ]),
        ("Optimizer", [
            ("Type", args.optimizer.upper()),
            ("Learning rate", f"{args.lr:.0e}"),
            ("Weight decay", args.weight_decay),
            ("Gradient clip", args.grad_clip if args.grad_clip > 0 else "Disabled"),
        ]),
        ("Scheduler", [
            ("Type", args.scheduler if args.scheduler != "none" else "Disabled"),
            ("Min LR", f"{args.min_lr:.0e}" if args.scheduler != "none" else "N/A"),
        ]),
        ("Early Stopping", [
            ("Patience", args.early_stop if args.early_stop > 0 else "Disabled"),
        ]),
        ("Checkpoints", [
            ("Save every N epochs", args.save_every if args.save_every > 0 else "Disabled"),
            ("Max checkpoints", args.max_checkpoints if args.save_every > 0 else "N/A"),
        ]),
        ("AEC (Error Cost)", [
            ("Enabled", "Yes" if getattr(args, "aec_enabled", False) else "No"),
        ]),
        ("AWR (Advantage-Weighted)", [
            ("Enabled", "Yes" if getattr(args, "awr_enabled", False) else "No"),
            ("Tau", getattr(args, "awr_tau", 1.0) if getattr(args, "awr_enabled", False) else "N/A"),
            ("Min group size", getattr(args, "awr_min_group", 2) if getattr(args, "awr_enabled", False) else "N/A"),
        ]),
        ("Margin Loss", [
            ("Enabled", "Yes" if getattr(args, "margin_enabled", False) else "No"),
            ("Margin value", getattr(args, "margin_value", 0.1) if getattr(args, "margin_enabled", False) else "N/A"),
            ("Weight", getattr(args, "margin_weight", 0.1) if getattr(args, "margin_enabled", False) else "N/A"),
        ]),
    ]

    for group_name, items in configs:
        print(f"\n  {Colors.YELLOW}{group_name}:{Colors.RESET}")
        for key, value in items:
            print(f"    {key:<20} : {value}")


def print_model_summary(q_func: torch.nn.Module, model_names: list[str]) -> None:
    """Print model architecture summary."""
    print_section("Model Architecture")

    total_params = sum(p.numel() for p in q_func.parameters())
    trainable_params = sum(p.numel() for p in q_func.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\n  {Colors.YELLOW}Components:{Colors.RESET}")
    print(f"    HistoryEncoder     : {q_func.history_encoder.__class__.__name__}")
    print(f"      - Backbone       : {q_func.history_encoder.config.model_name.split('/')[-1]}")
    print(f"      - Pooling        : {q_func.history_encoder.config.pooling_mode}")
    print(f"      - Output dim     : {q_func.history_encoder.output_dim}")
    print(f"    ModelEncoder       : {q_func.model_encoder.__class__.__name__}")
    print(f"      - Num models     : {len(model_names)}")
    print(f"      - Output dim     : {q_func.model_encoder.output_dim}")
    print(f"    QNetwork           : {q_func.q_network.__class__.__name__}")
    print(f"      - Hidden dim     : {q_func.q_network.config.hidden_dim}")
    print(f"      - Num layers     : {q_func.q_network.config.n_layers}")

    print(f"\n  {Colors.YELLOW}Parameters:{Colors.RESET}")
    print(f"    Total              : {total_params:>12,}")
    print(f"    Trainable          : {trainable_params:>12,} ({trainable_params/total_params:.1%})")
    print(f"    Frozen             : {frozen_params:>12,} ({frozen_params/total_params:.1%})")

    print(f"\n  {Colors.YELLOW}Models:{Colors.RESET}")
    for i, name in enumerate(model_names):
        print(f"    [{i}] {name}")


def print_data_summary(
    full_dataset,
    train_size: int,
    val_size: int,
    model_names: list[str],
    lambda_: float,
    aec_enabled: bool = False,
) -> tuple[list[float], list[float]]:
    """Print dataset statistics.

    Returns:
        Tuple of (final_scores, normalized_costs) lists.
    """
    print_section("Dataset Statistics")

    samples = full_dataset.samples
    final_scores = [s["final_score"] for s in samples]
    remaining_costs = [s["remaining_cost"] for s in samples]
    normalized_costs = [s.get("normalized_cost", 0) for s in samples]

    # AEC-related fields
    original_scores = [s.get("original_score", s["final_score"]) for s in samples]
    aec_penalties = [s.get("aec_penalty", 0) for s in samples]
    step_aec_penalties = [s.get("step_aec_penalty", 0) for s in samples]

    # Per-model statistics
    model_counts = Counter(s["model_idx"] for s in samples)
    model_scores = {i: [] for i in range(len(model_names))}
    for s in samples:
        model_scores[s["model_idx"]].append(s["final_score"])

    print(f"\n  {Colors.YELLOW}Split:{Colors.RESET}")
    print(f"    Total samples      : {len(samples):,}")
    print(f"    Training           : {train_size:,} ({train_size/len(samples):.0%})")
    print(f"    Validation         : {val_size:,} ({val_size/len(samples):.0%})")

    # Benchmark distribution
    benchmark_counts = Counter(s.get("benchmark", "scienceworld") for s in samples)
    print(f"\n  {Colors.YELLOW}Benchmark Distribution:{Colors.RESET}")
    for bm, count in sorted(benchmark_counts.items()):
        print(f"    {bm:<20} {count:>8,} ({count/len(samples):.1%})")

    print(f"\n  {Colors.YELLOW}Per-Model Distribution:{Colors.RESET}")
    print(f"    {'Model':<25} {'Count':>8} {'%':>6} {'Avg Score':>10}")
    print(f"    {'-' * 51}")
    for i, name in enumerate(model_names):
        count = model_counts.get(i, 0)
        pct = count / len(samples) * 100
        avg_score = np.mean(model_scores[i]) if model_scores[i] else 0
        print(f"    {name:<25} {count:>8,} {pct:>5.1f}% {avg_score:>10.1f}")

    print(f"\n  {Colors.YELLOW}Single-Head Targets (Q/value head):{Colors.RESET}")
    print(f"    {'Metric':<20} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
    print(f"    {'-' * 62}")
    print(f"    {'Score Target':<20} {min(final_scores):>10.1f} {max(final_scores):>10.1f} "
          f"{np.mean(final_scores):>10.1f} {np.std(final_scores):>10.1f}")
    print(f"    {'Cost Target (norm)':<20} {min(normalized_costs):>10.1f} {max(normalized_costs):>10.1f} "
          f"{np.mean(normalized_costs):>10.1f} {np.std(normalized_costs):>10.1f}")
    print(f"    {'Remaining Cost ($)':<20} {min(remaining_costs):>10.4f} {max(remaining_costs):>10.4f} "
          f"{np.mean(remaining_costs):>10.4f} {np.std(remaining_costs):>10.4f}")

    # AEC statistics if enabled
    if aec_enabled and any(p > 0 for p in aec_penalties):
        print(f"\n  {Colors.YELLOW}AEC Penalty Statistics:{Colors.RESET}")
        print(f"    {'Metric':<20} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
        print(f"    {'-' * 62}")
        print(f"    {'Original Score':<20} {min(original_scores):>10.3f} {max(original_scores):>10.3f} "
              f"{np.mean(original_scores):>10.3f} {np.std(original_scores):>10.3f}")
        print(f"    {'Adjusted Score':<20} {min(final_scores):>10.3f} {max(final_scores):>10.3f} "
              f"{np.mean(final_scores):>10.3f} {np.std(final_scores):>10.3f}")
        print(f"    {'Cumulative Penalty':<20} {min(aec_penalties):>10.3f} {max(aec_penalties):>10.3f} "
              f"{np.mean(aec_penalties):>10.3f} {np.std(aec_penalties):>10.3f}")
        print(f"    {'Step Penalty':<20} {min(step_aec_penalties):>10.4f} {max(step_aec_penalties):>10.4f} "
              f"{np.mean(step_aec_penalties):>10.4f} {np.std(step_aec_penalties):>10.4f}")

        # Count samples with non-zero penalties
        n_penalized = sum(1 for p in aec_penalties if p > 0)
        print(f"\n    Samples with penalty : {n_penalized:,} ({n_penalized / len(samples) * 100:.1f}%)")

    return final_scores, normalized_costs


def format_time(seconds: float) -> str:
    """Format seconds into human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def print_epoch_summary(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    train_score_loss: float,
    train_cost_loss: float,
    val_loss: float | None,
    lr: float,
    epoch_time: float,
    pred_score_mean: float,
    pred_cost_mean: float,
    best_val_loss: float,
    is_best: bool,
    epochs_no_improve: int,
    total_time: float,
) -> None:
    """Print epoch summary in a compact format."""
    eta = (total_epochs - epoch - 1) * epoch_time

    # Status indicator
    if is_best:
        status = f"{Colors.GREEN}★ Best{Colors.RESET}"
    elif val_loss is not None and epochs_no_improve > 0:
        status = f"{Colors.YELLOW}↓ {epochs_no_improve}{Colors.RESET}"
    else:
        status = ""

    # Build output line
    val_str = f"val={val_loss:.4f}" if val_loss is not None else ""

    print(
        f"  Epoch {epoch + 1:>3}/{total_epochs} │ "
        f"loss={train_loss:.4f} (s={train_score_loss:.3f} c={train_cost_loss:.3f}) {val_str:<12} │ "
        f"score={pred_score_mean:>6.1f} cost={pred_cost_mean:>5.1f} │ "
        f"lr={lr:.1e} │ "
        f"{format_time(epoch_time):>5} │ "
        f"ETA {format_time(eta):>5} │ "
        f"{status}"
    )


def print_training_summary(
    total_time: float,
    final_train_loss: float,
    best_val_loss: float,
    output_path: Path,
    best_path: Path | None,
    log_dir: Path,
    training_log_path: Path | None = None,
) -> None:
    """Print final training summary."""
    print_section("Training Complete")

    print(f"\n  {Colors.YELLOW}Results:{Colors.RESET}")
    print(f"    Total time         : {format_time(total_time)}")
    print(f"    Final train loss   : {final_train_loss:.4f}")
    if best_val_loss < float("inf"):
        print(f"    Best val loss      : {best_val_loss:.4f}")

    print(f"\n  {Colors.YELLOW}Saved Files:{Colors.RESET}")
    print(f"    Final model        : {output_path}")
    if best_path:
        print(f"    Best model         : {best_path}")
    if training_log_path:
        print(f"    Training log       : {training_log_path}")
    print(f"    TensorBoard logs   : {log_dir}")

    print(f"\n  {Colors.YELLOW}View TensorBoard:{Colors.RESET}")
    print(f"    tensorboard --logdir {log_dir.parent}")
    print()


def save_periodic_checkpoint(
    q_func: "NeuralQFunction",
    output_dir: Path,
    epoch: int,
    max_checkpoints: int,
) -> None:
    """Save periodic checkpoint and cleanup old ones.

    Args:
        q_func: The Q-function model to save.
        output_dir: Base output directory.
        epoch: Current epoch number (1-indexed).
        max_checkpoints: Maximum number of checkpoints to keep.
    """
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save current checkpoint
    checkpoint_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
    q_func.save(checkpoint_path)
    print(f"  {Colors.GREEN}Saved checkpoint: {checkpoint_path.name}{Colors.RESET}")

    # Cleanup old checkpoints if exceeding limit
    existing = sorted(checkpoint_dir.glob("epoch_*.pt"))
    while len(existing) > max_checkpoints:
        oldest = existing.pop(0)
        oldest.unlink()
        print(f"  {Colors.DIM}Removed old checkpoint: {oldest.name}{Colors.RESET}")


# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml

from miniagenticrouter.research.training import (
    NeuralQFunction,
    NeuralQFunctionConfig,
    ModelEncoderConfig,
    QNetworkConfig,
    TrajectoryDataset,
    collate_fn,
    collate_fn_precomputed,
)
from miniagenticrouter.research.training.error_detector import (
    ErrorDetector,
    AECConfig,
)
from miniagenticrouter.research.training.encoders import (
    HistoryEncoderConfig,
    PrecomputeConfig,
)
from miniagenticrouter.research.utils.config import get_model_pool


def load_training_config(config_path: Path | None = None) -> dict:
    """Load training configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default builtin config.

    Returns:
        Configuration dict with keys: history_encoder, training, model_encoder, q_network
    """
    if config_path is None:
        # Use builtin config
        from miniagenticrouter.config import builtin_config_dir
        config_path = builtin_config_dir / "research" / "training.yaml"

    with open(config_path) as f:
        return yaml.safe_load(f)


def create_q_function_config(yaml_config: dict) -> NeuralQFunctionConfig:
    """Create NeuralQFunctionConfig from YAML config dict.

    Args:
        yaml_config: Config dict loaded from training.yaml

    Returns:
        NeuralQFunctionConfig with proper sub-configs
    """
    # History encoder config
    he_cfg = yaml_config.get("history_encoder", {})
    history_encoder_config = HistoryEncoderConfig(
        model_name=he_cfg.get("model_name", HistoryEncoderConfig.model_name),
        freeze_encoder=he_cfg.get("freeze_encoder", True),
        max_tokens=he_cfg.get("max_tokens", 512),
        min_recent_turns=he_cfg.get("min_recent_turns", 1),
        use_projection=he_cfg.get("use_projection", True),
        pooling_mode=he_cfg.get("pooling_mode", "last_token"),
        backend=he_cfg.get("backend", HistoryEncoderConfig.backend),
        vllm_base_url=he_cfg.get("vllm_base_url", HistoryEncoderConfig.vllm_base_url),
        vllm_model_id=he_cfg.get("vllm_model_id"),
    )

    # Model encoder config
    me_cfg = yaml_config.get("model_encoder", {})
    model_encoder_config = ModelEncoderConfig(
        meta_embed_dim=me_cfg.get("meta_embed_dim", me_cfg.get("meta_hidden_dim", 32)),
        residual_embed_dim=me_cfg.get("residual_embed_dim", me_cfg.get("residual_dim", 16)),
        residual_l2_weight=me_cfg.get(
            "residual_l2_weight", me_cfg.get("residual_reg_weight", 0.001)
        ),
        output_dim=me_cfg.get("output_dim", 64),
    )

    # Q-network config
    qn_cfg = yaml_config.get("q_network", {})
    q_network_config = QNetworkConfig(
        hidden_dims=qn_cfg.get("hidden_dims"),
        dropout=qn_cfg.get("dropout", 0.0),
        hidden_dim=qn_cfg.get("hidden_dim", 256),
        n_layers=qn_cfg.get("n_layers", 2),
    )

    return NeuralQFunctionConfig(
        history_encoder_config=history_encoder_config,
        model_encoder_config=model_encoder_config,
        q_network_config=q_network_config,
    )


# ============================================================================
# AWR and Margin Loss Functions
# ============================================================================


def compute_awr_weights(
    score_targets: torch.Tensor,
    task_ids: list[str],
    step_indices: list[int],
    tau: float = 1.0,
    min_group_size: int = 2,
) -> torch.Tensor:
    """Compute AWR weights: w = exp(advantage / tau).

    AWR (Advantage-Weighted Regression) reweights samples so that
    high-advantage samples (better than average for their state)
    get higher weights, amplifying the score gap signal.

    Args:
        score_targets: Target scores [B].
        task_ids: List of task IDs for grouping.
        step_indices: List of step indices for grouping.
        tau: Temperature parameter. Smaller = sharper weights (0.1~10.0).
        min_group_size: Minimum samples per (task_id, step_idx) group.
            Groups smaller than this get weight=1 (advantage=0).

    Returns:
        Weight tensor [B], normalized to mean=1.0.
    """
    device = score_targets.device
    batch_size = len(score_targets)

    # Group by (task_id, step_idx)
    state_to_indices: dict[tuple, list[int]] = {}
    for i, (tid, sidx) in enumerate(zip(task_ids, step_indices)):
        key = (tid, sidx)
        if key not in state_to_indices:
            state_to_indices[key] = []
        state_to_indices[key].append(i)

    advantages = torch.zeros(batch_size, device=device)

    for state_key, indices in state_to_indices.items():
        if len(indices) < min_group_size:
            # Group too small, can't compute meaningful advantage
            continue

        # Compute baseline (mean score for this state)
        group_scores = score_targets[indices]
        baseline = group_scores.mean()

        # Compute advantage for each sample in group
        for idx in indices:
            advantages[idx] = score_targets[idx] - baseline

    # Compute weights = exp(advantage / tau), normalize to mean=1
    weights = torch.exp(advantages / tau)
    weights = weights / weights.mean()

    return weights




def build_value_targets(
    score_targets: torch.Tensor,
    cost_targets: torch.Tensor,
    lambda_: float,
) -> torch.Tensor:
    """Build single-head Q targets.

    The paper trains a single outcome/value head; the default lambda is 0,
    so the target is the normalized terminal score. A non-zero lambda keeps
    the older cost-penalized variant available for ablations.
    """
    return score_targets - lambda_ * cost_targets


def forward_q_values(
    q_func: "NeuralQFunction",
    batch: dict,
    *,
    precomputed: bool,
    device: torch.device,
    lambda_: float,
) -> torch.Tensor:
    """Forward pass compatible with the current single-head QFunction API."""
    model_indices = batch["model_indices"]
    if precomputed:
        embeddings = batch["embeddings"].to(device)
        output = q_func.forward_precomputed(embeddings, model_indices)
    else:
        output = q_func(batch["messages"], model_indices)

    # Older historical checkpoints/scripts used a dual-head (score, cost) API.
    # Accept that shape when present, but train the current single-head API.
    if isinstance(output, tuple):
        score_pred, cost_pred = output
        return score_pred - lambda_ * cost_pred
    return output


def compute_margin_loss(
    pred_scores: torch.Tensor,
    score_targets: torch.Tensor,
    task_ids: list[str],
    step_indices: list[int],
    margin: float = 0.1,
    min_group_size: int = 2,
) -> torch.Tensor:
    """Compute Margin Loss: best model should lead others by margin.

    For each state (task_id, step_idx), find the model with highest target
    score (ground truth best) and add penalty if predicted score doesn't
    lead other models by the specified margin.

    Args:
        pred_scores: Predicted scores [B].
        score_targets: Target scores [B].
        task_ids: List of task IDs for grouping.
        step_indices: List of step indices for grouping.
        margin: Target gap between best and other models.
        min_group_size: Minimum samples per group.

    Returns:
        Scalar margin loss.
    """
    device = pred_scores.device

    # Group by (task_id, step_idx)
    state_to_indices: dict[tuple, list[int]] = {}
    for i, (tid, sidx) in enumerate(zip(task_ids, step_indices)):
        key = (tid, sidx)
        if key not in state_to_indices:
            state_to_indices[key] = []
        state_to_indices[key].append(i)

    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    n_pairs = 0

    for state_key, indices in state_to_indices.items():
        if len(indices) < min_group_size:
            continue

        # Find best model (highest target score)
        best_local_idx = max(
            range(len(indices)),
            key=lambda i: score_targets[indices[i]].item()
        )
        best_idx = indices[best_local_idx]

        for local_i, idx in enumerate(indices):
            if local_i == best_local_idx:
                continue
            # We want: pred[best] > pred[other] + margin
            # Loss = ReLU(pred[other] - pred[best] + margin)
            diff = pred_scores[idx] - pred_scores[best_idx] + margin
            total_loss = total_loss + F.relu(diff)
            n_pairs += 1

    if n_pairs > 0:
        return total_loss / n_pairs
    else:
        return torch.tensor(0.0, device=device, requires_grad=True)


def train(
    args: argparse.Namespace,
    yaml_config: dict,
    model_names: list[str],
    aec_config: AECConfig | None = None,
    error_detector: ErrorDetector | None = None,
) -> None:
    """Run training loop.

    Args:
        args: Command-line arguments.
        yaml_config: Configuration dict loaded from training.yaml
        model_names: List of model names for routing.
        aec_config: AEC configuration for error cost penalty (None to disable).
        error_detector: Error detector for AEC penalty calculation (required if aec_config is set).
    """
    # Print banner and configuration
    print_banner()
    print_config(args)

    # Create Q-function config from YAML
    q_func_config = create_q_function_config(yaml_config)
    print(f"\n  {Colors.CYAN}Encoder max_tokens: {q_func_config.history_encoder_config.max_tokens}{Colors.RESET}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n  {Colors.GREEN}GPU: {gpu_name} ({gpu_mem:.1f} GB){Colors.RESET}")
    else:
        print(f"\n  {Colors.YELLOW}Running on CPU{Colors.RESET}")

    # Initialize TensorBoard writer
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.output_dir) / "runs" / f"lambda{args.lambda_}_{run_name}"
    writer = SummaryWriter(log_dir=str(log_dir))

    # Create precompute config if enabled
    precompute_config = None
    if args.precompute:
        he_cfg = yaml_config.get("history_encoder", {})
        precompute_config = PrecomputeConfig(
            enabled=True,
            backend=he_cfg.get("backend", PrecomputeConfig.backend),
            model_name=he_cfg.get("model_name", PrecomputeConfig.model_name),
            batch_size=args.precompute_batch_size,
            max_tokens=he_cfg.get("max_tokens", 512),
            min_recent_turns=he_cfg.get("min_recent_turns", 1),
            pooling_mode=he_cfg.get("pooling_mode", "last_token"),
            vllm_base_url=he_cfg.get("vllm_base_url", PrecomputeConfig.vllm_base_url),
            vllm_model_id=he_cfg.get("vllm_model_id"),
        )
        print(f"\n  {Colors.CYAN}Precomputation enabled:{Colors.RESET}")
        print(f"    Backend            : {precompute_config.backend}")
        print(f"    Batch size         : {precompute_config.batch_size}")
        print(f"    Pooling mode       : {precompute_config.pooling_mode}")
        if precompute_config.backend == "vllm":
            print(f"    vLLM base URL      : {precompute_config.vllm_base_url}")
            print(f"    vLLM request model : {precompute_config.vllm_model_id or precompute_config.model_name.split('/')[-1]}")

    # Load dataset
    print_section("Loading Data")
    print(f"\n  Loading from: {', '.join(str(d) for d in args.data_dirs)}")

    # Determine precomputation mode:
    # - precomputed_path: Load from file (highest priority)
    # - precompute: Compute online during init
    # - Neither: Online mode (no precomputation)
    effective_precompute_config = None
    if args.precomputed_path:
        print(f"  {Colors.CYAN}Loading precomputed embeddings from: {args.precomputed_path}{Colors.RESET}")
    elif precompute_config:
        effective_precompute_config = precompute_config

    full_dataset = TrajectoryDataset(
        trajectory_dir=args.data_dirs,
        model_names=model_names,
        lambda_=args.lambda_,
        max_samples=args.max_samples,
        precompute_config=effective_precompute_config,
        precomputed_path=args.precomputed_path,
        error_detector=error_detector,
        aec_config=aec_config,
    )

    if len(full_dataset) == 0:
        print(f"  {Colors.RED}ERROR: No samples loaded. Check trajectory directory.{Colors.RESET}")
        writer.close()
        return

    # Train/validation split
    val_size = int(len(full_dataset) * args.val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Print data statistics
    score_targets, cost_targets = print_data_summary(
        full_dataset, train_size, val_size, model_names, args.lambda_,
        aec_enabled=(aec_config is not None),
    )

    # Log hyperparameters to TensorBoard
    aec_info = ""
    if aec_config is not None:
        aec_info = f"""
- aec_enabled: True
- aec_p0: {aec_config.p0}
- aec_p1: {aec_config.p1}
- aec_w_min: {aec_config.w_min}
- aec_w_max: {aec_config.w_max}
- aec_min_penalty_factor: {aec_config.min_penalty_factor}"""
    else:
        aec_info = "\n- aec_enabled: False"

    writer.add_text("Hyperparameters", f"""
- lambda: {args.lambda_}
- cost_loss_weight: {args.cost_loss_weight}  # legacy; no effect when zero
- batch_size: {args.batch_size}
- learning_rate: {args.lr}
- epochs: {args.epochs}
- optimizer: {args.optimizer}
- weight_decay: {args.weight_decay}
- scheduler: {args.scheduler}
- grad_clip: {args.grad_clip}
- early_stop: {args.early_stop}
- train_samples: {len(train_dataset)}
- val_samples: {len(val_dataset)}
- score_target_mean: {np.mean(score_targets):.2f}
- score_target_std: {np.std(score_targets):.2f}
- cost_target_mean: {np.mean(cost_targets):.2f}
- cost_target_std: {np.std(cost_targets):.2f}{aec_info}
""")

    # Select collate function based on precomputation mode
    selected_collate_fn = collate_fn_precomputed if full_dataset.precomputed else collate_fn
    if full_dataset.precomputed:
        print(f"\n  {Colors.GREEN}Using precomputed embeddings mode{Colors.RESET}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=selected_collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=selected_collate_fn,
        num_workers=0,
    ) if val_size > 0 else None

    # Model
    print_section("Initializing Model")
    print("\n  Loading pretrained encoder...")
    q_func = NeuralQFunction(model_names=model_names, config=q_func_config, lambda_=args.lambda_)
    q_func.to(device)

    # Print model summary
    print_model_summary(q_func, model_names)

    # Parameter grouping (bias and norm don't get weight decay)
    decay_params = []
    no_decay_params = []
    for name, param in q_func.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    # Optimizer
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    else:
        optimizer = torch.optim.Adam(param_groups, lr=args.lr)

    # Learning rate scheduler
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr,
        )
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=args.min_lr,
        )

    # Training loop
    print_section("Training")
    print(f"\n  Batches per epoch: {len(train_loader)}")
    print(f"  Total iterations : {len(train_loader) * args.epochs:,}")
    print()

    # Initialize training logger
    training_logger = TrainingLogger(Path(args.output_dir))

    global_step = 0
    best_val_loss = float("inf")
    epochs_no_improve = 0
    training_start = time.time()
    final_train_loss = 0.0

    for epoch in range(args.epochs):
        q_func.train()
        total_loss = 0.0
        total_score_loss = 0.0
        total_cost_loss = 0.0
        total_reg_loss = 0.0
        total_pred_scores = []
        total_pred_costs = []
        total_score_targets = []
        total_cost_targets = []
        n_batches = 0
        epoch_start = time.time()
        last_grad_norm = 0.0

        # Progress bar for batches
        pbar = tqdm(
            train_loader,
            desc=f"  Epoch {epoch + 1:>3}/{args.epochs}",
            leave=False,
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
        )

        for batch_idx, batch in enumerate(pbar):
            batch_start = time.time()

            model_indices = batch["model_indices"]
            score_targets_batch = batch["score_targets"].to(device)
            cost_targets_batch = batch["cost_targets"].to(device)

            # Forward through the current single-head Q network.
            pred_scores = forward_q_values(
                q_func,
                batch,
                precomputed=full_dataset.precomputed,
                device=device,
                lambda_=args.lambda_,
            )
            pred_costs = torch.zeros_like(pred_scores)  # Kept for legacy logging fields.
            value_targets_batch = build_value_targets(
                score_targets_batch,
                cost_targets_batch,
                args.lambda_,
            )

            # Single-head loss over Q/value targets, with optional AWR and margin terms.

            # Get grouping fields for AWR/Margin
            task_ids = batch.get("task_ids")
            step_indices = batch.get("step_indices")

            # AWR: reweight samples by advantage
            awr_weights = None
            if args.awr_enabled and task_ids is not None:
                awr_weights = compute_awr_weights(
                    score_targets_batch,
                    task_ids,
                    step_indices,
                    tau=args.awr_tau,
                    min_group_size=args.awr_min_group,
                )
                # Weighted MSE on the Q/value target.
                score_loss = (awr_weights * (pred_scores - value_targets_batch) ** 2).mean()
                cost_loss = torch.zeros((), device=device)
            else:
                # Standard MSE on the Q/value target.
                score_loss = F.mse_loss(pred_scores, value_targets_batch)
                cost_loss = torch.zeros((), device=device)

            # Margin loss: enforce gap between best and other models
            margin_loss = torch.tensor(0.0, device=device)
            if args.margin_enabled and task_ids is not None:
                margin_loss = compute_margin_loss(
                    pred_scores,
                    score_targets_batch,
                    task_ids,
                    step_indices,
                    margin=args.margin_value,
                    min_group_size=args.awr_min_group,
                )

            # Combined loss
            loss = score_loss + args.cost_loss_weight * cost_loss + args.margin_weight * margin_loss
            reg_loss = q_func.get_regularization_loss()
            total_loss_batch = loss + reg_loss

            # Backward
            optimizer.zero_grad()
            total_loss_batch.backward()

            # Gradient clipping
            grad_norm = None
            if args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    q_func.parameters(),
                    max_norm=args.grad_clip,
                )
                last_grad_norm = grad_norm.item()

            optimizer.step()

            # Accumulate for logging
            total_loss += loss.item()
            total_score_loss += score_loss.item()
            total_cost_loss += cost_loss.item()
            total_reg_loss += reg_loss.item()
            total_pred_scores.extend(pred_scores.detach().cpu().numpy())
            total_pred_costs.extend(pred_costs.detach().cpu().numpy())
            total_score_targets.extend(score_targets_batch.detach().cpu().numpy())
            total_cost_targets.extend(cost_targets_batch.detach().cpu().numpy())
            n_batches += 1
            global_step += 1

            # TensorBoard: batch-level logging (every 10 batches)
            if global_step % 10 == 0:
                writer.add_scalar("Loss/train_batch", loss.item(), global_step)
                writer.add_scalar("Loss/score_batch", score_loss.item(), global_step)
                writer.add_scalar("Loss/cost_batch", cost_loss.item(), global_step)
                writer.add_scalar("Loss/reg_batch", reg_loss.item(), global_step)
                writer.add_scalar("Time/batch_ms", (time.time() - batch_start) * 1000, global_step)
                if grad_norm is not None:
                    writer.add_scalar("Gradients/norm", grad_norm.item(), global_step)
                # AWR/Margin logging
                if args.margin_enabled:
                    writer.add_scalar("Loss/margin_batch", margin_loss.item(), global_step)
                if args.awr_enabled and awr_weights is not None:
                    writer.add_scalar("AWR/weight_mean", awr_weights.mean().item(), global_step)
                    writer.add_scalar("AWR/weight_std", awr_weights.std().item(), global_step)
                    writer.add_scalar("AWR/weight_max", awr_weights.max().item(), global_step)

            # Update progress bar
            pbar.set_postfix_str(f"loss={loss.item():.4f} s={score_loss.item():.3f} c={cost_loss.item():.3f}")

        # Epoch metrics
        avg_loss = total_loss / n_batches
        avg_score_loss = total_score_loss / n_batches
        avg_cost_loss = total_cost_loss / n_batches
        final_train_loss = avg_loss
        epoch_time = time.time() - epoch_start
        total_time = time.time() - training_start
        pred_scores_arr = np.array(total_pred_scores)
        pred_costs_arr = np.array(total_pred_costs)
        score_targets_arr = np.array(total_score_targets)
        cost_targets_arr = np.array(total_cost_targets)

        # TensorBoard: epoch-level logging
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
        writer.add_scalar("Loss/score_epoch", avg_score_loss, epoch)
        writer.add_scalar("Loss/cost_epoch", avg_cost_loss, epoch)
        writer.add_scalar("Score/pred_mean", pred_scores_arr.mean(), epoch)
        writer.add_scalar("Score/pred_std", pred_scores_arr.std(), epoch)
        writer.add_scalar("Score/target_mean", score_targets_arr.mean(), epoch)
        writer.add_scalar("Cost/pred_mean", pred_costs_arr.mean(), epoch)
        writer.add_scalar("Cost/pred_std", pred_costs_arr.std(), epoch)
        writer.add_scalar("Cost/target_mean", cost_targets_arr.mean(), epoch)
        writer.add_scalar("Time/epoch_sec", epoch_time, epoch)
        writer.add_scalar("LR/current", current_lr, epoch)

        # Histograms (every 5 epochs to save space)
        if epoch % 5 == 0:
            writer.add_histogram("Score/predictions", pred_scores_arr, epoch)
            writer.add_histogram("Score/targets", score_targets_arr, epoch)
            writer.add_histogram("Cost/predictions", pred_costs_arr, epoch)
            writer.add_histogram("Cost/targets", cost_targets_arr, epoch)

        # Validation
        val_loss = None
        val_score_loss = None
        val_cost_loss = None
        is_best = False
        if val_loader is not None:
            q_func.eval()
            val_total_loss = 0.0
            val_total_score_loss = 0.0
            val_total_cost_loss = 0.0
            val_n_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    model_indices = batch["model_indices"]
                    score_targets_batch = batch["score_targets"].to(device)
                    cost_targets_batch = batch["cost_targets"].to(device)

                    # Forward through the current single-head Q network.
                    pred_scores = forward_q_values(
                        q_func,
                        batch,
                        precomputed=full_dataset.precomputed,
                        device=device,
                        lambda_=args.lambda_,
                    )
                    value_targets_batch = build_value_targets(
                        score_targets_batch,
                        cost_targets_batch,
                        args.lambda_,
                    )
                    pred_costs = torch.zeros_like(pred_scores)

                    batch_score_loss = F.mse_loss(pred_scores, value_targets_batch).item()
                    batch_cost_loss = 0.0
                    val_total_score_loss += batch_score_loss
                    val_total_cost_loss += batch_cost_loss
                    val_total_loss += batch_score_loss + args.cost_loss_weight * batch_cost_loss
                    val_n_batches += 1

            val_loss = val_total_loss / val_n_batches
            val_score_loss = val_total_score_loss / val_n_batches
            val_cost_loss = val_total_cost_loss / val_n_batches
            writer.add_scalar("Loss/validation", val_loss, epoch)
            writer.add_scalar("Loss/val_score", val_score_loss, epoch)
            writer.add_scalar("Loss/val_cost", val_cost_loss, epoch)

            # Save best model and track improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                is_best = True
                best_path = Path(args.output_dir) / "q_function_best.pt"
                best_path.parent.mkdir(parents=True, exist_ok=True)
                q_func.save(best_path)
            else:
                epochs_no_improve += 1

        # Learning rate scheduler step
        if scheduler is not None:
            if args.scheduler == "cosine":
                scheduler.step()
            elif args.scheduler == "plateau" and val_loss is not None:
                scheduler.step(val_loss)

        # Print epoch summary
        print_epoch_summary(
            epoch=epoch,
            total_epochs=args.epochs,
            train_loss=avg_loss,
            train_score_loss=avg_score_loss,
            train_cost_loss=avg_cost_loss,
            val_loss=val_loss,
            lr=current_lr,
            epoch_time=epoch_time,
            pred_score_mean=pred_scores_arr.mean(),
            pred_cost_mean=pred_costs_arr.mean(),
            best_val_loss=best_val_loss,
            is_best=is_best,
            epochs_no_improve=epochs_no_improve,
            total_time=total_time,
        )

        # Log epoch to file
        log_entry = EpochLog(
            epoch=epoch + 1,
            train_loss=avg_loss,
            train_score_loss=avg_score_loss,
            train_cost_loss=avg_cost_loss,
            val_loss=val_loss,
            val_score_loss=val_score_loss,
            val_cost_loss=val_cost_loss,
            learning_rate=current_lr,
            epoch_time=epoch_time,
            is_best=is_best,
            pred_score_mean=float(pred_scores_arr.mean()),
            pred_score_std=float(pred_scores_arr.std()),
            pred_cost_mean=float(pred_costs_arr.mean()),
            pred_cost_std=float(pred_costs_arr.std()),
            timestamp=datetime.now().isoformat(),
        )
        training_logger.log_epoch(log_entry)

        # Save periodic checkpoint
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save_periodic_checkpoint(
                q_func=q_func,
                output_dir=Path(args.output_dir),
                epoch=epoch + 1,
                max_checkpoints=args.max_checkpoints,
            )

        # Early stopping
        if args.early_stop > 0 and epochs_no_improve >= args.early_stop:
            print(f"\n  {Colors.YELLOW}Early stopping triggered after {epochs_no_improve} epochs without improvement{Colors.RESET}")
            break

    # Save final model
    output_path = Path(args.output_dir) / "q_function.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    q_func.save(output_path)

    # Determine best path
    best_path_final = Path(args.output_dir) / "q_function_best.pt" if val_loader else None

    # Close TensorBoard writer
    writer.close()

    # Print training summary
    total_time = time.time() - training_start
    print_training_summary(
        total_time=total_time,
        final_train_loss=final_train_loss,
        best_val_loss=best_val_loss,
        output_path=output_path,
        best_path=best_path_final,
        log_dir=log_dir,
        training_log_path=training_logger.log_path,
    )


def main() -> None:
    """Parse arguments and run training."""
    # First, load YAML config to get defaults
    # We need a pre-parser for --config
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    # Load YAML config
    yaml_config = load_training_config(pre_args.config)
    train_cfg = yaml_config.get("training", {})

    # Main parser with YAML defaults
    parser = argparse.ArgumentParser(
        description="Train NeuralQFunction for learned model routing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to training config YAML (default: builtin training.yaml)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        action="append",
        dest="data_dirs",
        help="Directory containing trajectory files (can specify multiple times)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/q_function",
        help="Directory to save trained model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=train_cfg.get("epochs", 100),
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=train_cfg.get("batch_size", 32),
        help="Batch size for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=train_cfg.get("learning_rate", 1e-4),
        help="Learning rate",
    )
    parser.add_argument(
        "--lambda_",
        type=float,
        default=train_cfg.get("lambda_", 0.0),
        help="Legacy cost tradeoff for target ablations; paper/default uses 0.0",
    )
    parser.add_argument(
        "--cost-loss-weight",
        type=float,
        default=train_cfg.get("cost_loss_weight", 0.0),
        help="Legacy dual-head cost loss weight; ignored by the current single-head network",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to load (for debugging)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=train_cfg.get("val_ratio", 0.1),
        help="Validation set ratio (0.0 to disable)",
    )

    # Optimizer settings
    parser.add_argument(
        "--optimizer",
        type=str,
        default=train_cfg.get("optimizer", "adamw"),
        choices=["adam", "adamw"],
        help="Optimizer type",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=train_cfg.get("weight_decay", 0.01),
        help="Weight decay (L2 regularization)",
    )

    # Learning rate scheduler
    parser.add_argument(
        "--scheduler",
        type=str,
        default=train_cfg.get("scheduler", "cosine"),
        choices=["none", "cosine", "plateau"],
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=train_cfg.get("min_lr", 1e-6),
        help="Minimum learning rate for scheduler",
    )

    # Gradient clipping
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=train_cfg.get("grad_clip", 1.0),
        help="Gradient clipping max norm (0 to disable)",
    )

    # Early stopping
    parser.add_argument(
        "--early-stop",
        type=int,
        default=train_cfg.get("early_stop", 10),
        help="Early stopping patience (0 to disable)",
    )

    # Checkpoint settings
    parser.add_argument(
        "--save-every",
        type=int,
        default=train_cfg.get("save_every", 0),
        help="Save checkpoint every N epochs (0 to disable)",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=train_cfg.get("max_checkpoints", 5),
        help="Maximum number of periodic checkpoints to keep",
    )

    # Precomputation settings
    precompute_cfg = yaml_config.get("precompute", {})
    parser.add_argument(
        "--precompute",
        action="store_true",
        default=precompute_cfg.get("enabled", False),
        help="Precompute embeddings with HuggingFace before training (faster training)",
    )
    parser.add_argument(
        "--precompute-batch-size",
        type=int,
        default=precompute_cfg.get("batch_size", 64),
        help="Batch size for HuggingFace precomputation",
    )
    parser.add_argument(
        "--precomputed-path",
        type=Path,
        default=None,
        help="Path to precomputed embeddings file (.pt). Skips online precomputation.",
    )
    parser.add_argument(
        "--model-pool",
        type=str,
        nargs="+",
        default=None,
        help="Override model pool (e.g., --model-pool openai/gpt-5 deepseek/v3.2)",
    )

    # AEC (Annealed Error Cost) settings
    parser.add_argument(
        "--aec-enabled",
        action="store_true",
        default=False,
        help="Enable Annealed Error Cost (AEC) penalty for error detection",
    )
    parser.add_argument(
        "--no-aec",
        action="store_true",
        default=False,
        help="Disable AEC even if enabled in error_rules.yaml",
    )

    # AWR (Advantage-Weighted Regression) settings
    awr_cfg = train_cfg.get("awr", {})
    parser.add_argument(
        "--awr-enabled",
        action="store_true",
        default=awr_cfg.get("enabled", False),
        help="Enable AWR weighted loss for score gap amplification",
    )
    parser.add_argument(
        "--awr-tau",
        type=float,
        default=awr_cfg.get("tau", 1.0),
        help="AWR temperature: smaller = sharper weights (0.1~10.0)",
    )
    parser.add_argument(
        "--awr-min-group",
        type=int,
        default=awr_cfg.get("min_group_size", 2),
        help="Minimum samples per (task_id, step_idx) group for AWR/Margin",
    )

    # Margin Loss settings
    margin_cfg = train_cfg.get("margin", {})
    parser.add_argument(
        "--margin-enabled",
        action="store_true",
        default=margin_cfg.get("enabled", False),
        help="Enable margin loss for enforcing score gaps",
    )
    parser.add_argument(
        "--margin-value",
        type=float,
        default=margin_cfg.get("margin", 0.1),
        help="Target gap between best and other models",
    )
    parser.add_argument(
        "--margin-weight",
        type=float,
        default=margin_cfg.get("weight", 0.1),
        help="Loss weight for margin term",
    )

    args = parser.parse_args()

    # Set default data directories if none specified
    if args.data_dirs is None:
        args.data_dirs = [Path("trajectories/roulette_propensity")]

    # Load model pool from config or CLI override
    if args.model_pool:
        model_names = args.model_pool
    else:
        model_names = get_model_pool(yaml_config)

    # Print model pool info
    print(f"  Model pool ({len(model_names)} models):")
    for i, name in enumerate(model_names):
        print(f"    [{i}] {name}")

    # Load AEC configuration if enabled
    aec_config = None
    error_detector = None
    if args.aec_enabled and not args.no_aec:
        error_detector = ErrorDetector()
        aec_config = error_detector.get_aec_config()
        if aec_config and aec_config.enabled:
            print(f"\n  {Colors.CYAN}AEC (Annealed Error Cost) enabled:{Colors.RESET}")
            print(f"    Warmup range       : p0={aec_config.p0}, p1={aec_config.p1}")
            print(f"    Weight range       : w_min={aec_config.w_min}, w_max={aec_config.w_max}")
            print(f"    Min penalty factor : {aec_config.min_penalty_factor}")
            print(f"    Severity coefs     : {aec_config.severity_coefficients}")
            print(f"    HLE categories     : {len(aec_config.expected_steps.get('hle', {}))}")
            print(f"    SW task names      : {len(aec_config.expected_steps.get('scienceworld', {}))}")
        else:
            print(f"  {Colors.YELLOW}AEC config not found or disabled in error_rules.yaml{Colors.RESET}")
            aec_config = None
            error_detector = None

    train(args, yaml_config, model_names, aec_config=aec_config, error_detector=error_detector)


if __name__ == "__main__":
    main()
