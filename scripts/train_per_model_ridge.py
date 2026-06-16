#!/usr/bin/env python
"""Per-model Ridge baseline for learned router.

Each model gets its own Ridge regressor:
  score_hat_a(z) = w_a · z + b_a
  cost_hat_a(z)  = v_a · z + c_a

This allows: score_hat(z, a1) - score_hat(z, a2) = (w_{a1} - w_{a2}) · z
which depends on the history embedding z, enabling context-aware model selection.

Usage:
    python scripts/train_per_model_ridge.py \
        --precomputed-path embeddings/roulette_propensity.pt \
        --data-dir trajectories/roulette_propensity \
        --output-dir outputs/per_model_ridge

    # With propensity weighting
    python scripts/train_per_model_ridge.py \
        --precomputed-path embeddings/roulette_propensity.pt \
        --data-dir trajectories/roulette_propensity \
        --output-dir outputs/per_model_ridge \
        --use-propensity

    # With multiple data directories
    python scripts/train_per_model_ridge.py \
        --precomputed-path embeddings/combined.pt \
        --data-dir trajectories/roulette_propensity \
        --data-dir trajectories/baseline_openai_gpt-5 \
        --output-dir outputs/per_model_ridge
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import joblib
import numpy as np
import torch
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from miniagenticrouter.research.training.dataset import TrajectoryDataset
from miniagenticrouter.research.training.error_detector import ErrorDetector, AECConfig
from miniagenticrouter.research.utils.config import get_model_pool, load_training_config


# ============================================================
# Terminal Colors
# ============================================================


class Colors:
    """ANSI color codes for terminal output."""

    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_banner() -> None:
    """Print script banner."""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}╔══════════════════════════════════════════════════════════════════╗
║            Per-model Ridge Baseline for Learned Router            ║
╚══════════════════════════════════════════════════════════════════╝{Colors.RESET}
"""
    print(banner)


# ============================================================
# Per-model training
# ============================================================


def train_per_model_ridge(
    X_by_model: dict[int, np.ndarray],
    y_score_by_model: dict[int, np.ndarray],
    y_cost_by_model: dict[int, np.ndarray],
    model_names: list[str],
    weights_by_model: dict[int, np.ndarray] | None = None,
    alpha: float | None = None,
    val_ratio: float = 0.1,
    random_state: int = 42,
) -> tuple[dict[int, Ridge], dict[int, Ridge], dict]:
    """Train per-model Ridge regressors.

    Args:
        X_by_model: {model_idx: (N_a, encoder_dim)} feature matrices.
        y_score_by_model: {model_idx: (N_a,)} score targets.
        y_cost_by_model: {model_idx: (N_a,)} cost targets.
        weights_by_model: Optional {model_idx: (N_a,)} sample weights.
        alpha: Ridge regularization. If None, use cross-validation.
        val_ratio: Validation set ratio.
        random_state: Random state for reproducibility.

    Returns:
        Tuple of (ridge_score_models, ridge_cost_models, metrics).
    """
    ridge_score_models: dict[int, Ridge] = {}
    ridge_cost_models: dict[int, Ridge] = {}
    metrics: dict = {"per_model": {}, "alpha": None}

    # Use the first model with enough data to select alpha.
    if alpha is None:
        print(f"  {Colors.CYAN}Running RidgeCV to select alpha...{Colors.RESET}")
        for model_idx in sorted(X_by_model.keys()):
            X = X_by_model[model_idx]
            y = y_score_by_model[model_idx]
            if len(X) >= 10:  # Enough samples for CV
                cv = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
                cv.fit(X, y)
                alpha = float(cv.alpha_)
                print(f"  Selected alpha: {alpha} (using model {model_names[model_idx]})")
                break
        if alpha is None:
            alpha = 1.0
            print(f"  {Colors.YELLOW}Warning: Not enough data for CV, using default alpha=1.0{Colors.RESET}")

    metrics["alpha"] = alpha

    # Train one Ridge model per routed model.
    for model_idx in sorted(X_by_model.keys()):
        X = X_by_model[model_idx]
        y_score = y_score_by_model[model_idx]
        y_cost = y_cost_by_model[model_idx]
        weights = weights_by_model[model_idx] if weights_by_model else None

        if len(X) < 5:
            print(f"  {Colors.YELLOW}Warning: Model {model_names[model_idx]} has only {len(X)} samples, skipping{Colors.RESET}")
            continue

        # Train/val split
        indices = np.arange(len(X))
        if len(X) < 10:
            # Too little data for a reliable split.
            train_idx = indices
            val_idx = indices
        else:
            train_idx, val_idx = train_test_split(
                indices, test_size=val_ratio, random_state=random_state
            )

        X_train, X_val = X[train_idx], X[val_idx]
        y_score_train, y_score_val = y_score[train_idx], y_score[val_idx]
        y_cost_train, y_cost_val = y_cost[train_idx], y_cost[val_idx]
        w_train = weights[train_idx] if weights is not None else None

        # Train Ridge heads.
        ridge_score = Ridge(alpha=alpha)
        ridge_cost = Ridge(alpha=alpha)

        ridge_score.fit(X_train, y_score_train, sample_weight=w_train)
        ridge_cost.fit(X_train, y_cost_train, sample_weight=w_train)

        ridge_score_models[model_idx] = ridge_score
        ridge_cost_models[model_idx] = ridge_cost

        # Evaluate.
        score_pred_val = ridge_score.predict(X_val)
        cost_pred_val = ridge_cost.predict(X_val)

        model_metrics = {
            "model_name": model_names[model_idx],
            "n_total": len(X),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "score_val_mse": float(mean_squared_error(y_score_val, score_pred_val)),
            "score_val_r2": float(r2_score(y_score_val, score_pred_val)) if len(val_idx) > 1 else 0.0,
            "cost_val_mse": float(mean_squared_error(y_cost_val, cost_pred_val)),
            "cost_val_r2": float(r2_score(y_cost_val, cost_pred_val)) if len(val_idx) > 1 else 0.0,
        }
        metrics["per_model"][model_idx] = model_metrics

        print(f"    Model {model_idx} ({model_names[model_idx]}): "
              f"n={len(X)}, score_R²={model_metrics['score_val_r2']:.4f}, "
              f"cost_R²={model_metrics['cost_val_r2']:.4f}")

    metrics["n_models"] = len(ridge_score_models)

    return ridge_score_models, ridge_cost_models, metrics


# ============================================================
# Main
# ============================================================


def main() -> None:
    """Main entry point."""
    print_banner()

    parser = argparse.ArgumentParser(
        description="Train per-model Ridge baseline for learned router",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--precomputed-path",
        type=Path,
        required=True,
        help="Path to precomputed embeddings (.pt file)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        action="append",
        required=True,
        dest="data_dirs",
        help="Directory containing trajectory files (can specify multiple times)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for trained models and metrics",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Ridge regularization parameter. If not specified, use cross-validation.",
    )
    parser.add_argument(
        "--use-propensity",
        action="store_true",
        help="Use propensity as sample weight (importance weighting)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--model-pool",
        type=str,
        nargs="+",
        default=None,
        help="Override model pool from config (e.g., --model-pool openai/gpt-5 deepseek/v3.2)",
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
    args = parser.parse_args()

    # Get model pool (CLI override or from training.yaml)
    yaml_config = load_training_config()
    if args.model_pool:
        model_names = args.model_pool
    else:
        model_names = get_model_pool(yaml_config)

    # Print configuration
    print(f"  {Colors.YELLOW}Model pool ({len(model_names)} models):{Colors.RESET}")
    for i, name in enumerate(model_names):
        print(f"    [{i}] {name}")
    print(f"  {Colors.YELLOW}Configuration:{Colors.RESET}")
    print(f"    Precomputed path: {args.precomputed_path}")
    print(f"    Data directories: {args.data_dirs}")
    print(f"    Output directory: {args.output_dir}")
    print(f"    Alpha: {args.alpha or 'auto (CV)'}")
    print(f"    Use propensity: {args.use_propensity}")
    print(f"    Validation ratio: {args.val_ratio}")
    print(f"    Random seed: {args.seed}")
    print(f"    AEC enabled: {args.aec_enabled and not args.no_aec}")

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
        else:
            print(f"  {Colors.YELLOW}AEC config not found or disabled in error_rules.yaml{Colors.RESET}")
            aec_config = None
            error_detector = None

    # 1. Load data through TrajectoryDataset to reuse normalization logic.
    print(f"\n{Colors.CYAN}Loading data via TrajectoryDataset...{Colors.RESET}")
    dataset = TrajectoryDataset(
        trajectory_dir=args.data_dirs,
        model_names=model_names,
        lambda_=0.0,  # We need raw score and cost
        precomputed_path=args.precomputed_path,
        error_detector=error_detector,
        aec_config=aec_config,
    )

    # Load encoder config from precomputed file
    precomputed_data = torch.load(args.precomputed_path, weights_only=False)
    encoder_config = precomputed_data.get("config", {})

    if not dataset.precomputed or dataset.embeddings is None:
        print(f"  {Colors.RED}ERROR: Precomputed embeddings not loaded{Colors.RESET}")
        return

    embeddings = dataset.embeddings.numpy()  # (N, encoder_dim)
    encoder_dim = embeddings.shape[1]
    print(f"  Loaded {len(dataset)} samples with embedding dimension {encoder_dim}")

    # 2. Build feature matrices grouped by model.
    print(f"\n{Colors.CYAN}Building per-model feature matrices...{Colors.RESET}")

    X_by_model: dict[int, list] = {i: [] for i in range(len(model_names))}
    y_score_by_model: dict[int, list] = {i: [] for i in range(len(model_names))}
    y_cost_by_model: dict[int, list] = {i: [] for i in range(len(model_names))}
    propensity_by_model: dict[int, list] = {i: [] for i in range(len(model_names))}

    for i, sample in enumerate(dataset.samples):
        model_idx = sample["model_idx"]

        if model_idx < 0 or model_idx >= len(model_names):
            continue

        # Use only the history embedding; do not concatenate model features.
        hist_emb = embeddings[i]  # (encoder_dim,)

        X_by_model[model_idx].append(hist_emb)
        y_score_by_model[model_idx].append(sample["final_score"])
        y_cost_by_model[model_idx].append(sample["normalized_cost"])
        propensity_by_model[model_idx].append(sample.get("propensity") or 1.0)

    # Convert to numpy arrays.
    for model_idx in range(len(model_names)):
        if len(X_by_model[model_idx]) > 0:
            X_by_model[model_idx] = np.array(X_by_model[model_idx])
            y_score_by_model[model_idx] = np.array(y_score_by_model[model_idx])
            y_cost_by_model[model_idx] = np.array(y_cost_by_model[model_idx])
            propensity_by_model[model_idx] = np.array(propensity_by_model[model_idx])
        else:
            X_by_model[model_idx] = np.array([]).reshape(0, encoder_dim)
            y_score_by_model[model_idx] = np.array([])
            y_cost_by_model[model_idx] = np.array([])
            propensity_by_model[model_idx] = np.array([])

    # Print per-model sample counts.
    print(f"\n  Per-model data distribution:")
    total_samples = 0
    for model_idx in range(len(model_names)):
        n = len(X_by_model[model_idx])
        total_samples += n
        print(f"    {model_names[model_idx]}: {n} samples")
    print(f"    Total: {total_samples} samples")

    if total_samples == 0:
        print(f"{Colors.RED}ERROR: No valid samples to train on!{Colors.RESET}")
        sys.exit(1)

    # 4. Train per-model Ridge models.
    print(f"\n{Colors.CYAN}Training per-model Ridge models...{Colors.RESET}")
    weights = propensity_by_model if args.use_propensity else None
    ridge_score_models, ridge_cost_models, metrics = train_per_model_ridge(
        X_by_model,
        y_score_by_model,
        y_cost_by_model,
        model_names,
        weights_by_model=weights,
        alpha=args.alpha,
        val_ratio=args.val_ratio,
        random_state=args.seed,
    )

    # 5. Print results.
    print(f"\n{Colors.GREEN}{Colors.BOLD}Training Results:{Colors.RESET}")
    print(f"  Alpha: {metrics['alpha']}")
    print(f"  Models trained: {metrics['n_models']}")
    print()
    for model_idx, model_metrics in metrics["per_model"].items():
        print(f"  {Colors.BOLD}{model_metrics['model_name']}:{Colors.RESET}")
        print(f"    Samples: {model_metrics['n_total']} (train: {model_metrics['n_train']}, val: {model_metrics['n_val']})")
        print(f"    Score - Val MSE: {model_metrics['score_val_mse']:.4f}, R²: {model_metrics['score_val_r2']:.4f}")
        print(f"    Cost  - Val MSE: {model_metrics['cost_val_mse']:.6f}, R²: {model_metrics['cost_val_r2']:.4f}")

    # 6. Save artifacts.
    print(f"\n{Colors.CYAN}Saving models and metrics...{Colors.RESET}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save per-model Ridge models in a structured format
    models_data = {
        "score_models": {
            model_names[idx]: ridge_score_models[idx]
            for idx in ridge_score_models
        },
        "cost_models": {
            model_names[idx]: ridge_cost_models[idx]
            for idx in ridge_cost_models
        },
        "model_names": model_names,
        "encoder_dim": encoder_dim,
        "encoder_config": encoder_config,  # For inference-time consistency check
    }
    joblib.dump(models_data, args.output_dir / "per_model_ridge.pkl")

    # Save metadata
    meta_info = {
        "model_names": model_names,
        "encoder_dim": encoder_dim,
        "use_propensity": args.use_propensity,
        "created_at": datetime.now().isoformat(),
        "precomputed_path": str(args.precomputed_path),
        "data_dirs": [str(d) for d in args.data_dirs],
    }
    metrics["meta"] = meta_info

    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{Colors.GREEN}{Colors.BOLD}Training complete!{Colors.RESET}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Files saved:")
    print(f"    - per_model_ridge.pkl")
    print(f"    - metrics.json")


if __name__ == "__main__":
    main()
