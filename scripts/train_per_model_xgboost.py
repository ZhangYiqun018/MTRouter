#!/usr/bin/env python
"""Per-model XGBoost baseline for learned router.

Each model gets its own XGBRegressor:
  score_hat_a(z) = XGB_score_a(z)
  cost_hat_a(z)  = XGB_cost_a(z)

This allows: score_hat(z, a1) - score_hat(z, a2) depends on the history embedding z,
enabling context-aware model selection.

Uses early_stopping for automatic hyperparameter tuning.

Usage:
    python scripts/train_per_model_xgboost.py \
        --precomputed-path embeddings/roulette_propensity.pt \
        --data-dir trajectories/roulette_propensity \
        --output-dir outputs/per_model_xgboost

    # With propensity weighting
    python scripts/train_per_model_xgboost.py \
        --precomputed-path embeddings/roulette_propensity.pt \
        --data-dir trajectories/roulette_propensity \
        --output-dir outputs/per_model_xgboost \
        --use-propensity

    # With multiple data directories
    python scripts/train_per_model_xgboost.py \
        --precomputed-path embeddings/combined.pt \
        --data-dir trajectories/roulette_propensity \
        --data-dir trajectories/baseline_openai_gpt-5 \
        --output-dir outputs/per_model_xgboost
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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch

try:
    from xgboost import XGBRegressor
    from xgboost.callback import EarlyStopping, TrainingCallback
    _XGBOOST_IMPORT_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as e:
    XGBRegressor = object  # type: ignore[assignment]
    _XGBOOST_IMPORT_ERROR = e

    class TrainingCallback:  # type: ignore[no-redef]
        """Placeholder so --help works when xgboost is not installed."""

        pass

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
║           Per-model XGBoost Baseline for Learned Router          ║
╚══════════════════════════════════════════════════════════════════╝{Colors.RESET}
"""
    print(banner)


# ============================================================
# Training progress callback
# ============================================================


class ProgressCallback(TrainingCallback):
    """Custom callback to show training progress with tqdm."""

    def __init__(self, pbar: tqdm, head_name: str, n_estimators: int):
        self.pbar = pbar
        self.head_name = head_name
        self.n_estimators = n_estimators
        self._current_round = 0

    def after_iteration(self, model, epoch: int, evals_log: dict) -> bool:
        self._current_round = epoch + 1
        # Update progress bar description with current validation metric
        if evals_log:
            for data_name, metric_dict in evals_log.items():
                for metric_name, values in metric_dict.items():
                    if values:
                        latest_val = values[-1]
                        self.pbar.set_postfix_str(
                            f"{self.head_name}: iter {self._current_round}, val_{metric_name}={latest_val:.4f}"
                        )
        return False  # False means continue training


# ============================================================
# Per-model XGBoost training
# ============================================================


def train_per_model_xgboost(
    X_by_model: dict[int, np.ndarray],
    y_score_by_model: dict[int, np.ndarray],
    y_cost_by_model: dict[int, np.ndarray],
    model_names: list[str],
    weights_by_model: dict[int, np.ndarray] | None = None,
    xgb_params: dict | None = None,
    val_ratio: float = 0.15,
    early_stopping_rounds: int = 30,
    random_state: int = 42,
) -> tuple[dict[int, XGBRegressor], dict[int, XGBRegressor], dict]:
    """Train per-model XGBoost regressors.

    Args:
        X_by_model: {model_idx: (N_a, encoder_dim)} feature matrices.
        y_score_by_model: {model_idx: (N_a,)} score targets.
        y_cost_by_model: {model_idx: (N_a,)} cost targets.
        model_names: List of model names for display.
        weights_by_model: Optional {model_idx: (N_a,)} sample weights.
        xgb_params: XGBoost parameters dict.
        val_ratio: Validation set ratio for early stopping.
        early_stopping_rounds: Number of rounds for early stopping.
        random_state: Random state for reproducibility.

    Returns:
        Tuple of (xgb_score_models, xgb_cost_models, metrics).
    """
    xgb_score_models: dict[int, XGBRegressor] = {}
    xgb_cost_models: dict[int, XGBRegressor] = {}
    metrics: dict = {"per_model": {}, "xgb_params": xgb_params}

    # Count valid models for progress bar
    valid_model_indices = [
        idx for idx in sorted(X_by_model.keys())
        if len(X_by_model[idx]) >= 10
    ]
    n_estimators = xgb_params.get("n_estimators", 500)

    # Train one XGBoost model per routed model with a progress bar.
    with tqdm(
        valid_model_indices,
        desc="Training models",
        unit="model",
        ncols=100,
    ) as model_pbar:
        for model_idx in model_pbar:
            X = X_by_model[model_idx]
            y_score = y_score_by_model[model_idx]
            y_cost = y_cost_by_model[model_idx]
            weights = weights_by_model[model_idx] if weights_by_model else None

            model_name_short = model_names[model_idx].split("/")[-1][:15]
            model_pbar.set_description(f"Training {model_name_short}")

            # Train/val split for early stopping
            indices = np.arange(len(X))
            train_idx, val_idx = train_test_split(
                indices, test_size=val_ratio, random_state=random_state
            )

            X_train, X_val = X[train_idx], X[val_idx]
            y_score_train, y_score_val = y_score[train_idx], y_score[val_idx]
            y_cost_train, y_cost_val = y_cost[train_idx], y_cost[val_idx]
            w_train = weights[train_idx] if weights is not None else None
            w_val = weights[val_idx] if weights is not None else None

            # Create XGBoost models with early stopping
            xgb_score = XGBRegressor(
                **xgb_params,
                random_state=random_state,
                early_stopping_rounds=early_stopping_rounds,
                callbacks=[ProgressCallback(model_pbar, "score", n_estimators)],
            )
            xgb_cost = XGBRegressor(
                **xgb_params,
                random_state=random_state,
                early_stopping_rounds=early_stopping_rounds,
                callbacks=[ProgressCallback(model_pbar, "cost", n_estimators)],
            )

            # Train score model
            model_pbar.set_postfix_str("training score head...")
            xgb_score.fit(
                X_train, y_score_train,
                sample_weight=w_train,
                eval_set=[(X_val, y_score_val)],
                sample_weight_eval_set=[w_val] if w_val is not None else None,
                verbose=False,
            )

            # Train cost model
            model_pbar.set_postfix_str("training cost head...")
            xgb_cost.fit(
                X_train, y_cost_train,
                sample_weight=w_train,
                eval_set=[(X_val, y_cost_val)],
                sample_weight_eval_set=[w_val] if w_val is not None else None,
                verbose=False,
            )

            xgb_score_models[model_idx] = xgb_score
            xgb_cost_models[model_idx] = xgb_cost

            # Evaluate.
            score_pred_val = xgb_score.predict(X_val)
            cost_pred_val = xgb_cost.predict(X_val)

            model_metrics = {
                "model_name": model_names[model_idx],
                "n_total": len(X),
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "score_best_iteration": xgb_score.best_iteration,
                "cost_best_iteration": xgb_cost.best_iteration,
                "score_val_mse": float(mean_squared_error(y_score_val, score_pred_val)),
                "score_val_rmse": float(np.sqrt(mean_squared_error(y_score_val, score_pred_val))),
                "score_val_r2": float(r2_score(y_score_val, score_pred_val)) if len(val_idx) > 1 else 0.0,
                "cost_val_mse": float(mean_squared_error(y_cost_val, cost_pred_val)),
                "cost_val_rmse": float(np.sqrt(mean_squared_error(y_cost_val, cost_pred_val))),
                "cost_val_r2": float(r2_score(y_cost_val, cost_pred_val)) if len(val_idx) > 1 else 0.0,
            }
            metrics["per_model"][model_idx] = model_metrics

            # Update progress bar with final metrics
            model_pbar.set_postfix_str(
                f"score_R²={model_metrics['score_val_r2']:.3f}, "
                f"cost_R²={model_metrics['cost_val_r2']:.3f}"
            )

    # Print skipped models
    skipped_models = [idx for idx in sorted(X_by_model.keys()) if len(X_by_model[idx]) < 10]
    for model_idx in skipped_models:
        print(f"  {Colors.YELLOW}Warning: Model {model_names[model_idx]} has only {len(X_by_model[model_idx])} samples, skipped{Colors.RESET}")

    metrics["n_models"] = len(xgb_score_models)

    return xgb_score_models, xgb_cost_models, metrics


# ============================================================
# Main
# ============================================================


def main() -> None:
    """Main entry point."""
    print_banner()

    parser = argparse.ArgumentParser(
        description="Train per-model XGBoost baseline for learned router",
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
        "--use-propensity",
        action="store_true",
        help="Use propensity as sample weight (importance weighting)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio for early stopping",
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

    # XGBoost specific parameters
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=500,
        help="Maximum number of boosting rounds",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum tree depth",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Boosting learning rate (eta)",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=30,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--min-child-weight",
        type=float,
        default=5.0,
        help="Minimum sum of instance weight in a child",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.8,
        help="Subsample ratio of training instances",
    )
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=0.8,
        help="Subsample ratio of columns when constructing each tree",
    )
    parser.add_argument(
        "--reg-alpha",
        type=float,
        default=0.1,
        help="L1 regularization term on weights",
    )
    parser.add_argument(
        "--reg-lambda",
        type=float,
        default=1.0,
        help="L2 regularization term on weights",
    )
    parser.add_argument(
        "--tree-method",
        type=str,
        default="gpu_hist",
        choices=["hist", "exact", "approx", "gpu_hist"],
        help="XGBoost tree construction algorithm",
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

    if _XGBOOST_IMPORT_ERROR is not None:
        raise SystemExit(
            "xgboost is required for this trainer. Install the research extras "
            "with `pip install -e '.[research]'`, or install `xgboost` directly."
        ) from _XGBOOST_IMPORT_ERROR

    # Build XGBoost parameters dict
    xgb_params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "tree_method": args.tree_method,
        "objective": "reg:squarederror",
    }
    # GPU doesn't use n_jobs; only set for CPU methods
    if args.tree_method != "gpu_hist":
        xgb_params["n_jobs"] = -1

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
    print(f"    Use propensity: {args.use_propensity}")
    print(f"    Validation ratio: {args.val_ratio}")
    print(f"    Random seed: {args.seed}")
    print(f"  {Colors.YELLOW}XGBoost parameters:{Colors.RESET}")
    print(f"    n_estimators: {args.n_estimators}")
    print(f"    max_depth: {args.max_depth}")
    print(f"    learning_rate: {args.learning_rate}")
    print(f"    early_stopping_rounds: {args.early_stopping_rounds}")
    print(f"    min_child_weight: {args.min_child_weight}")
    print(f"    subsample: {args.subsample}")
    print(f"    colsample_bytree: {args.colsample_bytree}")
    print(f"    reg_alpha: {args.reg_alpha}")
    print(f"    reg_lambda: {args.reg_lambda}")
    print(f"    tree_method: {args.tree_method}")
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

    # 4. Train per-model XGBoost models.
    print(f"\n{Colors.CYAN}Training per-model XGBoost models...{Colors.RESET}")
    weights = propensity_by_model if args.use_propensity else None
    xgb_score_models, xgb_cost_models, metrics = train_per_model_xgboost(
        X_by_model,
        y_score_by_model,
        y_cost_by_model,
        model_names,
        weights_by_model=weights,
        xgb_params=xgb_params,
        val_ratio=args.val_ratio,
        early_stopping_rounds=args.early_stopping_rounds,
        random_state=args.seed,
    )

    # 5. Print results.
    print(f"\n{Colors.GREEN}{Colors.BOLD}Training Results:{Colors.RESET}")
    print(f"  Models trained: {metrics['n_models']}")
    print()
    for model_idx, model_metrics in metrics["per_model"].items():
        print(f"  {Colors.BOLD}{model_metrics['model_name']}:{Colors.RESET}")
        print(f"    Samples: {model_metrics['n_total']} (train: {model_metrics['n_train']}, val: {model_metrics['n_val']})")
        print(f"    Score - Val RMSE: {model_metrics['score_val_rmse']:.4f}, R²: {model_metrics['score_val_r2']:.4f}, best_iter: {model_metrics['score_best_iteration']}")
        print(f"    Cost  - Val RMSE: {model_metrics['cost_val_rmse']:.6f}, R²: {model_metrics['cost_val_r2']:.4f}, best_iter: {model_metrics['cost_best_iteration']}")

    # 6. Save artifacts.
    print(f"\n{Colors.CYAN}Saving models and metrics...{Colors.RESET}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save per-model XGBoost models in a structured format
    models_data = {
        "score_models": {
            model_names[idx]: xgb_score_models[idx]
            for idx in xgb_score_models
        },
        "cost_models": {
            model_names[idx]: xgb_cost_models[idx]
            for idx in xgb_cost_models
        },
        "model_names": model_names,
        "encoder_dim": encoder_dim,
        "encoder_config": encoder_config,  # For inference-time consistency check
        "xgb_params": xgb_params,
    }
    joblib.dump(models_data, args.output_dir / "per_model_xgboost.pkl")

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
    print(f"    - per_model_xgboost.pkl")
    print(f"    - metrics.json")


if __name__ == "__main__":
    main()
