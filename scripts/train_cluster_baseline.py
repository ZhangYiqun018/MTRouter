#!/usr/bin/env python
"""Cluster baseline for learned router.

Uses KMeans clustering to partition the history embedding space into K regions.
Each region maintains per-model statistics (score_mean, cost_mean).

At inference time:
    1. Encode history -> z
    2. Find nearest cluster center
    3. Look up Q = score_mean - λ * cost_mean

Usage:
    python scripts/train_cluster_baseline.py \
        --precomputed-path embeddings/roulette_propensity.pt \
        --data-dir trajectories/roulette_propensity \
        --output-dir outputs/cluster_baseline

    # With more clusters
    python scripts/train_cluster_baseline.py \
        --precomputed-path embeddings/roulette_propensity.pt \
        --data-dir trajectories/roulette_propensity \
        --output-dir outputs/cluster_baseline \
        --n-clusters 64

    # With propensity weighting
    python scripts/train_cluster_baseline.py \
        --precomputed-path embeddings/roulette_propensity.pt \
        --data-dir trajectories/roulette_propensity \
        --output-dir outputs/cluster_baseline \
        --use-propensity
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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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
║              Cluster Baseline for Learned Router                 ║
╚══════════════════════════════════════════════════════════════════╝{Colors.RESET}
"""
    print(banner)


# ============================================================
# Clustering
# ============================================================


def compute_cluster_stats(
    labels: np.ndarray,
    model_indices: np.ndarray,
    scores: np.ndarray,
    costs: np.ndarray,
    propensities: np.ndarray | None,
    n_clusters: int,
    model_names: list[str],
    use_propensity: bool = False,
) -> dict[str, dict[str, dict]]:
    """Compute per-cluster per-model statistics.

    Args:
        labels: Cluster labels for each sample.
        model_indices: Model index for each sample.
        scores: Final scores for each sample.
        costs: Normalized costs for each sample.
        propensities: Propensity weights (optional).
        n_clusters: Number of clusters.
        model_names: List of model names (used as keys).
        use_propensity: Whether to use propensity weighting.

    Returns:
        Dict: {cluster_id: {model_name: {"score_mean", "cost_mean", "count"}}}
    """
    cluster_stats: dict[str, dict[str, dict]] = {}
    n_models = len(model_names)

    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        cluster_stats[str(cluster_id)] = {}

        for model_idx in range(n_models):
            model_name = model_names[model_idx]
            model_mask = model_indices == model_idx
            combined_mask = cluster_mask & model_mask

            count = int(combined_mask.sum())
            if count == 0:
                # No samples for this cluster-model pair
                cluster_stats[str(cluster_id)][model_name] = {
                    "score_mean": 0.0,
                    "cost_mean": 0.0,
                    "count": 0,
                }
                continue

            cluster_scores = scores[combined_mask]
            cluster_costs = costs[combined_mask]

            if use_propensity and propensities is not None:
                # Weighted mean using inverse propensity
                weights = 1.0 / np.clip(propensities[combined_mask], 0.1, 10.0)
                weights = weights / weights.sum()
                score_mean = float(np.sum(cluster_scores * weights))
                cost_mean = float(np.sum(cluster_costs * weights))
            else:
                score_mean = float(cluster_scores.mean())
                cost_mean = float(cluster_costs.mean())

            cluster_stats[str(cluster_id)][model_name] = {
                "score_mean": score_mean,
                "cost_mean": cost_mean,
                "count": count,
            }

    return cluster_stats


# ============================================================
# Main
# ============================================================


def main() -> None:
    """Main entry point."""
    print_banner()

    parser = argparse.ArgumentParser(
        description="Train cluster baseline for learned router",
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
        "--n-clusters",
        type=int,
        default=32,
        help="Number of clusters",
    )
    parser.add_argument(
        "--use-propensity",
        action="store_true",
        help="Use propensity as sample weight (importance weighting)",
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
    args = parser.parse_args()

    # Load model pool from config or CLI override
    if args.model_pool:
        model_names = args.model_pool
    else:
        yaml_config = load_training_config()
        model_names = get_model_pool(yaml_config)

    # Print configuration
    print(f"  {Colors.YELLOW}Model pool ({len(model_names)} models):{Colors.RESET}")
    for i, name in enumerate(model_names):
        print(f"    [{i}] {name}")
    print(f"\n  {Colors.YELLOW}Configuration:{Colors.RESET}")
    print(f"    Precomputed path: {args.precomputed_path}")
    print(f"    Data directories: {args.data_dirs}")
    print(f"    Output directory: {args.output_dir}")
    print(f"    Number of clusters: {args.n_clusters}")
    print(f"    Use propensity: {args.use_propensity}")
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

    # 2. Build feature matrix and targets
    print(f"\n{Colors.CYAN}Building feature matrix...{Colors.RESET}")

    X_list = []
    model_indices_list = []
    scores_list = []
    costs_list = []
    propensities_list = []

    for i, sample in enumerate(dataset.samples):
        model_idx = sample["model_idx"]

        if model_idx < 0 or model_idx >= len(model_names):
            continue

        X_list.append(embeddings[i])
        model_indices_list.append(model_idx)
        scores_list.append(sample["final_score"])
        costs_list.append(sample["normalized_cost"])
        propensities_list.append(sample.get("propensity") or 1.0)

    X = np.array(X_list)
    model_indices = np.array(model_indices_list)
    scores = np.array(scores_list)
    costs = np.array(costs_list)
    propensities = np.array(propensities_list)

    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Samples per model:")
    for model_idx in range(len(model_names)):
        count = (model_indices == model_idx).sum()
        print(f"    {model_names[model_idx]}: {count}")

    if len(X) == 0:
        print(f"{Colors.RED}ERROR: No valid samples to train on!{Colors.RESET}")
        sys.exit(1)

    # 4. KMeans clustering
    print(f"\n{Colors.CYAN}Running KMeans clustering...{Colors.RESET}")
    kmeans = KMeans(
        n_clusters=args.n_clusters,
        random_state=args.seed,
        n_init=1,
        max_iter=300,
        verbose=1,  # Show iteration progress
    )
    labels = kmeans.fit_predict(X)

    # Compute clustering metrics
    inertia = kmeans.inertia_
    silhouette = silhouette_score(X, labels) if args.n_clusters > 1 else 0.0

    print(f"  Inertia: {inertia:.2f}")
    print(f"  Silhouette score: {silhouette:.4f}")

    # Cluster size distribution
    cluster_sizes = [int((labels == i).sum()) for i in range(args.n_clusters)]
    print(f"  Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, "
          f"mean={np.mean(cluster_sizes):.1f}")

    # 5. Compute per-cluster per-model statistics
    print(f"\n{Colors.CYAN}Computing cluster statistics...{Colors.RESET}")
    cluster_stats = compute_cluster_stats(
        labels=labels,
        model_indices=model_indices,
        scores=scores,
        costs=costs,
        propensities=propensities,
        n_clusters=args.n_clusters,
        model_names=model_names,
        use_propensity=args.use_propensity,
    )

    # Count sparse clusters (cluster-model pairs with 0 samples)
    sparse_count = 0
    for cluster_id in range(args.n_clusters):
        for model_name in model_names:
            if cluster_stats[str(cluster_id)][model_name]["count"] == 0:
                sparse_count += 1

    total_pairs = args.n_clusters * len(model_names)
    print(f"  Sparse cluster-model pairs: {sparse_count}/{total_pairs} "
          f"({100*sparse_count/total_pairs:.1f}%)")

    # 6. Print sample cluster statistics
    print(f"\n{Colors.GREEN}{Colors.BOLD}Sample Cluster Statistics:{Colors.RESET}")
    for cluster_id in range(min(3, args.n_clusters)):  # Show first 3 clusters
        print(f"\n  Cluster {cluster_id} (size={cluster_sizes[cluster_id]}):")
        for model_name in model_names:
            stats = cluster_stats[str(cluster_id)][model_name]
            print(f"    {model_name}: score={stats['score_mean']:.2f}, "
                  f"cost={stats['cost_mean']:.2f}, n={stats['count']}")

    # 7. Save
    print(f"\n{Colors.CYAN}Saving models and metrics...{Colors.RESET}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save KMeans model
    joblib.dump(kmeans, args.output_dir / "cluster_kmeans.pkl")

    # Save cluster statistics
    stats_data = {
        "n_clusters": args.n_clusters,
        "model_names": model_names,
        "encoder_dim": encoder_dim,
        "encoder_config": encoder_config,  # For inference-time consistency check
        "cluster_stats": cluster_stats,
    }
    with open(args.output_dir / "cluster_stats.json", "w") as f:
        json.dump(stats_data, f, indent=2)

    # Save metrics
    metrics = {
        "n_clusters": args.n_clusters,
        "n_samples": len(X),
        "encoder_dim": encoder_dim,
        "inertia": float(inertia),
        "silhouette_score": float(silhouette),
        "cluster_sizes": cluster_sizes,
        "sparse_pairs": sparse_count,
        "use_propensity": args.use_propensity,
        "meta": {
            "model_names": model_names,
            "created_at": datetime.now().isoformat(),
            "precomputed_path": str(args.precomputed_path),
            "data_dirs": [str(d) for d in args.data_dirs],
        },
    }
    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{Colors.GREEN}{Colors.BOLD}Training complete!{Colors.RESET}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Files saved:")
    print(f"    - cluster_kmeans.pkl")
    print(f"    - cluster_stats.json")
    print(f"    - metrics.json")


if __name__ == "__main__":
    main()
