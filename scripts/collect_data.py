#!/usr/bin/env python3
"""Data collection script for research experiments.

Usage:
    # Dry run (see what would be collected)
    python scripts/collect_data.py --dry-run

    # Collect with Roulette mode (default, uses config from data_split.yaml)
    python scripts/collect_data.py --split train

    # Collect baseline for a single model
    python scripts/collect_data.py --mode baseline --model-index 0

    # Override YAML config with CLI args
    python scripts/collect_data.py --runs 5 --workers 8

    # Collect on a specific task only
    python scripts/collect_data.py --task-filter "boil" --runs 1
"""

# Disable tokenizers parallelism to avoid deadlocks in multiprocessing
# Must be set BEFORE importing transformers/tokenizers
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load local .env BEFORE importing miniagenticrouter (which loads global config)
from pathlib import Path

from dotenv import load_dotenv

_local_env = Path(__file__).parent.parent / ".env"
if _local_env.exists():
    load_dotenv(_local_env, override=True)

import argparse

import yaml

from miniagenticrouter.research import (
    BaselineMode,
    ClusterMode,
    CollectionConfig,
    Collector,
    DataSplit,
    HeuristicMode,
    LearnedMode,
    LLMRouterMode,
    MixedMode,
    PerModelRidgeMode,
    PerModelXGBoostMode,
    RouletteMode,
)
from miniagenticrouter.research.utils.config import (
    get_default_config_path,
    get_model_pool,
    load_training_config,
)


def load_collection_config(path: Path | None = None) -> dict:
    """Load collection config from data_split.yaml."""
    if path is None:
        path = get_default_config_path()
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("collection", {})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect research trajectories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be collected
  python scripts/collect_data.py --first-try --dry-run

  # Collect training data with roulette mode
  python scripts/collect_data.py --first-try --split train

  # Collect baseline with model 0 only
  python scripts/collect_data.py --first-try --mode baseline --model-index 0

  # Test on a single task
  python scripts/collect_data.py --first-try --task-filter "boil" --runs 1
""",
    )
    parser.add_argument(
        "--mode",
        choices=["roulette", "baseline", "learned", "mixed", "per_model_ridge", "per_model_xgboost", "cluster", "heuristic", "llmrouter"],
        default="roulette",
        help="Collection mode (default: roulette)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Q-function checkpoint for learned/mixed mode",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Model directory for ridge mode (default: outputs/ridge_baseline)",
    )
    parser.add_argument(
        "--lambda",
        type=float,
        default=1.0,
        dest="lambda_",
        help="Cost penalty coefficient for ridge mode (default: 1.0)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="Learned policy ratio for mixed mode (0-1, default: 0.5)",
    )
    parser.add_argument(
        "--use-batching",
        action="store_true",
        help="Enable dynamic batching for learned mode inference",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Max batch size for batched inference (default: 8)",
    )
    parser.add_argument(
        "--batch-timeout",
        type=float,
        default=0.02,
        help="Timeout in seconds to wait for batch to fill (default: 0.02)",
    )
    parser.add_argument(
        "--task-type",
        choices=["scienceworld", "hle"],
        default="scienceworld",
        help="Task type for heuristic mode (default: scienceworld)",
    )
    parser.add_argument(
        "--policy-model",
        type=str,
        default="openai/gpt-5",
        help="Policy model for llmrouter mode (default: openai/gpt-5)",
    )
    parser.add_argument(
        "--history-turns",
        type=int,
        default=3,
        help="Number of conversation turns to show for llmrouter (0=all, default: 3)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test_id", "ood_test"],
        default="train",
        help="Data split to collect (default: train)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for baseline mode (e.g., 'deepseek/deepseek-v3.2')",
    )
    # These override YAML config if specified
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Runs per variation (default: from YAML)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel workers (default: 4)",
    )
    parser.add_argument(
        "--step-limit",
        type=int,
        default=None,
        help="Max steps per episode (default: from YAML)",
    )
    parser.add_argument(
        "--cost-limit",
        type=float,
        default=None,
        help="Max cost per episode in USD (default: from YAML)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./trajectories"),
        help="Output directory (default: ./trajectories)",
    )
    parser.add_argument(
        "--task-filter",
        type=str,
        default=None,
        help="Filter tasks by name pattern (e.g., 'boil', 'chem*')",
    )
    parser.add_argument(
        "--first-try",
        action="store_true",
        help="Use first-try subset (9 tasks instead of 13)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be collected without actually running",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing trajectories (default: skip existing)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to data_split.yaml (default: builtin)",
    )
    parser.add_argument(
        "--model-pool",
        type=str,
        nargs="+",
        default=None,
        help="Override model pool (e.g., --model-pool openai/gpt-5 deepseek/v3.2)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load model pool from config or CLI override
    if args.model_pool:
        model_names = args.model_pool
    else:
        training_config = load_training_config()
        model_names = get_model_pool(training_config)

    print(f"Model pool ({len(model_names)} models):")
    for i, name in enumerate(model_names):
        print(f"  [{i}] {name}")

    # Load YAML config
    yaml_collection = load_collection_config(args.config)
    yaml_agent = yaml_collection.get("agent", {})
    yaml_mode = yaml_collection.get(args.mode, {})

    # Determine final config values (CLI overrides YAML)
    runs_per_variation = (
        args.runs
        if args.runs is not None
        else yaml_mode.get("runs_per_variation", 3)
    )
    step_limit = (
        args.step_limit
        if args.step_limit is not None
        else yaml_agent.get("step_limit", 50)
    )
    cost_limit = (
        args.cost_limit
        if args.cost_limit is not None
        else yaml_agent.get("cost_limit", 5.0)
    )
    record_propensity = yaml_mode.get("record_propensity", args.mode == "roulette")

    # Load data split
    print(f"Loading data split (first_try={args.first_try})...")
    split = DataSplit.from_yaml(path=args.config, use_first_try=args.first_try)

    # Show model configuration only for modes that use data_split model_configs
    # (roulette and baseline modes use roulette_model_pool; learned/mixed/ridge/cluster use training.yaml model_pool)
    if args.mode in ("roulette", "baseline"):
        model_configs = split.get_model_configs()
        print(f"\nRoulette model pool ({len(model_configs)} models):")
        for i, cfg in enumerate(model_configs):
            print(f"  [{i}] {cfg['model_name']}")
    else:
        # For learned/mixed/ridge/cluster modes, model_pool was already printed above
        model_configs = None

    # Create collection mode
    if args.mode == "roulette":
        mode = RouletteMode(record_propensity=record_propensity)
        print(f"\nMode: Roulette (random model selection, propensity={record_propensity})")
    elif args.mode == "learned":
        # Learned mode: use trained Q-function
        if not args.checkpoint:
            print("Error: --checkpoint required for learned mode")
            return
        if not args.checkpoint.exists():
            print(f"Error: Checkpoint not found: {args.checkpoint}")
            return
        # Use fixed model names order (must match training order)
        mode = LearnedMode(
            checkpoint=args.checkpoint,
            model_names=model_names,
            use_batching=args.use_batching,
            batch_size=args.batch_size,
            timeout=args.batch_timeout,
        )
        batching_info = f", batching={args.use_batching}" if args.use_batching else ""
        print(f"\nMode: Learned (checkpoint: {args.checkpoint.name}{batching_info})")
    elif args.mode == "mixed":
        # Mixed mode: beta% learned + (1-beta)% roulette
        if not args.checkpoint:
            print("Error: --checkpoint required for mixed mode")
            return
        if not args.checkpoint.exists():
            print(f"Error: Checkpoint not found: {args.checkpoint}")
            return
        if not 0.0 <= args.beta <= 1.0:
            print(f"Error: --beta must be in [0, 1], got {args.beta}")
            return
        # Use fixed model names order (must match training order)
        mode = MixedMode(
            checkpoint=args.checkpoint,
            model_names=model_names,
            beta=args.beta,
            use_batching=args.use_batching,
            batch_size=args.batch_size,
            timeout=args.batch_timeout,
        )
        batching_info = f", batching={args.use_batching}" if args.use_batching else ""
        print(f"\nMode: Mixed (beta={args.beta}, checkpoint: {args.checkpoint.name}{batching_info})")
        print(f"  Strategy: {int(args.beta*100)}% learned + {int((1-args.beta)*100)}% roulette")
    elif args.mode == "per_model_ridge":
        # Per-model Ridge mode: each model has its own Ridge regressor
        model_dir = args.model_dir or Path("outputs/per_model_ridge")
        if not model_dir.exists():
            print(f"Error: Model directory not found: {model_dir}")
            return
        mode = PerModelRidgeMode(
            model_dir=model_dir,
            lambda_=args.lambda_,
            use_batching=args.use_batching,
            batch_size=args.batch_size,
            timeout=args.batch_timeout,
        )
        batching_info = f", batching={args.use_batching}" if args.use_batching else ""
        print(f"\nMode: Per-model Ridge (model_dir: {model_dir.name}, λ={args.lambda_}{batching_info})")
    elif args.mode == "per_model_xgboost":
        # Per-model XGBoost mode: each model has its own XGBoost regressor
        model_dir = args.model_dir or Path("outputs/per_model_xgboost")
        if not model_dir.exists():
            print(f"Error: Model directory not found: {model_dir}")
            return
        mode = PerModelXGBoostMode(
            model_dir=model_dir,
            lambda_=args.lambda_,
            use_batching=args.use_batching,
            batch_size=args.batch_size,
            timeout=args.batch_timeout,
        )
        batching_info = f", batching={args.use_batching}" if args.use_batching else ""
        print(f"\nMode: Per-model XGBoost (model_dir: {model_dir.name}, λ={args.lambda_}{batching_info})")
    elif args.mode == "cluster":
        # Cluster mode: KMeans clustering for state-dependent routing
        model_dir = args.model_dir or Path("outputs/cluster_baseline")
        if not model_dir.exists():
            print(f"Error: Model directory not found: {model_dir}")
            return
        mode = ClusterMode(
            model_dir=model_dir,
            lambda_=args.lambda_,
            use_batching=args.use_batching,
            batch_size=args.batch_size,
            timeout=args.batch_timeout,
        )
        batching_info = f", batching={args.use_batching}" if args.use_batching else ""
        print(f"\nMode: Cluster (model_dir: {model_dir.name}, λ={args.lambda_}{batching_info})")
    elif args.mode == "heuristic":
        # Heuristic mode: rule-based model selection
        mode = HeuristicMode(
            task_type=args.task_type,
        )
        print(f"\nMode: Heuristic (task_type: {args.task_type})")
        print(f"  Strong model: {mode.strong_model}")
        print(f"  Value model: {mode.value_model}")
        print(f"  Cheap pool: {mode.cheap_pool}")
        print(f"  Max steps: {mode.max_steps}")
    elif args.mode == "llmrouter":
        # LLMRouter mode: LLM-based model selection
        mode = LLMRouterMode(
            policy_model_name=args.policy_model,
            history_turns=args.history_turns,
        )
        print(f"\nMode: LLMRouter (policy: {args.policy_model}, history_turns: {args.history_turns})")
    else:
        # Baseline mode: use specified model or default to first
        model_name = args.model or model_configs[0]["model_name"]
        available = [cfg["model_name"] for cfg in model_configs]
        # Allow certain models to bypass validation (e.g., openrouter/auto)
        passthrough_models = {"openrouter/auto"}
        if model_name not in available and model_name not in passthrough_models:
            print(f"Error: Model '{model_name}' not found.")
            print(f"Available: {available}")
            return
        mode = BaselineMode(model_name=model_name)
        print(f"\nMode: Baseline (model: {model_name})")

    # Create collector config
    config = CollectionConfig(
        output_dir=args.output_dir,
        runs_per_variation=runs_per_variation,
        step_limit=step_limit,
        cost_limit=cost_limit,
        workers=args.workers,
        record_propensity=record_propensity,
        skip_existing=not args.force,
    )

    print(f"\nCollection config (from YAML + CLI overrides):")
    print(f"  Output: {config.output_dir}")
    print(f"  Split: {args.split}")
    print(f"  Runs per variation: {config.runs_per_variation}")
    print(f"  Step limit: {config.step_limit}")
    print(f"  Cost limit: ${config.cost_limit}")
    print(f"  Workers: {config.workers}")
    print(f"  Skip existing: {config.skip_existing}")

    # Create collector
    collector = Collector(
        data_split=split,
        mode=mode,
        config=config,
    )

    # Dry run or actual collection
    if args.dry_run:
        print("\n" + "=" * 60)
        collector.collect(
            split=args.split,
            task_filter=args.task_filter,
            dry_run=True,
        )
        print("=" * 60)
        print("\nThis was a dry run. Remove --dry-run to actually collect data.")
        return

    # Run collection (progress bar is handled internally by Collector)
    print("\n")  # Clear line for Rich
    results = collector.collect(
        split=args.split,
        task_filter=args.task_filter,
    )

    # Summary
    print("\n" + "=" * 60)
    print(collector.summary(results))
    print("=" * 60)

    # Save summary to file
    if results:
        summary_path = collector.save_summary(results, split=args.split)
        print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
