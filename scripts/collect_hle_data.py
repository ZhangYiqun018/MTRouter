#!/usr/bin/env python3
"""HLE data collection script for research experiments.

Usage:
    # Dry run (see what would be collected)
    python scripts/collect_hle_data.py --dry-run

    # Collect with Roulette mode (random model selection)
    python scripts/collect_hle_data.py --split train

    # Collect baseline for a single model
    python scripts/collect_hle_data.py --mode baseline --model openai/gpt-5

    # Override default config with CLI args
    python scripts/collect_hle_data.py --runs 3 --workers 8

    # Collect on a specific category only
    python scripts/collect_hle_data.py --category-filter "Math"
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

from miniagenticrouter.research.data import HLEDataSplit
from miniagenticrouter.research.collection import (
    BaselineMode,
    ClusterMode,
    HeuristicMode,
    LearnedMode,
    LLMRouterMode,
    MixedMode,
    PerModelRidgeMode,
    PerModelXGBoostMode,
    RouletteMode,
)
from miniagenticrouter.research.collection.hle_collector import (
    HLECollectionConfig,
    HLECollector,
)
from miniagenticrouter.research.utils.config import (
    get_model_pool,
    load_training_config,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect HLE research trajectories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be collected
  python scripts/collect_hle_data.py --dry-run

  # Collect training data with roulette mode
  python scripts/collect_hle_data.py --split train

  # Collect baseline with a specific model
  python scripts/collect_hle_data.py --mode baseline --model openai/gpt-5

  # Test on Math category only
  python scripts/collect_hle_data.py --category-filter "Math" --runs 1

  # Collect a single task by ID
  python scripts/collect_hle_data.py --task-id 6755d8a01c505b5224374708 --runs 1

  # Collect first 10 tasks only
  python scripts/collect_hle_data.py --limit 10 --runs 1
""",
    )
    parser.add_argument(
        "--mode",
        choices=["roulette", "baseline", "heuristic", "learned", "mixed",
                 "per_model_ridge", "per_model_xgboost", "cluster", "llmrouter"],
        default="roulette",
        help="Collection mode (default: roulette)",
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
        help="Model name for baseline mode (e.g., 'openai/gpt-5')",
    )
    # Learned/Mixed mode arguments
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Q-function checkpoint for learned/mixed mode",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="Learned policy ratio for mixed mode (0-1, default: 0.5)",
    )
    parser.add_argument(
        "--lambda",
        type=float,
        default=1.0,
        dest="lambda_",
        help="Cost penalty coefficient (default: 1.0)",
    )
    # Ridge/XGBoost/Cluster mode arguments
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Model directory for ridge/xgboost/cluster mode",
    )
    # Batching arguments
    parser.add_argument(
        "--use-batching",
        action="store_true",
        help="Enable dynamic batching for inference",
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
    # Model pool override
    parser.add_argument(
        "--model-pool",
        type=str,
        nargs="+",
        default=None,
        help="Override model pool (e.g., --model-pool openai/gpt-5 deepseek/v3.2)",
    )
    # LLMRouter mode arguments
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
        "--runs",
        type=int,
        default=3,
        help="Runs per task (default: 3)",
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
        help="Max steps per episode (default: from hle.yaml)",
    )
    parser.add_argument(
        "--cost-limit",
        type=float,
        default=None,
        help="Max cost per episode in USD (default: from hle.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./hle_trajectories"),
        help="Output directory (default: ./hle_trajectories)",
    )
    parser.add_argument(
        "--category-filter",
        type=str,
        default=None,
        help="Filter tasks by category (e.g., 'Math', 'Physics')",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Collect a specific task by ID",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tasks to collect",
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
        help="Path to hle_data_split.yaml (default: builtin)",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Disable LLM Judge evaluation",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Model name for LLM Judge (default: from config)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load HLE data split
    print(f"Loading HLE data split...")
    split = HLEDataSplit.from_yaml(path=args.config)
    split.initialize_splits()

    # Show split summary
    print(split.summary())

    # Get model configs from data split
    model_configs = split.get_model_configs()
    print(f"\nModel pool ({len(model_configs)} models):")
    for i, cfg in enumerate(model_configs):
        print(f"  [{i}] {cfg['model_name']}")

    # Load model pool for learned/mixed/ridge/xgboost/cluster modes
    if args.model_pool:
        model_names = args.model_pool
    else:
        training_config = load_training_config()
        model_names = get_model_pool(training_config)

    # Create collection mode
    if args.mode == "roulette":
        mode = RouletteMode(record_propensity=True)
        print(f"\nMode: Roulette (random model selection)")
    elif args.mode == "heuristic":
        # Heuristic mode: rule-based model selection
        mode = HeuristicMode(task_type="hle")
        print(f"\nMode: Heuristic (task_type: hle)")
        print(f"  Strong model: {mode.strong_model}")
        print(f"  Value model: {mode.value_model}")
        print(f"  Cheap pool: {mode.cheap_pool}")
        print(f"  Max steps: {mode.max_steps}")
    elif args.mode == "learned":
        # Learned mode: use trained Q-function
        if not args.checkpoint:
            print("Error: --checkpoint required for learned mode")
            return
        if not args.checkpoint.exists():
            print(f"Error: Checkpoint not found: {args.checkpoint}")
            return
        mode = LearnedMode(
            checkpoint=args.checkpoint,
            model_names=model_names,
            lambda_=args.lambda_,
            use_batching=args.use_batching,
            batch_size=args.batch_size,
            timeout=args.batch_timeout,
        )
        batching_info = f", batching={args.use_batching}" if args.use_batching else ""
        print(f"\nMode: Learned (checkpoint: {args.checkpoint.name}, λ={args.lambda_}{batching_info})")
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
        mode = MixedMode(
            checkpoint=args.checkpoint,
            model_names=model_names,
            beta=args.beta,
            lambda_=args.lambda_,
            use_batching=args.use_batching,
            batch_size=args.batch_size,
            timeout=args.batch_timeout,
        )
        batching_info = f", batching={args.use_batching}" if args.use_batching else ""
        print(f"\nMode: Mixed (beta={args.beta}, checkpoint: {args.checkpoint.name}{batching_info})")
        print(f"  Strategy: {int(args.beta*100)}% learned + {int((1-args.beta)*100)}% roulette")
    elif args.mode == "per_model_ridge":
        # Per-model Ridge mode
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
        # Per-model XGBoost mode
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
        # Cluster mode: KMeans clustering
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
    elif args.mode == "llmrouter":
        # LLMRouter mode: LLM-based model selection
        mode = LLMRouterMode(
            policy_model_name=args.policy_model,
            history_turns=args.history_turns,
        )
        print(f"\nMode: LLMRouter (policy: {args.policy_model}, history_turns: {args.history_turns})")
    else:
        # Baseline mode
        model_name = args.model
        if model_name is None:
            if model_configs:
                model_name = model_configs[0]["model_name"]
            else:
                print("Error: No models configured and --model not specified")
                return
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
    config = HLECollectionConfig(
        output_dir=args.output_dir,
        runs_per_task=args.runs,
        step_limit=args.step_limit,
        cost_limit=args.cost_limit,
        workers=args.workers,
        skip_existing=not args.force,
        enable_judge=not args.no_judge,
        judge_model=args.judge_model,
    )

    # Get effective step/cost limits (from yaml if not specified)
    effective_step_limit = config.step_limit if config.step_limit is not None else "(from hle.yaml)"
    effective_cost_limit = f"${config.cost_limit}" if config.cost_limit is not None else "(from hle.yaml)"

    print(f"\nCollection config:")
    print(f"  Output: {config.output_dir}")
    print(f"  Split: {args.split}")
    print(f"  Runs per task: {config.runs_per_task}")
    print(f"  Step limit: {effective_step_limit}")
    print(f"  Cost limit: {effective_cost_limit}")
    print(f"  Workers: {config.workers}")
    print(f"  Skip existing: {config.skip_existing}")
    print(f"  Enable Judge: {config.enable_judge}")

    # Create collector
    collector = HLECollector(
        data_split=split,
        mode=mode,
        config=config,
    )

    # Dry run or actual collection
    if args.dry_run:
        print("\n" + "=" * 60)
        collector.collect(
            split=args.split,
            category_filter=args.category_filter,
            task_id=args.task_id,
            limit=args.limit,
            dry_run=True,
        )
        print("=" * 60)
        print("\nThis was a dry run. Remove --dry-run to actually collect data.")
        return

    # Run collection
    print("\n")
    results = collector.collect(
        split=args.split,
        category_filter=args.category_filter,
        task_id=args.task_id,
        limit=args.limit,
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
