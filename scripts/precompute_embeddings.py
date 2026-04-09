#!/usr/bin/env python
"""Standalone embedding precomputation script.

This script precomputes embeddings for trajectory data and saves them to disk.
The precomputed embeddings can then be loaded during training for faster startup.

Supports two backends:
- HuggingFace (hf): Uses local HuggingFace model
- vLLM (vllm): Uses vLLM server via HTTP API

Usage:
    # Precompute embeddings using HuggingFace (default)
    python scripts/precompute_embeddings.py \
        --data-dir trajectories/roulette_propensity \
        --output embeddings/roulette_propensity.pt

    # Precompute embeddings using vLLM server
    python scripts/precompute_embeddings.py \
        --data-dir trajectories/roulette_propensity \
        --output embeddings/roulette_propensity_vllm.pt \
        --backend vllm \
        --vllm-base-url http://localhost:8000

    # With caching and parallel workers (vLLM only)
    python scripts/precompute_embeddings.py \
        --data-dir trajectories/roulette_propensity \
        --output embeddings/roulette.pt \
        --backend vllm \
        --cache-path embeddings/cache.db \
        --max-workers 16

    # Precompute embeddings for multiple directories
    python scripts/precompute_embeddings.py \
        --data-dir trajectories/roulette_propensity \
        --data-dir trajectories/mixed_b1 \
        --output embeddings/combined.pt

    # With custom batch size
    python scripts/precompute_embeddings.py \
        --data-dir trajectories/roulette_propensity \
        --output embeddings/roulette.pt \
        --batch-size 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml

from miniagenticrouter.research.training.dataset import (
    TrajectoryDataset,
)
from miniagenticrouter.research.training.encoders import (
    PrecomputeConfig,
    create_precomputer,
)
from miniagenticrouter.research.utils.config import get_model_pool


class Colors:
    """ANSI color codes for terminal output."""
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


def print_banner() -> None:
    """Print script banner."""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}╔══════════════════════════════════════════════════════════════════╗
║              Standalone Embedding Precomputation                 ║
╚══════════════════════════════════════════════════════════════════╝{Colors.RESET}
"""
    print(banner)


def load_training_config(config_path: Path | None = None) -> dict:
    """Load training configuration from YAML file."""
    if config_path is None:
        from miniagenticrouter.config import builtin_config_dir
        config_path = builtin_config_dir / "research" / "training.yaml"

    with open(config_path) as f:
        return yaml.safe_load(f)


def load_samples(
    data_dirs: list[str],
    model_names: list[str],
    *,
    skip_invalid_models: bool = True,
    max_samples: int | None = None,
) -> list[dict]:
    """Load trajectory samples from directories.

    Args:
        data_dirs: List of directories containing trajectory files.
        model_names: List of model names for index mapping.

    Returns:
        List of sample dicts with messages, task_id, step_idx, etc.
    """
    from tqdm import tqdm

    # We reuse TrajectoryDataset's parsing logic to ensure sample format matches training:
    # - each sample includes "messages", "task_id", "step_idx", "model_idx", etc.
    dataset = TrajectoryDataset(
        trajectory_dir=[Path(d) for d in data_dirs],
        model_names=model_names,
        lambda_=0.0,
        max_samples=max_samples,
        skip_invalid_models=skip_invalid_models,
        precompute_config=None,
        precomputed_path=None,
    )
    return dataset.samples


def main() -> None:
    """Main entry point."""
    print_banner()

    # Pre-parser to get --config first
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    # Load YAML config to get defaults
    yaml_config = load_training_config(pre_args.config)
    precompute_cfg = yaml_config.get("precompute", {})

    # Main parser with YAML defaults
    parser = argparse.ArgumentParser(
        description="Precompute embeddings for trajectory data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        action="append",
        required=True,
        help="Directory containing trajectory files (can specify multiple times)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for precomputed embeddings (.pt file)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to training config YAML (default: builtin training.yaml)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=precompute_cfg.get("batch_size", 64),
        help="Batch size for encoding",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["hf", "vllm"],
        default=precompute_cfg.get("backend", "hf"),
        help="Embedding backend: 'hf' (HuggingFace) or 'vllm' (vLLM server)",
    )
    parser.add_argument(
        "--vllm-base-url",
        type=str,
        default=precompute_cfg.get("vllm_base_url", "http://localhost:8000"),
        help="vLLM server base URL (only used when --backend=vllm)",
    )
    parser.add_argument(
        "--vllm-timeout",
        type=float,
        default=precompute_cfg.get("vllm_timeout", 60.0),
        help="vLLM request timeout in seconds (only used when --backend=vllm)",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=precompute_cfg.get("cache_path"),
        help="Path to SQLite cache file for embeddings (only used when --backend=vllm)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=precompute_cfg.get("max_workers", 8),
        help="Maximum parallel workers for vLLM requests (only used when --backend=vllm)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to load (debugging)",
    )
    parser.add_argument(
        "--no-skip-invalid-models",
        action="store_false",
        dest="skip_invalid_models",
        default=True,
        help="Do not skip samples with unknown model_idx",
    )
    parser.add_argument(
        "--model-pool",
        type=str,
        nargs="+",
        default=None,
        help="Override model pool from config (e.g., --model-pool openai/gpt-5 deepseek/v3.2)",
    )
    args = parser.parse_args()

    # Get model pool (CLI override or from config)
    if args.model_pool:
        model_names = args.model_pool
    else:
        model_names = get_model_pool(yaml_config)

    # Print configuration
    print(f"  {Colors.YELLOW}Model pool ({len(model_names)} models):{Colors.RESET}")
    for i, name in enumerate(model_names):
        print(f"    [{i}] {name}")
    print(f"  {Colors.YELLOW}Data directories:{Colors.RESET}")
    for i, d in enumerate(args.data_dir):
        print(f"    [{i}] {d}")
    print(f"  {Colors.YELLOW}Output:{Colors.RESET} {args.output}")
    print(f"  {Colors.YELLOW}Backend:{Colors.RESET} {args.backend}")
    print(f"  {Colors.YELLOW}Batch size:{Colors.RESET} {args.batch_size}")
    if args.backend == "vllm":
        print(f"  {Colors.YELLOW}vLLM URL:{Colors.RESET} {args.vllm_base_url}")
        print(f"  {Colors.YELLOW}Max workers:{Colors.RESET} {args.max_workers}")
        if args.cache_path:
            print(f"  {Colors.YELLOW}Cache path:{Colors.RESET} {args.cache_path}")

    # Get history encoder config (yaml_config already loaded by pre-parser)
    he_cfg = yaml_config.get("history_encoder", {})

    # Create precompute config
    precompute_config = PrecomputeConfig(
        enabled=True,
        backend=args.backend,
        model_name=he_cfg.get("model_name", PrecomputeConfig.model_name),
        batch_size=args.batch_size,
        max_tokens=he_cfg.get("max_tokens", 8192),
        min_recent_turns=he_cfg.get("min_recent_turns", 1),
        pooling_mode=he_cfg.get("pooling_mode", "last_token"),
        vllm_base_url=args.vllm_base_url,
        vllm_timeout=args.vllm_timeout,
        vllm_model_id=he_cfg.get("vllm_model_id"),
    )

    print(f"\n  {Colors.YELLOW}Encoder config:{Colors.RESET}")
    print(f"    Model: {precompute_config.model_name.split('/')[-1]}")
    print(f"    Backend: {precompute_config.backend}")
    print(f"    Max tokens: {precompute_config.max_tokens}")
    print(f"    Min recent turns: {precompute_config.min_recent_turns}")
    print(f"    Pooling mode: {precompute_config.pooling_mode}")
    if precompute_config.backend == "vllm":
        print(f"    vLLM request model: {precompute_config.vllm_model_id or precompute_config.model_name.split('/')[-1]}")

    # Load samples
    print(f"\n{Colors.CYAN}Loading trajectory samples...{Colors.RESET}")
    samples = load_samples(
        args.data_dir,
        model_names,
        skip_invalid_models=args.skip_invalid_models,
        max_samples=args.max_samples,
    )

    if len(samples) == 0:
        print(f"{Colors.YELLOW}ERROR: No samples loaded. Check data directories.{Colors.RESET}")
        return

    print(f"\n  Total samples: {len(samples)}")

    # Print benchmark distribution
    from collections import Counter
    benchmark_counts = Counter(s.get("benchmark", "scienceworld") for s in samples)
    print(f"\n  {Colors.YELLOW}Benchmark distribution:{Colors.RESET}")
    for bm, count in sorted(benchmark_counts.items()):
        print(f"    {bm:<20} {count:>8,} ({count/len(samples):.1%})")

    # Initialize precomputer and compute embeddings
    print(f"\n{Colors.CYAN}Initializing encoder ({args.backend} backend)...{Colors.RESET}")
    precomputer = create_precomputer(precompute_config, cache_path=args.cache_path)

    print(f"\n{Colors.CYAN}Computing embeddings...{Colors.RESET}")
    if args.backend == "vllm":
        embeddings = precomputer.precompute(
            samples, show_progress=True, max_workers=args.max_workers
        )
    else:
        embeddings = precomputer.precompute(samples, show_progress=True)

    # Extract sample IDs using episode_id for unique identification
    # episode_id = sha256(source_path)[:16] - computed in parser.to_training_samples()
    sample_ids = [(s["episode_id"], s["step_idx"]) for s in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(
            "Duplicate (episode_id, step_idx) keys detected while saving precomputed embeddings. "
            "This should not happen if episode_id is computed correctly from source_path.\n"
            f"First 5 keys: {sample_ids[:5]}\n"
        )

    # Create sample metadata for debugging and analysis
    sample_metadata = [
        {
            "task_id": s["task_id"],
            "episode_id": s["episode_id"],
            "step_idx": s["step_idx"],
            "source_path": s["source_path"],
        }
        for s in samples
    ]

    # Save to disk
    print(f"\n{Colors.CYAN}Saving to disk...{Colors.RESET}")
    precomputer.save(
        path=args.output,
        embeddings=embeddings,
        sample_ids=sample_ids,
        source_dirs=args.data_dir,
        sample_metadata=sample_metadata,
    )

    # Cleanup
    precomputer.cleanup()

    print(f"\n{Colors.GREEN}{Colors.BOLD}Precomputation complete!{Colors.RESET}")
    print(f"  Output: {args.output}")
    print(f"  Samples: {len(samples)}")
    print(f"  Embedding dim: {precomputer.encoder_dim}")


if __name__ == "__main__":
    main()
