#!/usr/bin/env python
"""Test Q-function inference with trained checkpoint.

Usage:
    python scripts/test_inference.py --checkpoint outputs/q_function/q_function_best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from miniagenticrouter.research.training.q_network import NeuralQFunction
from miniagenticrouter.research.utils.config import get_model_pool, load_training_config

# Test cases with different complexity levels
TEST_CASES = [
    {
        "name": "Simple greeting",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
    },
    {
        "name": "Math problem",
        "messages": [
            {"role": "user", "content": "What is the derivative of x^3 + 2x^2 - 5x + 3?"}
        ],
    },
    {
        "name": "Code generation",
        "messages": [
            {
                "role": "user",
                "content": "Write a Python function to find the longest common subsequence of two strings.",
            }
        ],
    },
    {
        "name": "Complex reasoning",
        "messages": [
            {
                "role": "user",
                "content": "Explain the difference between supervised and unsupervised learning, and provide examples of when you would use each.",
            }
        ],
    },
    {
        "name": "Multi-turn conversation",
        "messages": [
            {"role": "user", "content": "I need help with a Python project."},
            {
                "role": "assistant",
                "content": "I'd be happy to help! What kind of project are you working on?",
            },
            {
                "role": "user",
                "content": "It's a web scraper. I'm having trouble with async requests.",
            },
        ],
    },
]


def main():
    parser = argparse.ArgumentParser(description="Test Q-function inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/q_function/q_function_best.pt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--model-pool",
        type=str,
        nargs="+",
        default=None,
        help="Override model pool (e.g., --model-pool openai/gpt-5 deepseek/v3.2)",
    )
    args = parser.parse_args()

    # Load model pool from config or CLI override
    if args.model_pool:
        model_names = args.model_pool
    else:
        yaml_config = load_training_config()
        model_names = get_model_pool(yaml_config)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Device: {args.device}")
    print(f"Models: {model_names}\n")

    # Initialize and load
    q_func = NeuralQFunction(model_names=model_names)
    q_func.load(checkpoint_path)
    q_func.to(args.device)
    q_func.eval()

    print("=" * 70)
    print("Q-Function Inference Test")
    print("=" * 70)

    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n[Test {i}] {test_case['name']}")
        print("-" * 50)

        # Show truncated message
        last_msg = test_case["messages"][-1]["content"]
        display_msg = last_msg[:80] + "..." if len(last_msg) > 80 else last_msg
        print(f"Query: {display_msg}")

        # Get Q-values
        q_values = q_func.predict(
            history=test_case["messages"],
            available_models=model_names,
        )

        # Display results
        print("\nQ-values:")
        sorted_indices = q_values.argsort()[::-1]
        for rank, idx in enumerate(sorted_indices, 1):
            model_name = model_names[idx]
            q_val = q_values[idx]
            marker = ">>> " if rank == 1 else "    "
            print(f"  {marker}[{rank}] {model_name:<25} Q = {q_val:>8.2f}")

        selected = model_names[sorted_indices[0]]
        print(f"\nSelected: {selected}")

    print("\n" + "=" * 70)
    print("Inference test completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
