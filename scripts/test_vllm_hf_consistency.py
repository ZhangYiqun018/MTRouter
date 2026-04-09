#!/usr/bin/env python
"""Test embedding consistency between vLLM and HuggingFace.

This script compares embeddings produced by vLLM's embed() and HuggingFace's
forward() for the same input texts, to verify they are aligned for
training (vLLM precompute) and inference (HF online).

Usage:
    python scripts/test_vllm_hf_consistency.py
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def get_hf_embedding(
    text: str,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    pooling_mode: str = "last_token",
) -> torch.Tensor:
    """Get embedding using HuggingFace (same as HistoryEncoder)."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state  # (1, seq_len, hidden)

        if pooling_mode == "last_token":
            # Get last non-padding token
            seq_len = inputs["attention_mask"].sum() - 1
            emb = last_hidden[0, seq_len, :]
        else:  # cls
            emb = last_hidden[0, 0, :]

    return emb.cpu()


def get_vllm_embedding(text: str, llm) -> torch.Tensor:
    """Get embedding using vLLM."""
    outputs = llm.embed([text])
    emb = outputs[0].outputs.embedding
    return torch.tensor(emb)


def compare_embeddings(
    hf_emb: torch.Tensor,
    vllm_emb: torch.Tensor,
    name: str = "test",
) -> dict:
    """Compare two embeddings and return metrics."""
    # Ensure same dtype
    hf_emb = hf_emb.float()
    vllm_emb = vllm_emb.float()

    # Cosine similarity
    cos_sim = F.cosine_similarity(hf_emb.unsqueeze(0), vllm_emb.unsqueeze(0)).item()

    # L2 distance
    l2_dist = torch.norm(hf_emb - vllm_emb).item()

    # Relative error
    rel_error = l2_dist / (torch.norm(hf_emb).item() + 1e-8)

    # Max absolute difference
    max_diff = torch.max(torch.abs(hf_emb - vllm_emb)).item()

    return {
        "name": name,
        "cosine_similarity": cos_sim,
        "l2_distance": l2_dist,
        "relative_error": rel_error,
        "max_abs_diff": max_diff,
        "hf_norm": torch.norm(hf_emb).item(),
        "vllm_norm": torch.norm(vllm_emb).item(),
    }


def main():
    # Model path (same as training)
    model_name = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"

    print("=" * 70)
    print("vLLM vs HuggingFace Embedding Consistency Test")
    print("=" * 70)
    print(f"\nModel: {model_name}")

    # Test cases with varying lengths
    test_cases = [
        ("short", "Hello world."),
        ("medium", "The quick brown fox jumps over the lazy dog. " * 5),
        ("long", "This is a longer text to test embedding consistency. " * 20),
        ("code", "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"),
        ("chinese", "这是一个测试中文文本的例子，用于验证embedding的一致性。"),
        ("mixed", "Task: Write a function to calculate factorial.\nUser: def fact(n): return 1 if n==0 else n*fact(n-1)"),
    ]

    # =========================================================================
    # Load HuggingFace model
    # =========================================================================
    print("\n" + "-" * 70)
    print("Loading HuggingFace model...")
    print("-" * 70)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    hf_model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).cuda().eval()

    print(f"HF Model loaded. Device: {hf_model.device}")

    # Get HF embeddings first
    print("\nComputing HF embeddings...")
    hf_embeddings = {}
    for name, text in test_cases:
        hf_emb = get_hf_embedding(text, hf_model, tokenizer, pooling_mode="last_token")
        hf_embeddings[name] = hf_emb
        print(f"  {name}: shape={hf_emb.shape}, norm={torch.norm(hf_emb):.4f}")

    # Free HF model memory
    del hf_model
    torch.cuda.empty_cache()
    gc.collect()
    print("\nHF model unloaded to free GPU memory.")

    # =========================================================================
    # Load vLLM model
    # =========================================================================
    print("\n" + "-" * 70)
    print("Loading vLLM model...")
    print("-" * 70)

    from vllm import LLM

    vllm_model = LLM(
        model=model_name,
        task="embed",
        gpu_memory_utilization=0.5,
        trust_remote_code=True,
        dtype="auto",
    )

    # Get vLLM embeddings
    print("\nComputing vLLM embeddings...")
    vllm_embeddings = {}
    for name, text in test_cases:
        vllm_emb = get_vllm_embedding(text, vllm_model)
        vllm_embeddings[name] = vllm_emb
        print(f"  {name}: shape={vllm_emb.shape}, norm={torch.norm(vllm_emb):.4f}")

    # Free vLLM model
    del vllm_model
    torch.cuda.empty_cache()
    gc.collect()
    print("\nvLLM model unloaded.")

    # =========================================================================
    # Compare embeddings
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    results = []
    for name, _ in test_cases:
        result = compare_embeddings(
            hf_embeddings[name],
            vllm_embeddings[name],
            name=name,
        )
        results.append(result)

    # Print table
    print(f"\n{'Case':<12} {'Cos Sim':>10} {'L2 Dist':>10} {'Rel Err':>10} {'Max Diff':>10}")
    print("-" * 54)

    for r in results:
        cos_color = "\033[92m" if r["cosine_similarity"] > 0.99 else "\033[93m" if r["cosine_similarity"] > 0.95 else "\033[91m"
        reset = "\033[0m"
        print(
            f"{r['name']:<12} "
            f"{cos_color}{r['cosine_similarity']:>10.6f}{reset} "
            f"{r['l2_distance']:>10.4f} "
            f"{r['relative_error']:>10.6f} "
            f"{r['max_abs_diff']:>10.6f}"
        )

    # Summary
    avg_cos_sim = np.mean([r["cosine_similarity"] for r in results])
    min_cos_sim = min(r["cosine_similarity"] for r in results)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Average Cosine Similarity: {avg_cos_sim:.6f}")
    print(f"Minimum Cosine Similarity: {min_cos_sim:.6f}")

    if min_cos_sim > 0.99:
        print("\n\033[92m✓ PASS: vLLM and HuggingFace embeddings are highly consistent.\033[0m")
        print("  Training with vLLM precompute and inference with HF should be aligned.")
    elif min_cos_sim > 0.95:
        print("\n\033[93m⚠ WARNING: Some difference detected, but likely acceptable.\033[0m")
        print("  Consider monitoring model performance closely.")
    else:
        print("\n\033[91m✗ FAIL: Significant difference between vLLM and HuggingFace embeddings!\033[0m")
        print("  This may cause training/inference misalignment.")
        print("  Recommend using HF for both training and inference.")

    print()


if __name__ == "__main__":
    main()
