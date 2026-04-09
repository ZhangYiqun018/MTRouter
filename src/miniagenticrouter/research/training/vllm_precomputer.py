"""DEPRECATED: vLLM-based embedding precomputer.

This module is deprecated in favor of HFPrecomputer which ensures
training/inference consistency by using the same HuggingFace backend.

The vLLM approach had subtle numerical differences compared to HuggingFace,
causing training/inference mismatch issues.

For new code, use HFPrecomputer from encoders.py instead.

Example migration:
    # Old (deprecated):
    from miniagenticrouter.research.training.vllm_precomputer import VLLMPrecomputer
    precomputer = VLLMPrecomputer(config)

    # New (recommended):
    from miniagenticrouter.research.training.encoders import HFPrecomputer
    precomputer = HFPrecomputer(config)
"""

from __future__ import annotations

import gc
import warnings
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoTokenizer

# Emit deprecation warning on import
warnings.warn(
    "vllm_precomputer is deprecated. Use HFPrecomputer from encoders.py instead. "
    "The vLLM backend causes training/inference inconsistencies.",
    DeprecationWarning,
    stacklevel=2,
)


@dataclass
class VLLMPrecomputeConfig:
    """DEPRECATED: Configuration for vLLM embedding precomputation.

    Use HistoryEncoderConfig from encoders.py instead.
    """

    enabled: bool = False
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    batch_size: int = 256
    gpu_memory_utilization: float = 0.8
    max_tokens: int = 512
    min_recent_turns: int = 1


# Import standalone segmentation functions from encoders
from miniagenticrouter.research.training.encoders import segment_history_standalone


class VLLMPrecomputer:
    """DEPRECATED: Precompute embeddings using vLLM.

    This class is deprecated because vLLM produces embeddings with subtle
    numerical differences compared to HuggingFace, causing training/inference
    mismatch issues.

    Use HFPrecomputer from encoders.py instead for consistent embeddings.

    Example:
        >>> # DEPRECATED - do not use
        >>> precomputer = VLLMPrecomputer(config)
        >>> embeddings = precomputer.precompute(samples)
        >>> precomputer.cleanup()
    """

    def __init__(self, config: VLLMPrecomputeConfig):
        """Initialize VLLMPrecomputer.

        Args:
            config: Precomputation configuration.
        """
        warnings.warn(
            "VLLMPrecomputer is deprecated. Use HFPrecomputer instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        from vllm import LLM

        self.config = config

        # Load tokenizer for segmentation (lightweight, CPU only)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
        )

        # Load vLLM for embedding
        print(f"Loading vLLM embedding model: {config.model_name}")
        print(f"GPU memory utilization: {config.gpu_memory_utilization:.0%}")
        self.llm = LLM(
            model=config.model_name,
            task="embed",
            gpu_memory_utilization=config.gpu_memory_utilization,
            trust_remote_code=True,
            dtype="auto",
        )

        # Detect embedding dimension by running a test embed
        test_output = self.llm.embed(["test"])
        self.encoder_dim = len(test_output[0].outputs.embedding)
        print(f"Encoder dimension: {self.encoder_dim}")

    def precompute(
        self,
        samples: list[dict[str, Any]],
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Precompute embeddings for all samples.

        Args:
            samples: List of sample dicts with 'messages' key.
            show_progress: Whether to show progress bars.

        Returns:
            Unified embeddings, shape (N, encoder_dim).
        """
        from tqdm import tqdm

        n_samples = len(samples)
        print(f"Precomputing embeddings for {n_samples} samples...")

        # Step 1: Segment all messages into a single formatted string (CPU, fast)
        texts = []

        pbar = tqdm(samples, desc="Segmenting messages", disable=not show_progress)
        for sample in pbar:
            text = segment_history_standalone(
                messages=sample["messages"],
                tokenizer=self.tokenizer,
                max_tokens=self.config.max_tokens,
                min_recent_turns=self.config.min_recent_turns,
            )
            texts.append(text)

        # Step 2: Batch encode with vLLM (single pass)
        embeddings = self._encode_batch(texts, desc="Encoding histories", show_progress=show_progress)

        print(f"Precomputation complete. Shape: ({n_samples}, {self.encoder_dim})")
        return embeddings

    def _encode_batch(
        self,
        texts: list[str],
        desc: str = "Encoding",
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Encode texts in batches using vLLM.

        Args:
            texts: List of text strings.
            desc: Progress bar description.
            show_progress: Whether to show progress bar.

        Returns:
            Embeddings tensor of shape (N, encoder_dim).
        """
        from tqdm import tqdm

        embeddings_list = []
        n_batches = (len(texts) + self.config.batch_size - 1) // self.config.batch_size

        pbar = tqdm(range(n_batches), desc=desc, disable=not show_progress)
        for i in pbar:
            start = i * self.config.batch_size
            end = min(start + self.config.batch_size, len(texts))
            batch_texts = texts[start:end]

            # Handle empty strings
            non_empty_indices = [j for j, t in enumerate(batch_texts) if t]
            non_empty_texts = [t for t in batch_texts if t]

            if non_empty_texts:
                outputs = self.llm.embed(non_empty_texts)
                non_empty_embs = torch.tensor(
                    [o.outputs.embedding for o in outputs],
                    dtype=torch.float32,
                )
            else:
                non_empty_embs = torch.zeros(0, self.encoder_dim)

            # Fill in zeros for empty strings
            batch_embs = torch.zeros(len(batch_texts), self.encoder_dim)
            for j, orig_idx in enumerate(non_empty_indices):
                batch_embs[orig_idx] = non_empty_embs[j]

            embeddings_list.append(batch_embs)

        return torch.cat(embeddings_list, dim=0)

    def cleanup(self) -> None:
        """Release GPU memory by deleting vLLM and clearing caches."""
        print("Cleaning up vLLM and freeing GPU memory...")
        del self.llm
        del self.tokenizer

        torch.cuda.empty_cache()
        gc.collect()
        print("Cleanup complete.")
