"""vLLM HTTP client for embedding requests.

This module provides a client for making embedding requests to a vLLM server
using the OpenAI-compatible /v1/embeddings API.

Example:
    >>> config = VLLMClientConfig(base_url="http://localhost:8000")
    >>> client = VLLMClient(config)
    >>> embeddings = client.embed(["Hello world", "Test text"])
    >>> print(len(embeddings[0]))  # Embedding dimension
    1024
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch
from openai import OpenAI


class EmbeddingCache:
    """SQLite-based embedding cache.

    Caches text embeddings to avoid redundant vLLM requests.
    Uses (SHA256 hash of text, model_name) as composite key.

    Example:
        >>> cache = EmbeddingCache("embeddings.db", model_name="Qwen3-Embedding-0.6B")
        >>> cache.put("hello world", [0.1, 0.2, 0.3])
        >>> cache.get("hello world")
        [0.1, 0.2, 0.3]
    """

    def __init__(self, cache_path: Path | str, model_name: str):
        """Initialize cache with SQLite database.

        Args:
            cache_path: Path to SQLite database file.
            model_name: Model name to distinguish embeddings from different models.
        """
        self.cache_path = Path(cache_path)
        self.model_name = model_name
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(str(self.cache_path), check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                text_hash TEXT NOT NULL,
                model_name TEXT NOT NULL,
                embedding BLOB,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (text_hash, model_name)
            )
        """)
        self.conn.commit()

    @staticmethod
    def hash_text(text: str) -> str:
        """Compute SHA256 hash of text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> list[float] | None:
        """Get cached embedding for text.

        Args:
            text: Input text.

        Returns:
            Cached embedding or None if not found.
        """
        text_hash = self.hash_text(text)
        with self._lock:
            cursor = self.conn.execute(
                "SELECT embedding FROM embeddings WHERE text_hash = ? AND model_name = ?",
                (text_hash, self.model_name),
            )
            row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None

    def put(self, text: str, embedding: list[float]) -> None:
        """Store embedding in cache.

        Args:
            text: Input text.
            embedding: Embedding vector.
        """
        text_hash = self.hash_text(text)
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO embeddings (text_hash, model_name, embedding) VALUES (?, ?, ?)",
                (text_hash, self.model_name, json.dumps(embedding)),
            )
            self.conn.commit()

    def get_batch(self, texts: list[str]) -> dict[int, list[float]]:
        """Get cached embeddings for multiple texts.

        Args:
            texts: List of input texts.

        Returns:
            Dict mapping index to embedding for cache hits only.
        """
        result = {}
        for i, text in enumerate(texts):
            emb = self.get(text)
            if emb is not None:
                result[i] = emb
        return result

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()

    def stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dict with total_entries, model_name, and cache_path.
        """
        with self._lock:
            cursor = self.conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE model_name = ?",
                (self.model_name,),
            )
            count = cursor.fetchone()[0]
        return {
            "total_entries": count,
            "model_name": self.model_name,
            "cache_path": str(self.cache_path),
        }


@dataclass
class VLLMClientConfig:
    """Configuration for vLLM HTTP client.

    Attributes:
        base_url: Base URL of the vLLM server (e.g., "http://localhost:8000" or "http://localhost:8000/v1").
        model_name: HuggingFace model name or local path (legacy; used as fallback).
        request_model: Model ID to send in API requests (recommended).
        api_key: API key for OpenAI-compatible server (often unused; "dummy" is typical).
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retry attempts on failure.
        batch_size: Number of texts per batch request.
    """

    base_url: str = "http://localhost:8000"
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    request_model: str | None = None
    api_key: str | None = None
    timeout: float = 60.0
    max_retries: int = 3
    batch_size: int = 64


class VLLMClient:
    """HTTP client for vLLM embedding API.

    Uses the OpenAI-compatible /v1/embeddings endpoint to get embeddings
    from a vLLM server.

    Example:
        >>> config = VLLMClientConfig(base_url="http://localhost:8000")
        >>> client = VLLMClient(config)
        >>>
        >>> # Single request
        >>> embeddings = client.embed(["text1", "text2"])
        >>>
        >>> # Batch request with progress bar
        >>> all_embeddings = client.embed_batch(long_text_list, show_progress=True)
    """

    def __init__(self, config: VLLMClientConfig):
        """Initialize VLLMClient.

        Args:
            config: Client configuration.
        """
        self.config = config
        api_key = self.config.api_key
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_TOKEN") or "dummy"

        # OpenAI SDK expects a base URL that includes the /v1 prefix for compatibility.
        # Accept both "http://host:port" and "http://host:port/v1".
        base_url = self.config.base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
        )
        self._encoder_dim: int | None = None

    @property
    def encoder_dim(self) -> int:
        """Get embedding dimension (auto-detected on first request).

        Returns:
            Embedding dimension from the model.
        """
        if self._encoder_dim is None:
            # Make a test request to detect dimension
            test_emb = self.embed(["test"])[0]
            self._encoder_dim = len(test_emb)
        return self._encoder_dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embeddings, each embedding is a list of floats.

        Note:
            This uses the OpenAI Python SDK against a vLLM OpenAI-compatible server.
        """
        model = self.config.request_model or self.config.model_name
        response = self.client.embeddings.create(
            model=model,
            input=texts,
        )
        # Sort by index to ensure correct order
        data_sorted = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in data_sorted]

    def embed_batch(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Batch embed multiple texts with progress tracking.

        Args:
            texts: List of text strings to embed.
            show_progress: Whether to show progress bar.

        Returns:
            Embeddings tensor of shape (N, encoder_dim).
        """
        from tqdm import tqdm

        embeddings_list = []
        n_batches = (len(texts) + self.config.batch_size - 1) // self.config.batch_size

        pbar = tqdm(range(n_batches), desc="Encoding (vLLM)", disable=not show_progress)
        for i in pbar:
            start = i * self.config.batch_size
            end = min(start + self.config.batch_size, len(texts))
            batch_texts = texts[start:end]

            # Handle empty strings separately
            non_empty_indices = [j for j, t in enumerate(batch_texts) if t]
            non_empty_texts = [t for t in batch_texts if t]

            if non_empty_texts:
                embs = self.embed(non_empty_texts)
                non_empty_embs = torch.tensor(embs, dtype=torch.float32)
            else:
                non_empty_embs = torch.zeros(0, self.encoder_dim)

            # Fill in zeros for empty strings
            batch_embs = torch.zeros(len(batch_texts), self.encoder_dim)
            for j, orig_idx in enumerate(non_empty_indices):
                batch_embs[orig_idx] = non_empty_embs[j]

            embeddings_list.append(batch_embs)

        return torch.cat(embeddings_list, dim=0)

    def embed_single(self, text: str) -> list[float]:
        """Get embedding for a single text.

        Args:
            text: Input text string.

        Returns:
            Embedding as list of floats, or zeros if text is empty.
        """
        if not text:
            return [0.0] * self.encoder_dim
        return self.embed([text])[0]

    def embed_batch_parallel(
        self,
        texts: list[str],
        max_workers: int = 8,
        show_progress: bool = True,
        cache: EmbeddingCache | None = None,
    ) -> torch.Tensor:
        """Parallel batch embedding with single-request-per-thread.

        Sends one text per vLLM request using multiple threads.
        Uses futures list to maintain ordering.

        Args:
            texts: List of text strings to embed.
            max_workers: Maximum parallel threads.
            show_progress: Whether to show progress bar.
            cache: Optional EmbeddingCache for caching results.

        Returns:
            Embeddings tensor of shape (N, encoder_dim).
        """
        from tqdm import tqdm

        n = len(texts)
        results: list[list[float] | None] = [None] * n

        # Step 1: Check cache for hits
        uncached_indices: list[int] = []
        if cache is not None:
            cached = cache.get_batch(texts)
            for i, emb in cached.items():
                results[i] = emb
            uncached_indices = [i for i in range(n) if results[i] is None]
        else:
            uncached_indices = list(range(n))

        cache_hits = n - len(uncached_indices)
        if show_progress and cache_hits > 0:
            print(f"Cache hits: {cache_hits}/{n} ({cache_hits / n * 100:.1f}%)")

        # Step 2: Parallel requests for uncached texts using futures
        if uncached_indices:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks, futures list maintains order with uncached_indices
                futures: list[Future[list[float]]] = [
                    executor.submit(self.embed_single, texts[i]) for i in uncached_indices
                ]

                # Collect results in order
                pbar = tqdm(
                    total=len(futures),
                    desc="Encoding (vLLM parallel)",
                    disable=not show_progress,
                )
                for idx, future in zip(uncached_indices, futures):
                    emb = future.result()  # Block until this future completes
                    results[idx] = emb

                    # Write to cache
                    if cache is not None:
                        cache.put(texts[idx], emb)

                    pbar.update(1)

                pbar.close()

        # Step 3: Convert to tensor
        embeddings = torch.tensor(results, dtype=torch.float32)
        return embeddings

    def health_check(self) -> bool:
        """Check if the vLLM server is available.

        Returns:
            True if server is responding, False otherwise.
        """
        try:
            self.client.models.list()
            return True
        except Exception:
            return False

    def get_available_models(self) -> list[str]:
        """Get list of available models on the server.

        Returns:
            List of model IDs available on the server.

        Raises:
            Exception: If the request fails.
        """
        models = self.client.models.list()
        return [m.id for m in models.data]
