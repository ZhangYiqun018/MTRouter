"""Cluster-based Q-function for learned routing.

Uses KMeans clustering to partition history embedding space into K regions.
Each region maintains per-model statistics for score and cost.

At inference:
    1. Encode history -> z
    2. Find nearest cluster center
    3. Look up Q = score_mean - λ * cost_mean

This approach provides:
- State-dependent routing (different clusters can prefer different models)
- High interpretability (each cluster has clear statistics)
- Piecewise constant policy (Q values are discrete per cluster)
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import torch

from miniagenticrouter.research.routers.learned import QFunction, validate_model_names

if TYPE_CHECKING:
    from sklearn.cluster import KMeans

    from miniagenticrouter.research.training.encoders import HistoryEncoder


class ClusterQFunction(QFunction):
    """Q-function using KMeans clustering for score and cost lookup.

    Each cluster maintains per-model statistics (score_mean, cost_mean).
    At inference, we find the nearest cluster and look up the Q-value.

    Example:
        >>> q_func = ClusterQFunction(
        ...     model_dir="outputs/cluster_baseline",
        ...     lambda_=1.0,
        ... )
        >>> q_values = q_func.predict(messages, ["openai/gpt-5", "minimax/minimax-m2"])
    """

    def __init__(
        self,
        model_dir: Path | str | None = None,
        lambda_: float = 1.0,
        encoder_model: str | None = None,
        device: str = "cuda",
    ):
        """Initialize ClusterQFunction.

        Args:
            model_dir: Directory containing cluster_kmeans.pkl and cluster_stats.json.
            lambda_: Cost penalty coefficient for Q = score - λ * cost.
            encoder_model: HuggingFace model for history encoding.
                If None, reads from training.yaml config file.
            device: Device for encoding ("cuda" or "cpu").
        """
        self.lambda_ = lambda_
        self.encoder_model_name = encoder_model
        self.device = device

        # Cluster model and stats (loaded lazily)
        self.kmeans: KMeans | None = None
        self.cluster_stats: dict[str, dict[str, dict]] | None = None
        self._model_names: list[str] = []
        self._encoder_dim: int = 0
        self._n_clusters: int = 0

        # Expected encoder config (for consistency validation)
        self._expected_encoder_config: dict | None = None

        # Model name to canonical name mapping (for flexible matching)
        self._name_to_canonical: dict[str, str] = {}

        # HistoryEncoder (loaded lazily)
        self._history_encoder: HistoryEncoder | None = None

        # Thread safety
        self._encoder_lock = threading.Lock()

        if model_dir is not None:
            self.load(model_dir)

    def load(
        self,
        path: Path | str,
        expected_model_names: list[str] | None = None,
    ) -> None:
        """Load cluster model and statistics from directory.

        Args:
            path: Directory containing cluster_kmeans.pkl and cluster_stats.json.
            expected_model_names: Expected model names for validation.
                If provided, raises ValueError if checkpoint model names
                don't match (including order).
        """
        path = Path(path)

        # Load KMeans model
        self.kmeans = joblib.load(path / "cluster_kmeans.pkl")
        self._n_clusters = self.kmeans.n_clusters

        # Load cluster statistics
        with open(path / "cluster_stats.json") as f:
            data = json.load(f)

        self.cluster_stats = data["cluster_stats"]
        self._model_names = data["model_names"]
        self._encoder_dim = data["encoder_dim"]
        self._expected_encoder_config = data.get("encoder_config")  # May be None for old files

        # Validate model names if expected
        validate_model_names(
            self._model_names,
            expected_model_names,
            source="checkpoint",
        )

        # Build name mapping for flexible model name matching
        self._name_to_canonical = {}
        for name in self._model_names:
            self._name_to_canonical[name] = name
            self._name_to_canonical[name.lower()] = name
            if "/" in name:
                short_name = name.split("/")[-1]
                self._name_to_canonical[short_name] = name
                self._name_to_canonical[short_name.lower()] = name

        print(f"Loaded cluster model from {path}")
        print(f"  Clusters: {self._n_clusters}, Models: {len(self._model_names)}")

    def save(self, path: Path | str) -> None:
        """Save cluster model and statistics to directory.

        Args:
            path: Output directory.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save KMeans model
        joblib.dump(self.kmeans, path / "cluster_kmeans.pkl")

        # Save statistics
        data = {
            "n_clusters": self._n_clusters,
            "model_names": self._model_names,
            "encoder_dim": self._encoder_dim,
            "cluster_stats": self.cluster_stats,
        }
        with open(path / "cluster_stats.json", "w") as f:
            json.dump(data, f, indent=2)

    def _ensure_encoder(self) -> None:
        """Lazily initialize the HistoryEncoder.

        Thread-safe: uses lock to prevent multiple initializations.

        Encoder configuration resolution order:
        1. Explicit encoder_model parameter from __init__ (model_name only)
        2. training.yaml config file (full history_encoder config)
        3. Fallback to HuggingFace Hub: Qwen/Qwen3-Embedding-0.6B
        """
        if self._history_encoder is not None:
            return

        with self._encoder_lock:
            if self._history_encoder is not None:
                return

            from miniagenticrouter.research.training.encoders import (
                HistoryEncoder,
                HistoryEncoderConfig,
            )

            # Load config from training.yaml
            encoder_config: dict = {}
            try:
                from miniagenticrouter.research.utils.config import (
                    load_training_config,
                )
                training_config = load_training_config()
                encoder_config = training_config.get("history_encoder", {})
            except FileNotFoundError:
                pass

            # Resolve model name (explicit param takes precedence)
            model_name = self.encoder_model_name or encoder_config.get("model_name")
            if model_name is None:
                model_name = "Qwen/Qwen3-Embedding-0.6B"

            # Resolve backend and vLLM settings from config
            backend = encoder_config.get("backend", "hf")
            vllm_base_url = encoder_config.get("vllm_base_url", "http://localhost:8000")
            vllm_model_id = encoder_config.get("vllm_model_id")
            max_tokens = encoder_config.get("max_tokens", 8192)
            pooling_mode = encoder_config.get("pooling_mode", "last_token")
            min_recent_turns = encoder_config.get("min_recent_turns", 1)

            # Validate against expected config from model file (training-time config)
            if self._expected_encoder_config:
                expected = self._expected_encoder_config
                # Critical: model_name determines embedding space
                expected_model = expected.get("model_name")
                if expected_model and expected_model != model_name:
                    print(f"  Warning: model_name mismatch (expected={expected_model}, got={model_name})")
                # Critical: pooling_mode affects embedding extraction
                expected_pooling = expected.get("pooling_mode")
                if expected_pooling and expected_pooling != pooling_mode:
                    print(f"  Warning: pooling_mode mismatch (expected={expected_pooling}, got={pooling_mode})")
                # Backend and max_tokens validation
                if expected.get("backend") != backend:
                    print(f"  Warning: encoder backend mismatch (expected={expected.get('backend')}, got={backend})")
                if expected.get("max_tokens") != max_tokens:
                    print(f"  Warning: max_tokens mismatch (expected={expected.get('max_tokens')}, got={max_tokens})")
                # Use min_recent_turns from training if available
                if expected.get("min_recent_turns") is not None:
                    min_recent_turns = expected.get("min_recent_turns")

            print(f"Loading HistoryEncoder: {model_name} (backend={backend})")
            config = HistoryEncoderConfig(
                model_name=model_name,
                freeze_encoder=True,
                max_tokens=max_tokens,
                min_recent_turns=min_recent_turns,
                pooling_mode=pooling_mode,
                backend=backend,
                vllm_base_url=vllm_base_url,
                vllm_model_id=vllm_model_id,
            )
            encoder = HistoryEncoder(config)
            if backend == "hf":
                encoder.to(self.device)
            encoder.eval()
            self._history_encoder = encoder

    def _encode_history(self, history: str | list[dict]) -> np.ndarray:
        """Encode conversation history to embedding.

        Args:
            history: Either formatted text or list of messages.

        Returns:
            Embedding vector of shape (encoder_dim,).
        """
        self._ensure_encoder()

        with self._encoder_lock:
            with torch.no_grad():
                if isinstance(history, list):
                    embedding = self._history_encoder.forward(messages=history)
                else:
                    embedding = self._history_encoder.forward(text=history)

            return embedding.cpu().numpy()

    def _get_canonical_name(self, model_name: str) -> str | None:
        """Get canonical model name for flexible matching.

        Args:
            model_name: Model name string.

        Returns:
            Canonical model name or None if not found.
        """
        if model_name in self._name_to_canonical:
            return self._name_to_canonical[model_name]

        lower_name = model_name.lower()
        if lower_name in self._name_to_canonical:
            return self._name_to_canonical[lower_name]

        if "/" in model_name:
            short_name = model_name.split("/")[-1]
            if short_name in self._name_to_canonical:
                return self._name_to_canonical[short_name]
            if short_name.lower() in self._name_to_canonical:
                return self._name_to_canonical[short_name.lower()]

        return None

    def predict(
        self,
        history: str | list[dict],
        available_models: list[str],
    ) -> np.ndarray:
        """Predict Q-values for each available model.

        Args:
            history: Conversation history (text or messages).
            available_models: List of model names.

        Returns:
            Q-value array, shape=(len(available_models),).
        """
        if self.kmeans is None or self.cluster_stats is None:
            raise ValueError("Cluster model not loaded. Call load() first.")

        # 1. Encode history
        history_emb = self._encode_history(history)  # (encoder_dim,)

        # 2. Find nearest cluster
        cluster_id = self.kmeans.predict(history_emb.reshape(1, -1))[0]
        cluster_id_str = str(cluster_id)

        # 3. Look up Q-values
        q_values = []
        for model_name in available_models:
            canonical_name = self._get_canonical_name(model_name)
            if canonical_name is None:
                # Unknown model: return neutral Q-value
                q_values.append(0.0)
                continue

            # Get cluster-model statistics (keyed by canonical model name)
            if cluster_id_str in self.cluster_stats and canonical_name in self.cluster_stats[cluster_id_str]:
                stats = self.cluster_stats[cluster_id_str][canonical_name]
                score_mean = stats["score_mean"]
                cost_mean = stats["cost_mean"]
            else:
                # No data for this cluster-model pair
                score_mean = 0.0
                cost_mean = 0.0

            # Q = score - λ * cost
            q = score_mean - self.lambda_ * cost_mean
            q_values.append(q)

        return np.array(q_values)

    def predict_decomposed(
        self,
        history: str | list[dict],
        available_models: list[str],
    ) -> dict[str, Any]:
        """Predict score and cost separately for analysis.

        Args:
            history: Conversation history.
            available_models: List of model names.

        Returns:
            Dict with 'score', 'cost', 'q_values', 'cluster_id' arrays.
        """
        if self.kmeans is None or self.cluster_stats is None:
            raise ValueError("Cluster model not loaded. Call load() first.")

        history_emb = self._encode_history(history)
        cluster_id = self.kmeans.predict(history_emb.reshape(1, -1))[0]
        cluster_id_str = str(cluster_id)

        scores = []
        costs = []
        counts = []
        for model_name in available_models:
            canonical_name = self._get_canonical_name(model_name)
            if canonical_name is None:
                scores.append(0.0)
                costs.append(0.0)
                counts.append(0)
                continue

            if cluster_id_str in self.cluster_stats and canonical_name in self.cluster_stats[cluster_id_str]:
                stats = self.cluster_stats[cluster_id_str][canonical_name]
                scores.append(stats["score_mean"])
                costs.append(stats["cost_mean"])
                counts.append(stats["count"])
            else:
                scores.append(0.0)
                costs.append(0.0)
                counts.append(0)

        scores = np.array(scores)
        costs = np.array(costs)
        q_values = scores - self.lambda_ * costs

        return {
            "score": scores,
            "cost": costs,
            "q_values": q_values,
            "cluster_id": cluster_id,
            "counts": counts,
        }

    def set_lambda(self, lambda_: float) -> None:
        """Update the cost penalty coefficient.

        This allows dynamic adjustment without retraining.

        Args:
            lambda_: New cost penalty coefficient.
        """
        self.lambda_ = lambda_


# =============================================================================
# Batched Inference for Multi-threading
# =============================================================================


@dataclass
class ClusterInferenceRequest:
    """A single inference request for batched processing."""

    messages: list[dict[str, Any]]
    available_models: list[str]
    result_event: threading.Event = field(default_factory=threading.Event)
    result: np.ndarray | None = None
    error: Exception | None = None


class BatchedClusterQFunction:
    """Thread-safe batched cluster Q-function inference.

    Collects inference requests from multiple threads and processes them
    in batches to improve GPU utilization for encoding.

    Example:
        >>> cluster_q = ClusterQFunction(model_dir="outputs/cluster_baseline")
        >>> batched = BatchedClusterQFunction(cluster_q, batch_size=16)
        >>> # From multiple threads:
        >>> q_values = batched.predict(messages, available_models)
    """

    def __init__(
        self,
        cluster_q_func: ClusterQFunction,
        batch_size: int = 16,
        timeout: float = 0.02,
    ):
        """Initialize BatchedClusterQFunction.

        Args:
            cluster_q_func: The underlying ClusterQFunction.
            batch_size: Maximum batch size before processing.
            timeout: Maximum seconds to wait for batch to fill.
        """
        self.cluster_q_func = cluster_q_func
        self.batch_size = batch_size
        self.timeout = timeout

        self._request_queue: queue.Queue[ClusterInferenceRequest] = queue.Queue()
        self._shutdown = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

        # Ensure encoder is loaded
        self.cluster_q_func._ensure_encoder()

    def predict(
        self,
        history: list[dict[str, Any]],
        available_models: list[str],
    ) -> np.ndarray:
        """Predict Q-values (thread-safe, batched).

        This method blocks until the result is available.

        Args:
            history: Conversation history messages.
            available_models: List of model names to score.

        Returns:
            Q-value array, shape=(len(available_models),).

        Raises:
            RuntimeError: If inference failed.
        """
        request = ClusterInferenceRequest(
            messages=history,
            available_models=available_models,
        )

        self._request_queue.put(request)
        request.result_event.wait()

        if request.error is not None:
            raise RuntimeError(f"Batch inference failed: {request.error}")

        return request.result

    def _worker_loop(self) -> None:
        """Background worker that processes batches."""
        while not self._shutdown.is_set():
            batch = self._collect_batch()
            if batch:
                self._process_batch(batch)

    def _collect_batch(self) -> list[ClusterInferenceRequest]:
        """Collect requests until batch is full or timeout."""
        batch: list[ClusterInferenceRequest] = []

        try:
            first = self._request_queue.get(timeout=1.0)
            batch.append(first)
        except queue.Empty:
            return batch

        deadline = time.monotonic() + self.timeout
        while len(batch) < self.batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                req = self._request_queue.get(timeout=remaining)
                batch.append(req)
            except queue.Empty:
                break

        return batch

    def _process_batch(self, batch: list[ClusterInferenceRequest]) -> None:
        """Process a batch of requests with batched encoding.

        Steps:
        1. Batch encode all histories using HistoryEncoder.forward_batch()
        2. Batch find nearest clusters
        3. For each request, look up Q-values
        """
        try:
            with torch.no_grad():
                # Step 1: Batch encode all histories
                all_messages = [req.messages for req in batch]
                embeddings = self.cluster_q_func._history_encoder.forward_batch(all_messages)
                embeddings_np = embeddings.cpu().numpy()  # (N, encoder_dim)

                # Step 2: Batch find nearest clusters
                cluster_ids = self.cluster_q_func.kmeans.predict(embeddings_np)

                # Step 3: For each request, look up Q-values
                for i, req in enumerate(batch):
                    q_values = self._compute_q_for_cluster(
                        cluster_ids[i], req.available_models
                    )
                    req.result = q_values
                    req.result_event.set()

        except Exception as e:
            for req in batch:
                req.error = e
                req.result_event.set()

    def _compute_q_for_cluster(
        self,
        cluster_id: int,
        available_models: list[str],
    ) -> np.ndarray:
        """Compute Q-values for all available models given a cluster ID.

        Args:
            cluster_id: The cluster ID.
            available_models: List of model names.

        Returns:
            Q-value array of shape (len(available_models),).
        """
        cluster_id_str = str(cluster_id)
        q_values = []

        for model_name in available_models:
            canonical_name = self.cluster_q_func._get_canonical_name(model_name)
            if canonical_name is None:
                q_values.append(0.0)
                continue

            if (cluster_id_str in self.cluster_q_func.cluster_stats and
                    canonical_name in self.cluster_q_func.cluster_stats[cluster_id_str]):
                stats = self.cluster_q_func.cluster_stats[cluster_id_str][canonical_name]
                score_mean = stats["score_mean"]
                cost_mean = stats["cost_mean"]
            else:
                score_mean = 0.0
                cost_mean = 0.0

            q = score_mean - self.cluster_q_func.lambda_ * cost_mean
            q_values.append(q)

        return np.array(q_values)

    def shutdown(self) -> None:
        """Shutdown the worker thread."""
        self._shutdown.set()
        self._worker_thread.join(timeout=2.0)

    def set_lambda(self, lambda_: float) -> None:
        """Update the cost penalty coefficient."""
        self.cluster_q_func.set_lambda(lambda_)

    # QFunction interface compatibility
    def load(
        self,
        path: Path | str,
        expected_model_names: list[str] | None = None,
    ) -> None:
        """Load weights (delegates to underlying cluster_q_func)."""
        self.cluster_q_func.load(path, expected_model_names=expected_model_names)

    def save(self, path) -> None:
        """Save weights (delegates to underlying cluster_q_func)."""
        self.cluster_q_func.save(path)
