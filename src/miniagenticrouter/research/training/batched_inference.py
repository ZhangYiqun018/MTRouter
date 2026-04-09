"""Batched inference utilities for NeuralQFunction.

This module provides dynamic batching support for Q-function inference,
improving GPU utilization when multiple requests arrive concurrently.

Example:
    >>> from miniagenticrouter.research.training.batched_inference import BatchedQFunction
    >>> q_func = NeuralQFunction(model_names=["haiku", "sonnet", "opus"])
    >>> batched = BatchedQFunction(q_func, batch_size=8, timeout=0.02)
    >>>
    >>> # Use in LearnedRouter (thread-safe)
    >>> router = LearnedRouter(model_kwargs=..., q_function=batched)
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from miniagenticrouter.research.training.q_network import NeuralQFunction


@dataclass
class InferenceRequest:
    """A single inference request."""

    messages: list[dict[str, Any]]
    available_models: list[str]
    result_event: threading.Event = field(default_factory=threading.Event)
    result: np.ndarray | None = None
    error: Exception | None = None


class BatchedQFunction:
    """Thread-safe batched Q-function inference.

    Collects inference requests from multiple threads and processes them
    in batches to improve GPU utilization.

    This class implements the QFunction interface, so it can be used as
    a drop-in replacement for NeuralQFunction in LearnedRouter.

    Attributes:
        q_func: The underlying NeuralQFunction.
        batch_size: Maximum batch size.
        timeout: Maximum time to wait for batch to fill (seconds).

    Example:
        >>> batched = BatchedQFunction(q_func, batch_size=8, timeout=0.02)
        >>> # From multiple threads:
        >>> q_values = batched.predict(messages, available_models)
    """

    def __init__(
        self,
        q_func: NeuralQFunction,
        batch_size: int = 8,
        timeout: float = 0.02,
    ):
        """Initialize BatchedQFunction.

        Args:
            q_func: The underlying NeuralQFunction (should be in eval mode).
            batch_size: Maximum batch size before processing.
            timeout: Maximum seconds to wait for batch to fill.
        """
        self.q_func = q_func
        self.batch_size = batch_size
        self.timeout = timeout

        self._request_queue: queue.Queue[InferenceRequest] = queue.Queue()
        self._shutdown = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

        # Ensure model is in eval mode
        self.q_func.eval()

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
        request = InferenceRequest(
            messages=history,
            available_models=available_models,
        )

        self._request_queue.put(request)
        request.result_event.wait()  # Block until processed

        if request.error is not None:
            raise RuntimeError(f"Batch inference failed: {request.error}")

        return request.result

    def _worker_loop(self) -> None:
        """Background worker that processes batches."""
        while not self._shutdown.is_set():
            batch = self._collect_batch()
            if batch:
                self._process_batch(batch)

    def _collect_batch(self) -> list[InferenceRequest]:
        """Collect requests until batch is full or timeout."""
        batch: list[InferenceRequest] = []

        # Wait for first request (with long timeout)
        try:
            first = self._request_queue.get(timeout=1.0)
            batch.append(first)
        except queue.Empty:
            return batch

        # Collect more requests until batch full or timeout
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

    def _process_batch(self, batch: list[InferenceRequest]) -> None:
        """Process a batch of requests with fully vectorized computation.

        Assumes all requests have the same available_models (validated at runtime).
        This enables sharing model embeddings across all requests in the batch.
        """
        try:
            with torch.no_grad():
                # Step 1: Batch encode all histories → z_x_batch: (N, state_dim)
                all_messages = [req.messages for req in batch]
                z_x_batch = self.q_func.history_encoder.forward_batch(all_messages)

                # Ensure z_x_batch is on the same device as model (vLLM returns CPU tensors)
                device = self.q_func._get_device()
                z_x_batch = z_x_batch.to(device)

                # Step 2: Encode models once (shared across all requests)
                # All requests should have the same available_models
                available_models = batch[0].available_models

                # Collect valid model indices
                indices = []
                valid_positions = []
                for pos, model_name in enumerate(available_models):
                    idx = self.q_func.name_to_idx.get(model_name, -1)
                    if idx >= 0:
                        indices.append(idx)
                        valid_positions.append(pos)

                N = len(batch)
                M = len(available_models)

                # Handle case where all models are invalid
                if not indices:
                    q_numpy = np.full((N, M), float("-inf"))
                    for i, req in enumerate(batch):
                        req.result = q_numpy[i]
                        req.result_event.set()
                    return

                # z_m_valid: (num_valid, model_dim)
                z_m_valid = self.q_func.model_encoder.forward_batch(indices)
                num_valid = len(indices)

                # Step 3: Fully vectorized Q-value computation
                # Expand: z_x_batch (N, state_dim) → (N, num_valid, state_dim)
                #         z_m_valid (num_valid, model_dim) → (N, num_valid, model_dim)
                z_x_expanded = z_x_batch.unsqueeze(1).expand(N, num_valid, -1)
                z_m_expanded = z_m_valid.unsqueeze(0).expand(N, num_valid, -1)

                # Flatten for single QNetwork forward pass
                z_x_flat = z_x_expanded.reshape(N * num_valid, -1)
                z_m_flat = z_m_expanded.reshape(N * num_valid, -1)

                # Single forward pass: single-head returns Q-values directly
                q_flat = self.q_func.q_network(z_x_flat, z_m_flat)
                q_matrix = q_flat.reshape(N, num_valid)  # (N, num_valid)

                # Step 4: Fill results with -inf for invalid models
                q_full = np.full((N, M), float("-inf"))
                q_valid_numpy = q_matrix.cpu().numpy()
                for j, pos in enumerate(valid_positions):
                    q_full[:, pos] = q_valid_numpy[:, j]

                # Distribute results to requests
                for i, req in enumerate(batch):
                    req.result = q_full[i]
                    req.result_event.set()

        except Exception as e:
            # If batch-level error, propagate to all requests
            for req in batch:
                req.error = e
                req.result_event.set()

    def _compute_q_values_single(
        self,
        z_x: torch.Tensor,
        available_models: list[str],
    ) -> np.ndarray:
        """Compute Q-values for a single state embedding.

        Uses vectorized computation over all available models.
        """
        # Get valid model indices
        indices = []
        valid_positions = []
        for pos, model_name in enumerate(available_models):
            idx = self.q_func.name_to_idx.get(model_name, -1)
            if idx >= 0:
                indices.append(idx)
                valid_positions.append(pos)

        if not indices:
            return np.full(len(available_models), float("-inf"))

        # Batch encode models
        z_m = self.q_func.model_encoder.forward_batch(indices)

        # Expand z_x for broadcasting
        z_x_expanded = z_x.unsqueeze(0).expand(len(indices), -1)

        # Compute Q-values with single-head
        q_values_valid = self.q_func.q_network(z_x_expanded, z_m).squeeze(-1)

        # Fill result array
        q_values = np.full(len(available_models), float("-inf"))
        q_values[valid_positions] = q_values_valid.cpu().numpy()

        return q_values

    def shutdown(self) -> None:
        """Shutdown the worker thread."""
        self._shutdown.set()
        self._worker_thread.join(timeout=2.0)

    def set_lambda(self, lambda_: float) -> None:
        """Set the cost penalty coefficient.

        Delegates to underlying q_func.

        Args:
            lambda_: New cost penalty coefficient.
        """
        self.q_func.set_lambda(lambda_)

    # QFunction interface compatibility
    def load(self, path) -> None:
        """Load weights (delegates to underlying q_func)."""
        self.q_func.load(path)

    def save(self, path) -> None:
        """Save weights (delegates to underlying q_func)."""
        self.q_func.save(path)
