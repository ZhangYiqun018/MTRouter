"""State and model encoders for Q-function.

This module provides encoders for converting conversation history
into fixed-dimensional vectors for learned router training.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Backend type for embedding computation
BackendType = Literal["hf", "vllm"]


def last_token_pool(
    last_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract embeddings using last token pooling (for decoder-only models).

    Official implementation from Qwen3-Embedding.
    Reference: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B

    Args:
        last_hidden_states: Shape (batch, seq_len, hidden_dim).
        attention_mask: Shape (batch, seq_len).

    Returns:
        Embeddings of shape (batch, hidden_dim).
    """
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        # Left padding: last token is always the EOS
        return last_hidden_states[:, -1]
    else:
        # Right padding: find actual last token position
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]


@dataclass
class HistoryEncoderConfig:
    """Configuration for HistoryEncoder.

    Attributes:
        model_name: HuggingFace model name or local path for text encoding.
        freeze_encoder: Whether to freeze pretrained encoder parameters.
        max_tokens: Maximum total tokens for encoding (task + context).
        min_recent_turns: Minimum number of recent turns to always keep.
        use_projection: Deprecated (kept for checkpoint compatibility).
        pooling_mode: Pooling strategy for extracting embeddings.
        backend: Embedding backend type ("hf" for HuggingFace, "vllm" for vLLM server).
        vllm_base_url: Base URL of vLLM server (only used when backend="vllm").

    Token allocation priority:
        1. Task description (as many tokens as needed, up to max_tokens)
        2. Minimum recent turns (guaranteed)
        3. Additional context (fill remaining budget)
    """

    # Model settings
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    freeze_encoder: bool = True

    # Token budget (unified)
    max_tokens: int = 512
    min_recent_turns: int = 1

    # Deprecated: kept for backward compatibility with older configs/checkpoints
    use_projection: bool = True

    # Pooling mode
    pooling_mode: Literal["cls", "last_token"] = "last_token"
    """Pooling strategy for extracting embeddings:
    - "cls": Use first token [CLS] (for encoder models like BERT)
    - "last_token": Use last token [EOS] (for decoder-only models like Qwen3-Embedding)
    """

    # Backend configuration
    backend: BackendType = "hf"
    """Embedding backend type:
    - "hf": Use HuggingFace AutoModel locally
    - "vllm": Use vLLM server via HTTP API
    """

    vllm_base_url: str = "http://localhost:8000"
    """Base URL of vLLM server (only used when backend="vllm")."""

    vllm_model_id: str | None = None
    """Model ID to send in vLLM OpenAI-compatible requests.

    Note:
        This is intentionally separate from `model_name`, which is used for
        tokenization/segmentation (and may be a local filesystem path).
    """


class HistoryEncoder(nn.Module):
    """Unified history encoder for Q-function state representation.

    This encoder combines task and context into a single formatted string,
    then encodes it in one forward pass. This is more efficient than
    encoding task and context separately.

    Format: "[Task]\n{task}\n\n[Context]\n{context}"

    Token allocation priority:
    1. Task description (as many tokens as needed)
    2. Minimum recent turns (guaranteed)
    3. Additional context (fill remaining budget)

    Attributes:
        encoder_dim: Hidden size from the pretrained model (auto-detected).
        output_dim: Same as encoder_dim (no projection needed).

    Example:
        >>> config = HistoryEncoderConfig(max_tokens=512)
        >>> encoder = HistoryEncoder(config)
        >>> messages = [
        ...     {"role": "system", "content": "You are an assistant..."},
        ...     {"role": "user", "content": "<task_description>Your task is to boil water.</task_description>"},
        ...     {"role": "assistant", "content": "I will turn on the stove."},
        ... ]
        >>> embedding = encoder(messages=messages)
        >>> embedding.shape  # Depends on encoder's hidden_size
        torch.Size([1024])  # For Qwen3-Embedding-0.6B
    """

    def __init__(self, config: HistoryEncoderConfig | None = None):
        """Initialize HistoryEncoder.

        Args:
            config: Encoder configuration. Uses defaults if None.
        """
        super().__init__()
        self.config = config or HistoryEncoderConfig()

        # Initialize based on backend type
        if self.config.backend == "vllm":
            self._init_vllm_backend()
        else:  # Default to HuggingFace
            self._init_hf_backend()

        # Output dim is now directly encoder_dim (no projection needed)
        # Task + context are combined into single string before encoding
        self.output_dim = self.encoder_dim

    def _init_hf_backend(self) -> None:
        """Initialize HuggingFace backend."""
        # Set padding side based on pooling mode
        # - last_token: left padding (decoder-only models like Qwen3-Embedding)
        # - cls: right padding (encoder models like BERT)
        padding_side = "left" if self.config.pooling_mode == "last_token" else "right"

        # Load tokenizer and pretrained model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side=padding_side,
        )
        self.encoder = AutoModel.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=False,  # Disable lazy loading to avoid meta tensors
        )

        # Freeze encoder parameters if requested
        if self.config.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Get encoder output dimension (auto-detected from model)
        self.encoder_dim = self.encoder.config.hidden_size

        # Mark that we're using HF backend
        self.vllm_client = None

    def _init_vllm_backend(self) -> None:
        """Initialize vLLM backend."""
        from .vllm_client import VLLMClient, VLLMClientConfig

        # Load tokenizer for segmentation only
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )

        # Initialize vLLM client
        # vLLM expects a served model ID (often a short name), not a local path.
        request_model = self.config.vllm_model_id or Path(self.config.model_name).name
        client_config = VLLMClientConfig(
            base_url=self.config.vllm_base_url,
            model_name=self.config.model_name,  # legacy/fallback
            request_model=request_model,
        )
        self.vllm_client = VLLMClient(client_config)

        # Get encoder dimension from vLLM
        self.encoder_dim = self.vllm_client.encoder_dim

        # No local encoder for vLLM backend
        self.encoder = None

    def segment_history(
        self,
        messages: list[dict[str, Any]],
    ) -> str:
        """Format message list into a single string with task and context.

        Uses the standalone segment_history_standalone function to ensure
        consistency between training (precomputed) and inference (online).

        Token allocation priority:
        1. Task description (as many tokens as needed, up to max_tokens)
        2. Minimum recent turns (guaranteed)
        3. Additional context (fill remaining budget)

        Args:
            messages: List of message dicts [{role, content}, ...]

        Returns:
            Formatted string: "[Task]\n{task}\n\n[Context]\n{context}"
        """
        return segment_history_standalone(
            messages=messages,
            tokenizer=self.tokenizer,
            max_tokens=self.config.max_tokens,
            min_recent_turns=self.config.min_recent_turns,
        )

    # Note: Helper methods (_extract_task_description, _count_tokens, etc.) are now
    # standalone functions (_xxx_standalone) called by segment_history_standalone()
    # to ensure training/inference consistency.

    def _pool(
        self,
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply pooling based on config.

        Args:
            last_hidden_states: Shape (batch, seq_len, hidden_dim).
            attention_mask: Shape (batch, seq_len).

        Returns:
            Pooled embeddings of shape (batch, hidden_dim).
        """
        if self.config.pooling_mode == "cls":
            return last_hidden_states[:, 0, :]
        else:  # last_token
            return last_token_pool(last_hidden_states, attention_mask)

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode a single text string to a vector.

        Args:
            text: Input text string.

        Returns:
            Embedding tensor of shape (hidden_size,).
        """
        if not text:
            # Return zero vector for empty text
            return torch.zeros(self.encoder_dim, device=self._get_device())

        # Use vLLM backend if configured
        if self.config.backend == "vllm" and self.vllm_client is not None:
            emb = self.vllm_client.embed([text])[0]
            return torch.tensor(emb, dtype=torch.float32)

        # HuggingFace backend
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_tokens,
            padding=True,
        )
        inputs = {k: v.to(self._get_device()) for k, v in inputs.items()}

        if self.config.freeze_encoder:
            with torch.no_grad():
                outputs = self.encoder(**inputs)
        else:
            outputs = self.encoder(**inputs)

        # Use configured pooling strategy
        return self._pool(
            outputs.last_hidden_state,
            inputs["attention_mask"],
        ).squeeze(0)

    def forward(
        self,
        messages: list[dict[str, Any]] | None = None,
        text: str | None = None,
    ) -> torch.Tensor:
        """Encode history to state vector.

        Can pass messages to auto-segment, or pass pre-formatted text.

        Args:
            messages: Message list (optional).
            text: Pre-formatted text string (optional).

        Returns:
            State embedding of shape (output_dim,).
        """
        # Auto-segment if messages provided
        if messages is not None:
            text = self.segment_history(messages)

        # Single forward pass
        embedding = self.encode_text(text or "")

        # L2 normalize to match precomputed embeddings
        return F.normalize(embedding.unsqueeze(0), p=2, dim=-1).squeeze(0)

    def encode_texts_batch(self, texts: list[str]) -> torch.Tensor:
        """Batch encode multiple texts.

        Args:
            texts: List of text strings.

        Returns:
            Embeddings of shape (batch_size, encoder_dim).
        """
        if not texts:
            return torch.zeros(0, self.encoder_dim, device=self._get_device())

        # Use vLLM backend if configured
        # Use single requests per text to match precomputation behavior
        if self.config.backend == "vllm" and self.vllm_client is not None:
            embeddings_list = []
            for text in texts:
                if text:
                    emb = self.vllm_client.embed([text])[0]
                else:
                    emb = [0.0] * self.encoder_dim
                embeddings_list.append(emb)
            embeddings = torch.tensor(embeddings_list, dtype=torch.float32)
            return F.normalize(embeddings, p=2, dim=-1)

        # HuggingFace backend
        # Handle empty strings: record positions, process separately
        non_empty_indices = [i for i, t in enumerate(texts) if t]
        non_empty_texts = [t for t in texts if t]

        if not non_empty_texts:
            # All empty strings
            return torch.zeros(len(texts), self.encoder_dim, device=self._get_device())

        # Batch tokenize (padding handled by tokenizer based on padding_side)
        inputs = self.tokenizer(
            non_empty_texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_tokens,
            padding=True,
        )
        inputs = {k: v.to(self._get_device()) for k, v in inputs.items()}

        # Single forward pass
        if self.config.freeze_encoder:
            with torch.no_grad():
                outputs = self.encoder(**inputs)
        else:
            outputs = self.encoder(**inputs)

        # Apply configured pooling strategy
        non_empty_embeddings = self._pool(
            outputs.last_hidden_state,
            inputs["attention_mask"],
        )

        # Fill empty string positions with zero vectors
        result = torch.zeros(len(texts), self.encoder_dim, device=self._get_device())
        for i, orig_idx in enumerate(non_empty_indices):
            result[orig_idx] = non_empty_embeddings[i]

        # L2 normalize for consistency
        return F.normalize(result, p=2, dim=-1)

    def forward_batch(
        self,
        batch_messages: list[list[dict[str, Any]]],
    ) -> torch.Tensor:
        """True batch encoding.

        Args:
            batch_messages: List of message lists.

        Returns:
            Embeddings of shape (batch_size, output_dim).
        """
        # Step 1: Batch segmentation (CPU string operations, fast)
        texts = [self.segment_history(messages) for messages in batch_messages]

        # Step 2: Single batch encoding (one GPU forward pass)
        # encode_texts_batch already applies L2 normalization
        return self.encode_texts_batch(texts)

    def _get_device(self) -> torch.device:
        """Get the device where the model is located."""
        if self.encoder is None:
            # vLLM backend: embeddings are on CPU
            return torch.device("cpu")
        return next(self.encoder.parameters()).device


# =============================================================================
# Standalone Segmentation Functions (for precomputation)
# =============================================================================


def _extract_task_description_standalone(messages: list[dict[str, Any]]) -> str:
    """Extract task description from messages (standalone version).

    Supported formats:
    - ScienceWorld: first user message contains <task_description>...</task_description>
    - HLE: first user message contains a "## HLE Question" section

    Args:
        messages: List of message dicts.

    Returns:
        Extracted task description text.
    """
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")

            # Try to extract <task_description> tag content
            match = re.search(
                r"<task_description>(.*?)</task_description>",
                content,
                re.DOTALL,
            )
            if match:
                return match.group(1).strip()

            # HLE: extract the question block (exclude trailing instructions after '---')
            if "## HLE Question" in content:
                after = content.split("## HLE Question", 1)[1].strip()
                # Split on a markdown horizontal rule line if present
                parts = re.split(r"\n\s*---\s*\n", after, maxsplit=1)
                question_block = (parts[0] if parts else after).strip()
                if question_block:
                    return question_block

            # Fallback: find "Your task is to" sentence
            match = re.search(r"Your task is to[^.]+\.", content)
            if match:
                return match.group(0)

            # Last fallback: return full first user message
            # (token truncation happens later based on max_tokens)
            return content.strip()

    return ""


def _extract_initial_observation_standalone(content: str) -> str:
    """Extract initial_observation from first user message (standalone version).

    Args:
        content: First user message content.

    Returns:
        Extracted observation text.
    """
    match = re.search(
        r"<initial_observation>(.*?)</initial_observation>",
        content,
        re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return ""


def _count_tokens_standalone(text: str, tokenizer: Any) -> int:
    """Count actual tokens in text using tokenizer (standalone version).

    Args:
        text: Input text.
        tokenizer: HuggingFace tokenizer.

    Returns:
        Number of tokens.
    """
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def _truncate_to_tokens_standalone(text: str, max_tokens: int, tokenizer: Any) -> str:
    """Truncate text to fit within max_tokens (standalone version).

    Args:
        text: Input text.
        max_tokens: Maximum number of tokens.
        tokenizer: HuggingFace tokenizer.

    Returns:
        Truncated text.
    """
    if not text:
        return ""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return tokenizer.decode(truncated_tokens)


def _truncate_context_standalone(
    messages: list[dict[str, Any]],
    max_tokens: int,
    min_recent_turns: int,
    tokenizer: Any,
    start_turn: int = 1,
) -> str:
    """Truncate context to fit token budget, keeping recent messages (standalone version).

    Strategy:
    1. Start from the newest message and go backwards
    2. Accumulate tokens until max_tokens is reached
    3. Try to keep at least min_recent_turns, but max_tokens is a hard limit

    Note: max_tokens is a HARD limit. If min_recent_turns cannot fit within
    the budget, we stop early rather than exceeding the limit.

    Args:
        messages: List of context message dicts.
        max_tokens: Maximum tokens allowed for context (hard limit).
        min_recent_turns: Target number of recent turns to keep (best effort).
        tokenizer: HuggingFace tokenizer.
        start_turn: The turn number of the first message (for correct numbering).

    Returns:
        Formatted context string with turn numbers.
    """
    if not messages or max_tokens <= 0:
        return ""

    # First pass: calculate total turns for correct numbering
    total_turns = start_turn - 1
    for msg in messages:
        if msg.get("role") == "user":
            total_turns += 1

    # Collect messages from newest to oldest
    selected = []
    total_tokens = 0
    turn_count = 0
    current_turn = total_turns  # Start from the last turn

    # Fixed overhead per message for formatting:
    # "[Turn N - role]\n" (~6-8 tokens) + "\n\n" separator (~2 tokens)
    FORMAT_OVERHEAD_PER_MSG = 10

    for msg in reversed(messages):
        content = msg.get("content", "")
        role = msg.get("role", "")

        # Count actual tokens using tokenizer + formatting overhead
        msg_tokens = _count_tokens_standalone(content, tokenizer) + FORMAT_OVERHEAD_PER_MSG

        # Hard limit: stop if adding this message would exceed max_tokens
        # (even if we haven't reached min_recent_turns yet)
        if total_tokens + msg_tokens > max_tokens:
            break

        # Store message with its turn number
        selected.append((msg, current_turn))
        total_tokens += msg_tokens

        # Count turns (user message marks the start of a turn)
        if role == "user":
            turn_count += 1
            current_turn -= 1

    # Reverse back to chronological order
    selected = list(reversed(selected))

    # Format as text with turn numbers
    parts = []
    for msg, turn_num in selected:
        role = msg.get("role", "")
        content = msg.get("content", "")
        parts.append(f"[Turn {turn_num} - {role}]\n{content}")

    return "\n\n".join(parts)


def segment_history_standalone(
    messages: list[dict[str, Any]],
    tokenizer: Any,
    max_tokens: int = 512,
    min_recent_turns: int = 1,
) -> str:
    """Format message list into a single string with task and context.

    Extracts task description and recent context, then combines them into
    a single formatted string for encoding.

    Token allocation priority:
    1. Task description (as many tokens as needed, up to max_tokens)
    2. Minimum recent turns (guaranteed)
    3. Additional context (fill remaining budget)

    Args:
        messages: List of message dicts [{role, content}, ...]
        tokenizer: HuggingFace tokenizer for token counting.
        max_tokens: Maximum total tokens for encoding.
        min_recent_turns: Minimum number of recent turns to always keep.

    Returns:
        Formatted string: "[Task]\n{task}\n\n[Context]\n{context}"
    """
    # Step 1: Extract and count task description tokens
    task_segment = _extract_task_description_standalone(messages)
    task_tokens = _count_tokens_standalone(task_segment, tokenizer)

    # Reserve tokens for format markers
    format_overhead = 20  # "[Task]\n" + "\n\n[Context]\n"

    # If task alone exceeds budget, truncate it
    if task_tokens >= max_tokens - format_overhead:
        task_segment = _truncate_to_tokens_standalone(
            task_segment, max_tokens - format_overhead, tokenizer
        )
        return f"[Task]\n{task_segment}"  # No room for context

    # Step 2: Collect context messages (skip system and task_description part)
    context_messages = []
    first_user_seen = False

    for msg in messages:
        role = msg.get("role", "")

        if role == "system":
            continue  # Skip system prompt
        elif role == "user" and not first_user_seen:
            first_user_seen = True
            # For first user message, only keep observation part
            content = msg.get("content", "")
            obs_content = _extract_initial_observation_standalone(content)
            if obs_content:
                context_messages.append({"role": "user", "content": obs_content})
        else:
            context_messages.append(msg)

    # Step 3: Truncate context with remaining token budget
    remaining_tokens = max_tokens - task_tokens - format_overhead
    context_segment = _truncate_context_standalone(
        context_messages, remaining_tokens, min_recent_turns, tokenizer
    )

    # Step 4: Combine into single string
    if context_segment:
        return f"[Task]\n{task_segment}\n\n[Context]\n{context_segment}"
    else:
        return f"[Task]\n{task_segment}"


# =============================================================================
# HFPrecomputer for Embedding Precomputation
# =============================================================================


@dataclass
class PrecomputeConfig:
    """Configuration for embedding precomputation.

    Attributes:
        enabled: Whether to precompute embeddings.
        backend: Embedding backend type ("hf" for HuggingFace, "vllm" for vLLM server).
        model_name: Path to embedding model.
        batch_size: Batch size for encoding.
        max_tokens: Maximum tokens per segment.
        min_recent_turns: Minimum recent turns to keep in context.
        pooling_mode: Pooling strategy ("cls" or "last_token").
            Must match HistoryEncoderConfig.pooling_mode for consistency.
        vllm_base_url: Base URL of vLLM server (only used when backend="vllm").
        vllm_timeout: Request timeout for vLLM server (seconds).
        vllm_max_retries: Maximum retry attempts for vLLM requests.
    """

    enabled: bool = False
    backend: BackendType = "hf"
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    batch_size: int = 64  # HuggingFace batch size
    max_tokens: int = 512
    min_recent_turns: int = 1
    pooling_mode: Literal["cls", "last_token"] = "last_token"

    # vLLM-specific configuration
    vllm_base_url: str = "http://localhost:8000"
    vllm_timeout: float = 60.0
    vllm_max_retries: int = 3
    vllm_model_id: str | None = None


class HFPrecomputer:
    """HuggingFace-based embedding precomputer.

    Uses the same HuggingFace backend as online inference (HistoryEncoder),
    ensuring training/inference consistency.

    Key design:
    - Same tokenizer settings (padding_side based on pooling_mode)
    - Same pooling strategy (last_token or CLS based on pooling_mode)
    - Same L2 normalization

    Example:
        >>> config = PrecomputeConfig(enabled=True, batch_size=64)
        >>> precomputer = HFPrecomputer(config)
        >>> embeddings = precomputer.precompute(samples)
        >>> precomputer.cleanup()  # Free GPU memory
    """

    def __init__(self, config: PrecomputeConfig):
        """Initialize HFPrecomputer.

        Args:
            config: Precomputation configuration.
        """
        import gc

        self.config = config

        # Padding side depends on pooling mode (matches HistoryEncoder)
        # - last_token: left padding so EOS is always at position -1
        # - cls: right padding so CLS is always at position 0
        padding_side = "left" if config.pooling_mode == "last_token" else "right"

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            padding_side=padding_side,
        )

        # Load HuggingFace model
        print(f"Loading HuggingFace embedding model: {config.model_name}")
        self.encoder = AutoModel.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,  # FP16 for memory efficiency
        ).cuda().eval()

        # Get encoder dimension
        self.encoder_dim = self.encoder.config.hidden_size
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
            Embeddings tensor of shape (N, encoder_dim).
        """
        from tqdm import tqdm

        n_samples = len(samples)
        print(f"Precomputing embeddings for {n_samples} samples...")

        # Step 1: Segment all messages into combined strings (CPU, fast)
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

        # Step 2: Batch encode with HuggingFace (single pass)
        embeddings = self._encode_batch(
            texts, desc="Encoding histories", show_progress=show_progress
        )

        print(f"Precomputation complete. Shape: ({n_samples}, {self.encoder_dim})")
        return embeddings

    def _pool(
        self,
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply pooling based on config (matches HistoryEncoder._pool).

        Args:
            last_hidden_states: Shape (batch, seq_len, hidden_dim).
            attention_mask: Shape (batch, seq_len).

        Returns:
            Embeddings of shape (batch, hidden_dim).
        """
        if self.config.pooling_mode == "cls":
            return last_hidden_states[:, 0, :]
        else:  # last_token
            return last_token_pool(last_hidden_states, attention_mask)

    def _encode_batch(
        self,
        texts: list[str],
        desc: str = "Encoding",
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Encode texts in batches using HuggingFace.

        Uses the same logic as HistoryEncoder.encode_texts_batch():
        - Padding side based on pooling_mode
        - Pooling based on pooling_mode (CLS or last_token)
        - L2 normalization

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
                # Tokenize (padding_side already set in __init__ based on pooling_mode)
                inputs = self.tokenizer(
                    non_empty_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_tokens,
                    padding=True,
                ).to("cuda")

                with torch.no_grad():
                    outputs = self.encoder(**inputs)

                # Apply pooling based on pooling_mode (matches HistoryEncoder._pool)
                non_empty_embs = self._pool(
                    outputs.last_hidden_state,
                    inputs["attention_mask"],
                )
                # L2 normalize (matches HistoryEncoder.encode_texts_batch)
                non_empty_embs = F.normalize(non_empty_embs.float(), p=2, dim=-1)
            else:
                non_empty_embs = torch.zeros(0, self.encoder_dim, device="cuda")

            # Fill in zeros for empty strings
            batch_embs = torch.zeros(len(batch_texts), self.encoder_dim, device="cuda")
            for j, orig_idx in enumerate(non_empty_indices):
                batch_embs[orig_idx] = non_empty_embs[j]

            # Move to CPU to save GPU memory during precomputation
            embeddings_list.append(batch_embs.cpu())

        return torch.cat(embeddings_list, dim=0)

    def cleanup(self) -> None:
        """Release GPU memory by deleting model and clearing caches."""
        import gc

        print("Cleaning up HuggingFace model and freeing GPU memory...")
        del self.encoder
        del self.tokenizer

        torch.cuda.empty_cache()
        gc.collect()
        print("Cleanup complete.")

    def save(
        self,
        path: Path | str,
        embeddings: torch.Tensor,
        sample_ids: list[tuple[str, int]],
        source_dirs: list[str],
        sample_metadata: list[dict] | None = None,
    ) -> None:
        """Save precomputed embeddings to disk.

        Args:
            path: Output file path (.pt file).
            embeddings: Precomputed embeddings tensor, shape (N, encoder_dim).
            sample_ids: List of (episode_id, step_idx) tuples for each embedding.
            source_dirs: List of source trajectory directories.
            sample_metadata: Optional list of metadata dicts for each sample
                (task_id, episode_id, step_idx, source_path).
        """
        from datetime import datetime

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        embeddings_cpu = embeddings.detach().cpu().contiguous()
        if embeddings_cpu.ndim != 2:
            raise ValueError(f"Expected embeddings of shape (N, D), got {tuple(embeddings_cpu.shape)}")
        if len(sample_ids) != embeddings_cpu.shape[0]:
            raise ValueError(
                f"sample_ids length ({len(sample_ids)}) does not match embeddings rows ({embeddings_cpu.shape[0]})"
            )

        data = {
            "embeddings": embeddings_cpu,
            "sample_ids": sample_ids,
            "sample_metadata": sample_metadata,
            "config": {
                "backend": "hf",  # Mark as HuggingFace backend
                "model_name": self.config.model_name,
                "max_tokens": self.config.max_tokens,
                "min_recent_turns": self.config.min_recent_turns,
                "pooling_mode": self.config.pooling_mode,
                "encoder_dim": self.encoder_dim,
            },
            "source_dirs": source_dirs,
            "num_samples": len(sample_ids),
            "created_at": datetime.now().isoformat(),
            "format_version": 2,  # New format with episode_id
        }

        torch.save(data, path)
        print(f"Saved {len(sample_ids)} embeddings to {path}")
        print(f"  Config: max_tokens={self.config.max_tokens}, pooling_mode={self.config.pooling_mode}, encoder_dim={self.encoder_dim}")

    @staticmethod
    def load(path: Path | str) -> dict:
        """Load precomputed embeddings from disk.

        Args:
            path: Path to precomputed embeddings file.

        Returns:
            Dict with keys: embeddings, sample_ids, config, source_dirs,
            num_samples, created_at, sample_metadata.

        Raises:
            ValueError: If file uses old format (format_version < 2).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Precomputed embeddings file not found: {path}")

        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            # Older PyTorch without `weights_only` argument.
            data = torch.load(path, map_location="cpu")

        if not isinstance(data, dict):
            raise ValueError(f"Invalid precomputed embeddings file format: {path}")
        if "embeddings" not in data or "sample_ids" not in data or "config" not in data:
            raise ValueError(f"Missing required keys in precomputed embeddings file: {path}")

        # Require format_version >= 2 (uses episode_id)
        format_version = data.get("format_version", 1)
        if format_version < 2:
            raise ValueError(
                f"Precomputed embeddings file uses old format (version {format_version}).\n"
                f"Please regenerate embeddings with: python scripts/precompute_embeddings.py"
            )

        embeddings = data["embeddings"]
        if not isinstance(embeddings, torch.Tensor):
            raise ValueError(f"Expected 'embeddings' to be a torch.Tensor, got {type(embeddings)}")
        data["embeddings"] = embeddings.detach().cpu()

        print(f"Loaded {data['num_samples']} embeddings from {path}")
        print(f"  Config: max_tokens={data['config']['max_tokens']}, encoder_dim={data['config']['encoder_dim']}")
        print(f"  Created: {data['created_at']}")

        return data


# =============================================================================
# VLLMPrecomputer for vLLM-based Embedding Precomputation
# =============================================================================


class VLLMPrecomputer:
    """vLLM-based embedding precomputer.

    Uses vLLM's OpenAI-compatible /v1/embeddings API to compute embeddings.
    Ensures consistency with HFPrecomputer by using the same tokenizer for
    text segmentation.

    Features:
    - SQLite-based embedding cache (optional) to avoid redundant requests
    - Multi-threaded parallel requests (one text per request)
    - Token length assertion after segmentation

    Example:
        >>> config = PrecomputeConfig(enabled=True, backend="vllm", vllm_base_url="http://localhost:8000")
        >>> precomputer = VLLMPrecomputer(config, cache_path="cache.db")
        >>> embeddings = precomputer.precompute(samples, max_workers=16)
        >>> precomputer.cleanup()
    """

    def __init__(self, config: PrecomputeConfig, cache_path: Path | str | None = None):
        """Initialize VLLMPrecomputer.

        Args:
            config: Precomputation configuration.
            cache_path: Optional path to SQLite cache file for embeddings.
        """
        from .vllm_client import EmbeddingCache, VLLMClient, VLLMClientConfig

        self.config = config

        # Load tokenizer for segmentation (same as HF for consistency)
        print(f"Loading tokenizer for segmentation: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
        )

        # Initialize vLLM client
        request_model = config.vllm_model_id or Path(config.model_name).name
        client_config = VLLMClientConfig(
            base_url=config.vllm_base_url,
            model_name=config.model_name,  # legacy/fallback
            request_model=request_model,
            timeout=config.vllm_timeout,
            max_retries=config.vllm_max_retries,
            batch_size=config.batch_size,
        )
        print(f"Connecting to vLLM server at {config.vllm_base_url}")
        self.client = VLLMClient(client_config)

        # Get encoder dimension from vLLM
        self.encoder_dim = self.client.encoder_dim
        print(f"Encoder dimension: {self.encoder_dim}")

        # Initialize cache (optional)
        self.cache: EmbeddingCache | None = None
        if cache_path is not None:
            self.cache = EmbeddingCache(cache_path, model_name=request_model)
            print(f"Embedding cache enabled: {cache_path} (model: {request_model})")

    def precompute(
        self,
        samples: list[dict[str, Any]],
        show_progress: bool = True,
        max_workers: int = 8,
    ) -> torch.Tensor:
        """Precompute embeddings for all samples.

        Uses multi-threaded parallel processing where each thread:
        1. Segments the sample's messages
        2. Asserts token length
        3. Checks cache (if enabled)
        4. Encodes via vLLM (if cache miss)
        5. Stores in cache

        Args:
            samples: List of sample dicts with 'messages' key.
            show_progress: Whether to show progress bars.
            max_workers: Maximum parallel threads.

        Returns:
            Embeddings tensor of shape (N, encoder_dim).
        """
        import threading
        from concurrent.futures import Future, ThreadPoolExecutor

        from tqdm import tqdm

        n_samples = len(samples)
        print(f"Precomputing embeddings for {n_samples} samples using vLLM...")
        print(f"  Max workers: {max_workers}")
        if self.cache:
            print(f"  Cache: {self.cache.stats()}")

        # Thread-safe counters for cache stats
        cache_hits = 0
        cache_misses = 0
        counter_lock = threading.Lock()

        def process_sample(sample: dict[str, Any]) -> list[float]:
            """Process a single sample: segment -> cache check -> encode."""
            nonlocal cache_hits, cache_misses

            # Step 1: Segment messages
            text = segment_history_standalone(
                messages=sample["messages"],
                tokenizer=self.tokenizer,
                max_tokens=self.config.max_tokens,
                min_recent_turns=self.config.min_recent_turns,
            )

            # Step 2: Assert token length
            n_tokens = len(self.tokenizer.encode(text, add_special_tokens=False))
            assert n_tokens <= self.config.max_tokens, (
                f"Segmented text has {n_tokens} tokens > max_tokens={self.config.max_tokens}. "
                f"Sample: task_id={sample.get('task_id')}, step_idx={sample.get('step_idx')}"
            )

            # Step 3: Check cache
            if self.cache is not None:
                cached_emb = self.cache.get(text)
                if cached_emb is not None:
                    with counter_lock:
                        cache_hits += 1
                    return cached_emb

            # Step 4: Encode via vLLM
            with counter_lock:
                cache_misses += 1
            if not text:
                emb = [0.0] * self.encoder_dim
            else:
                emb = self.client.embed([text])[0]

            # Step 5: Store in cache
            if self.cache is not None:
                self.cache.put(text, emb)

            return emb

        # Submit all tasks using ThreadPoolExecutor
        results: list[list[float] | None] = [None] * n_samples

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks in order, futures list maintains correspondence
            futures: list[Future[list[float]]] = [
                executor.submit(process_sample, sample) for sample in samples
            ]

            # Collect results in order with progress bar
            pbar = tqdm(
                total=n_samples,
                desc="Segment + Encode (parallel)",
                disable=not show_progress,
            )
            for i, future in enumerate(futures):
                results[i] = future.result()
                pbar.update(1)
            pbar.close()

        # Convert to tensor and L2 normalize
        embeddings = torch.tensor(results, dtype=torch.float32)
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        print(f"Precomputation complete. Shape: ({n_samples}, {self.encoder_dim})")
        if self.cache:
            print(f"  Cache hits: {cache_hits}, misses: {cache_misses}")
            print(f"  Final cache stats: {self.cache.stats()}")

        return embeddings

    def cleanup(self) -> None:
        """Release resources."""
        print("Cleaning up vLLM precomputer resources...")
        if self.cache is not None:
            self.cache.close()
        del self.tokenizer
        del self.client
        print("Cleanup complete.")

    def save(
        self,
        path: Path | str,
        embeddings: torch.Tensor,
        sample_ids: list[tuple[str, int]],
        source_dirs: list[str],
        sample_metadata: list[dict] | None = None,
    ) -> None:
        """Save precomputed embeddings to disk.

        Args:
            path: Output file path (.pt file).
            embeddings: Precomputed embeddings tensor, shape (N, encoder_dim).
            sample_ids: List of (episode_id, step_idx) tuples for each embedding.
            source_dirs: List of source trajectory directories.
            sample_metadata: Optional list of metadata dicts for each sample
                (task_id, episode_id, step_idx, source_path).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        embeddings_cpu = embeddings.detach().cpu().contiguous()
        if embeddings_cpu.ndim != 2:
            raise ValueError(f"Expected embeddings of shape (N, D), got {tuple(embeddings_cpu.shape)}")
        if len(sample_ids) != embeddings_cpu.shape[0]:
            raise ValueError(
                f"sample_ids length ({len(sample_ids)}) does not match embeddings rows ({embeddings_cpu.shape[0]})"
            )

        data = {
            "embeddings": embeddings_cpu,
            "sample_ids": sample_ids,
            "sample_metadata": sample_metadata,
            "config": {
                "backend": "vllm",  # Mark as vLLM backend
                "model_name": self.config.model_name,
                "max_tokens": self.config.max_tokens,
                "min_recent_turns": self.config.min_recent_turns,
                "pooling_mode": self.config.pooling_mode,  # Match HFPrecomputer format
                "encoder_dim": self.encoder_dim,
                "vllm_base_url": self.config.vllm_base_url,
                "vllm_model_id": self.config.vllm_model_id or Path(self.config.model_name).name,
            },
            "source_dirs": source_dirs,
            "num_samples": len(sample_ids),
            "created_at": datetime.now().isoformat(),
            "format_version": 2,  # New format with episode_id
        }

        torch.save(data, path)
        print(f"Saved {len(sample_ids)} embeddings to {path}")
        print(f"  Config: backend=vllm, max_tokens={self.config.max_tokens}, pooling_mode={self.config.pooling_mode}, encoder_dim={self.encoder_dim}")


# =============================================================================
# Factory Function
# =============================================================================


def create_precomputer(
    config: PrecomputeConfig,
    cache_path: Path | str | None = None,
) -> HFPrecomputer | VLLMPrecomputer:
    """Create a precomputer based on configuration.

    Factory function that returns the appropriate precomputer based on
    the backend specified in the configuration.

    Args:
        config: Precomputation configuration.
        cache_path: Optional path to SQLite cache file (only used for vLLM backend).

    Returns:
        HFPrecomputer if backend="hf", VLLMPrecomputer if backend="vllm".

    Raises:
        ValueError: If backend is unknown.

    Example:
        >>> config = PrecomputeConfig(enabled=True, backend="hf")
        >>> precomputer = create_precomputer(config)
        >>> isinstance(precomputer, HFPrecomputer)
        True

        >>> config = PrecomputeConfig(enabled=True, backend="vllm")
        >>> precomputer = create_precomputer(config, cache_path="cache.db")
        >>> isinstance(precomputer, VLLMPrecomputer)
        True
    """
    if config.backend == "hf":
        if cache_path is not None:
            print("Warning: cache_path is ignored for HuggingFace backend")
        return HFPrecomputer(config)
    elif config.backend == "vllm":
        return VLLMPrecomputer(config, cache_path=cache_path)
    else:
        raise ValueError(f"Unknown embedding backend: {config.backend}. Use 'hf' or 'vllm'.")
