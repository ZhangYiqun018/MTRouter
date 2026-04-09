"""Trajectory parsing and feature extraction.

This module provides classes for parsing trajectory JSON files and
extracting features for training learned routers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from miniagenticrouter.research.training.error_detector import (
        AECConfig,
        ErrorDetector,
    )

# =============================================================================
# Benchmark Normalization Configuration
# =============================================================================


@dataclass
class BenchmarkNormConfig:
    """Normalization configuration for a specific benchmark.

    Attributes:
        benchmark: Benchmark identifier (e.g., "scienceworld", "hle").
        score_range: Tuple of (min_score, max_score) for score normalization.
        cost_limit: Default cost limit in USD (used as fallback).
    """

    benchmark: str
    score_range: tuple[float, float]
    cost_limit: float

    def normalize_score(self, raw_score: float) -> float:
        """Normalize raw score to [0, 1] range.

        Args:
            raw_score: Score in benchmark's native scale.

        Returns:
            Normalized score in [0, 1].
        """
        min_s, max_s = self.score_range
        if max_s == min_s:
            return 1.0 if raw_score >= max_s else 0.0
        return (raw_score - min_s) / (max_s - min_s)


# Predefined benchmark configurations
BENCHMARK_CONFIGS: dict[str, BenchmarkNormConfig] = {
    "scienceworld": BenchmarkNormConfig(
        benchmark="scienceworld",
        score_range=(-100.0, 100.0),  # Environment score range
        cost_limit=2.0,  # From data_split.yaml
    ),
    "hle": BenchmarkNormConfig(
        benchmark="hle",
        score_range=(0.0, 1.0),  # Binary: 0 (incorrect) or 1 (correct)
        cost_limit=5.0,  # From hle_data_split.yaml
    ),
}

# Default for backward compatibility
DEFAULT_BENCHMARK = "scienceworld"


# =============================================================================
# AEC (Annealed Error Cost) Calculation Functions
# =============================================================================


def compute_warmup_weight(
    p: float, p0: float, p1: float, w_min: float, w_max: float
) -> float:
    """Compute warmup weight using piecewise linear function.

    Args:
        p: Progress value (0 to 1).
        p0: Progress threshold where ramp starts.
        p1: Progress threshold where ramp ends.
        w_min: Minimum weight (used for p <= p0).
        w_max: Maximum weight (used for p >= p1).

    Returns:
        Warmup weight value.
    """
    if p <= p0:
        return w_min
    elif p >= p1:
        return w_max
    else:
        return w_min + (w_max - w_min) * (p - p0) / (p1 - p0)


def compute_step_aec_penalty(
    step_idx: int,
    error_types: list[str],
    severity_map: dict[str, str],
    aec_config: "AECConfig",
    N: int,
) -> float:
    """Compute AEC penalty for a single step.

    Args:
        step_idx: Step index (0-based).
        error_types: List of error rule names detected at this step.
        severity_map: Mapping from rule_name to severity.
        aec_config: AEC configuration.
        N: Expected steps for this task category.

    Returns:
        AEC penalty value for this step.
    """
    if not error_types or N <= 0:
        return 0.0

    base = 1.0 / N
    t = step_idx + 1  # 1-indexed for calculation
    p = min(t / N, 1.0)
    w = compute_warmup_weight(p, aec_config.p0, aec_config.p1, aec_config.w_min, aec_config.w_max)

    # Find the most severe error (max coefficient)
    max_coef = 0.0
    for error in error_types:
        severity = severity_map.get(error, "medium")
        coef = aec_config.severity_coefficients.get(severity, 1.0)
        max_coef = max(max_coef, coef)

    # Compute penalty with minimum floor
    penalty = base * max_coef * w
    min_penalty = aec_config.min_penalty_factor * base
    return max(penalty, min_penalty)


@dataclass
class StepFeature:
    """Features extracted from a single agent step.

    Corresponds to (h_t, a_t, c_t) tuple in the POMDP formulation.

    Attributes:
        step_idx: Step index (0-based).
        history_text: History context text h_t (truncated to max length).
        history_tokens: Estimated number of tokens in history.
        history_messages: Raw message list for encoder input.
        action_text: Action text a_t (assistant message content).
        action_tokens: Number of tokens in action.
        model_name: Name of the model that executed this step.
        model_idx: Index of the model in the pool (-1 if unknown).
        step_cost: Cost c_t for this step (USD).
        observation_text: Observation text (user response after action).
        propensity: Probability of selecting this model (only for roulette mode).
        prompt_tokens: Number of prompt tokens used.
        completion_tokens: Number of completion tokens used.
    """

    step_idx: int
    history_text: str
    history_tokens: int
    history_messages: list[dict[str, Any]]
    action_text: str
    action_tokens: int
    model_name: str
    model_idx: int
    step_cost: float
    observation_text: str
    propensity: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class TrajectoryFeature:
    """Features extracted from a complete trajectory.

    Attributes:
        task_id: Task identifier (e.g., "boil_var0").
        task_name: Task name (e.g., "boil").
        variation_idx: Variation index.
        final_score: Final environment score S_final.
        total_cost: Total cost for the episode.
        n_steps: Number of steps taken.
        exit_status: Exit status (e.g., "Submitted", "LimitsExceeded").
        steps: List of step features.
        meta: Additional metadata from the trajectory.
    """

    task_id: str
    task_name: str
    variation_idx: int
    final_score: float
    total_cost: float
    n_steps: int
    exit_status: str
    steps: list[StepFeature]
    meta: dict[str, Any] = field(default_factory=dict)

    def get_cumulative_cost(self, until_step: int | None = None) -> float:
        """Calculate cumulative cost up to a given step.

        Args:
            until_step: Step index (exclusive). If None, returns total cost.

        Returns:
            Cumulative cost from step 0 to until_step.
        """
        steps = self.steps[:until_step] if until_step is not None else self.steps
        return sum(s.step_cost for s in steps)

    def get_remaining_cost(self, from_step: int) -> float:
        """Calculate remaining cost from a given step to the end.

        Args:
            from_step: Step index (inclusive).

        Returns:
            Total cost from from_step to the end.
        """
        return sum(s.step_cost for s in self.steps[from_step:])

    def get_model_distribution(self) -> dict[str, int]:
        """Count model usage across all steps.

        Returns:
            Dict mapping model names to number of uses.
        """
        dist: dict[str, int] = {}
        for step in self.steps:
            dist[step.model_name] = dist.get(step.model_name, 0) + 1
        return dist

    def to_training_samples(
        self,
        lambda_: float = 0.0,
        error_detector: ErrorDetector | None = None,
        aec_config: AECConfig | None = None,
    ) -> list[dict[str, Any]]:
        """Convert trajectory to training samples with normalized scores.

        Score normalization:
        - ScienceWorld: (-100, 100) -> (0, 1) via linear mapping
        - HLE: (0, 1) -> (0, 1) (already normalized)

        Cost normalization is deferred to dataset layer, where per-benchmark
        mean costs are computed across all trajectories.

        AEC (Annealed Error Cost) penalty:
        - If aec_config is provided and enabled, applies time-dependent error penalties
        - Each step's final_score is adjusted: S' = S - cumulative_penalty
        - cumulative_penalty = sum of remaining step penalties from current to end

        Each sample contains:
        - x: State features (history_text, step_idx, etc.)
        - a: Model index
        - final_score: Normalized score in [0, 1] (or AEC-adjusted if enabled)
        - remaining_cost: Raw cost in USD (normalized later in dataset)
        - error_detected: Whether errors were detected in observation
        - error_types: List of detected error types
        - error_penalty: Total penalty value (legacy, for backward compatibility)
        - original_final_score: Original normalized score (before AEC adjustment)
        - aec_penalty: Cumulative AEC penalty from this step to end
        - step_aec_penalty: AEC penalty for this specific step

        Args:
            lambda_: Cost penalty coefficient (used for target_value only).
            error_detector: Optional error detector for analyzing observations.
            aec_config: Optional AEC configuration for time-dependent penalties.

        Returns:
            List of training sample dicts.
        """
        # Compute episode_id from source_path for unique identification
        source_path = self.meta.get("source_path", "")
        episode_id = hashlib.sha256(source_path.encode()).hexdigest()[:16]

        # Get normalization config from meta
        norm_config: BenchmarkNormConfig = self.meta.get(
            "norm_config", BENCHMARK_CONFIGS[DEFAULT_BENCHMARK]
        )
        benchmark = self.meta.get("benchmark", DEFAULT_BENCHMARK)

        # Normalize score to [0, 1]
        normalized_score = norm_config.normalize_score(self.final_score)

        # AEC setup: get task category and expected steps N
        aec_enabled = aec_config is not None and aec_config.enabled and error_detector is not None
        N = 20  # default
        severity_map: dict[str, str] = {}
        if aec_enabled:
            # Get task category: HLE uses meta["category"], SW uses task_name
            task_category = self.meta.get("category", self.task_name)
            N = aec_config.get_expected_steps(benchmark, task_category)
            severity_map = error_detector.get_severity_map(benchmark)

        # First pass: collect error info and compute step penalties
        step_error_info: list[tuple[bool, list[str], float]] = []  # (detected, types, legacy_penalty)
        step_aec_penalties: list[float] = []

        for step in self.steps:
            error_detected = False
            error_types: list[str] = []
            error_penalty = 0.0  # Legacy penalty

            if error_detector is not None:
                error_result = error_detector.detect(step.observation_text, benchmark)
                error_detected = error_result.has_error
                error_types = error_result.errors
                error_penalty = error_result.penalty

            step_error_info.append((error_detected, error_types, error_penalty))

            # Compute AEC penalty for this step
            if aec_enabled:
                step_aec = compute_step_aec_penalty(
                    step.step_idx, error_types, severity_map, aec_config, N
                )
            else:
                step_aec = 0.0
            step_aec_penalties.append(step_aec)

        # Compute cumulative AEC penalties (remaining penalty from each step to end)
        cumulative_aec_penalties: list[float] = []
        remaining = sum(step_aec_penalties)
        for pen in step_aec_penalties:
            cumulative_aec_penalties.append(remaining)
            remaining -= pen

        # Second pass: generate samples
        samples = []
        for i, step in enumerate(self.steps):
            remaining_cost = self.get_remaining_cost(step.step_idx)
            # Cost normalization placeholder - will be computed in dataset layer
            # using per-benchmark mean cost statistics
            normalized_cost = remaining_cost  # Raw USD, normalized later

            # Get error info for this step
            error_detected, error_types, error_penalty = step_error_info[i]
            step_aec = step_aec_penalties[i]
            cumulative_aec = cumulative_aec_penalties[i]

            # Compute adjusted score (S' = S - cumulative_aec_penalty)
            adjusted_score = normalized_score - cumulative_aec if aec_enabled else normalized_score

            # Target value with lambda penalty (score is normalized, cost is raw)
            # Note: This target_value is for backward compatibility
            # The dual-head network uses separate score and cost targets
            target = adjusted_score - lambda_ * normalized_cost

            samples.append(
                {
                    "step_idx": step.step_idx,
                    "history_text": step.history_text,
                    "history_tokens": step.history_tokens,
                    "messages": step.history_messages,
                    "model_name": step.model_name,
                    "model_idx": step.model_idx,
                    "step_cost": step.step_cost,
                    "propensity": step.propensity,
                    "target_value": target,
                    "final_score": adjusted_score,  # AEC-adjusted (or original if disabled)
                    "original_final_score": normalized_score,  # Original normalized score
                    "remaining_cost": remaining_cost,  # Raw USD
                    "normalized_cost": normalized_cost,  # Placeholder, computed in dataset
                    "task_id": self.task_id,
                    "source_path": source_path,
                    "episode_id": episode_id,
                    "benchmark": benchmark,  # Track benchmark for cost normalization
                    "raw_score": self.final_score,  # Original score for debugging
                    # Error detection fields
                    "error_detected": error_detected,
                    "error_types": error_types,
                    "error_penalty": error_penalty,  # Legacy penalty
                    # AEC fields
                    "aec_penalty": cumulative_aec,  # Cumulative penalty from this step to end
                    "step_aec_penalty": step_aec,  # Penalty for this specific step
                }
            )

        return samples


@dataclass
class ParserConfig:
    """Configuration for TrajectoryParser.

    Attributes:
        max_history_chars: Maximum characters in history text.
        include_system_prompt: Whether to include system prompt in history.
        model_name_mapping: Mapping from model names to indices.
        chars_per_token: Estimated characters per token for token counting.
    """

    max_history_chars: int = 50000
    include_system_prompt: bool = True
    model_name_mapping: dict[str, int] = field(default_factory=dict)
    chars_per_token: float = 4.0


class TrajectoryParser:
    """Parse trajectory JSON files and extract training features.

    This class handles:
    - Loading and validating trajectory JSON format
    - Extracting step-by-step features (h_t, a_t, c_t)
    - Computing target values for Q-learning

    Supported format: mini-agentic-router-1

    Example:
        >>> parser = TrajectoryParser(model_names=["haiku", "sonnet", "opus"])
        >>> feature = parser.parse(Path("trajectory.traj.json"))
        >>> for step in feature.steps:
        ...     print(f"Step {step.step_idx}: model={step.model_name}, cost=${step.step_cost:.4f}")
    """

    def __init__(
        self,
        model_names: list[str] | None = None,
        config: ParserConfig | None = None,
    ):
        """Initialize TrajectoryParser.

        Args:
            model_names: List of model names for index mapping.
            config: Parser configuration.
        """
        self.config = config or ParserConfig()
        self.model_names = model_names or []

        # Build name to index mapping
        self._name_to_idx: dict[str, int] = {}
        if model_names:
            for idx, name in enumerate(model_names):
                self._name_to_idx[name] = idx
                # Also add normalized versions (lowercase, without provider prefix)
                self._name_to_idx[name.lower()] = idx
                if "/" in name:
                    short_name = name.split("/")[-1]
                    self._name_to_idx[short_name] = idx
                    self._name_to_idx[short_name.lower()] = idx

    def parse(self, path: Path | str) -> TrajectoryFeature:
        """Parse a single trajectory file.

        Args:
            path: Path to the trajectory JSON file.

        Returns:
            TrajectoryFeature object.

        Raises:
            ValueError: If the file format is invalid.
            FileNotFoundError: If the file does not exist.
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return self._parse_data(data, source_path=str(path))

    def parse_batch(
        self,
        paths: list[Path] | list[str],
        skip_errors: bool = True,
    ) -> Iterator[TrajectoryFeature]:
        """Parse multiple trajectory files.

        Args:
            paths: List of trajectory file paths.
            skip_errors: Whether to skip files with parse errors.

        Yields:
            TrajectoryFeature objects.
        """
        for path in paths:
            try:
                yield self.parse(path)
            except Exception as e:
                if skip_errors:
                    continue
                raise e

    def _parse_data(
        self,
        data: dict[str, Any],
        source_path: str = "",
    ) -> TrajectoryFeature:
        """Parse trajectory data dict.

        Args:
            data: Trajectory data dict.
            source_path: Source file path for error messages.

        Returns:
            TrajectoryFeature object.
        """
        # Validate format
        if "messages" not in data:
            raise ValueError(f"Missing 'messages' in trajectory: {source_path}")

        info = data.get("info", {})
        messages = data["messages"]

        # Detect benchmark type
        benchmark = self._detect_benchmark(info)
        norm_config = BENCHMARK_CONFIGS.get(benchmark, BENCHMARK_CONFIGS[DEFAULT_BENCHMARK])

        # Extract basic info
        task_name = info.get("task_name", "")
        variation_idx = info.get("variation_idx", 0)
        task_id = info.get("task_id", f"{task_name}_var{variation_idx}")
        raw_final_score = info.get("final_score")
        final_score = (
            float(raw_final_score)
            if raw_final_score is not None
            else self._infer_final_score(info, benchmark)
        )
        exit_status = info.get("exit_status", "")

        model_stats = info.get("model_stats", {})
        total_cost = model_stats.get("instance_cost", 0.0)
        n_api_calls = model_stats.get("api_calls", 0)

        # Parse steps
        steps = self._parse_steps(messages)

        # Extract category for HLE (from benchmark_data)
        benchmark_data = info.get("benchmark_data", {})
        category = benchmark_data.get("category", "")

        return TrajectoryFeature(
            task_id=task_id,
            task_name=task_name,
            variation_idx=variation_idx,
            final_score=final_score,
            total_cost=total_cost,
            n_steps=len(steps),
            exit_status=exit_status,
            steps=steps,
            meta={
                "source_path": source_path,
                "n_api_calls": n_api_calls,
                "config": info.get("config", {}),
                "benchmark": benchmark,
                "norm_config": norm_config,
                "category": category,  # For AEC: HLE task category
            },
        )

    def _detect_benchmark(self, info: dict[str, Any]) -> str:
        """Detect which benchmark a trajectory belongs to.

        Detection priority:
        1. Explicit info.benchmark field (most reliable)
        2. Presence of benchmark_data.judge_result (indicates HLE)
        3. Presence of task_name + variation_idx (indicates SW)
        4. Default to scienceworld for backward compatibility

        Args:
            info: Trajectory info dict.

        Returns:
            Benchmark identifier string.
        """
        # Explicit benchmark field
        if "benchmark" in info:
            return info["benchmark"]

        # HLE-specific: presence of benchmark_data with judge_result
        benchmark_data = info.get("benchmark_data", {})
        if benchmark_data and "judge_result" in benchmark_data:
            return "hle"

        # SW-specific: has task_name and variation_idx
        if "task_name" in info and "variation_idx" in info:
            return "scienceworld"

        # Default for backward compatibility
        return DEFAULT_BENCHMARK

    def _infer_final_score(self, info: dict[str, Any], benchmark: str = "") -> float:
        """Infer final score in benchmark's native scale when missing.

        ScienceWorld trajectories record `info.final_score`. HLE stores
        evaluation results in `benchmark_data.judge_result.correct`.

        Returns raw scores in each benchmark's native scale:
        - ScienceWorld: -100 to 100
        - HLE: 0 or 1 (binary)

        Normalization to [0, 1] happens in to_training_samples().

        Args:
            info: Trajectory info dict.
            benchmark: Detected benchmark type.

        Returns:
            Inferred score in benchmark's native scale.
        """
        # HLE benchmark: return 0 or 1 (binary)
        if benchmark == "hle":
            benchmark_data = info.get("benchmark_data") or {}
            judge_result = benchmark_data.get("judge_result") or {}
            if isinstance(judge_result, dict):
                correct = judge_result.get("correct")
                if isinstance(correct, bool):
                    return 1.0 if correct else 0.0
            return 0.0

        # Generic boolean correctness field (future-proof)
        correct = info.get("correct")
        if isinstance(correct, bool):
            return 100.0 if correct else 0.0

        # Default for ScienceWorld or unknown
        return 0.0

    def _parse_steps(self, messages: list[dict]) -> list[StepFeature]:
        """Parse all steps from message list.

        A step consists of:
        - History: all messages before the assistant message
        - Action: the assistant message
        - Observation: the user message after (if any)

        Args:
            messages: List of message dicts.

        Returns:
            List of StepFeature objects.
        """
        steps = []
        step_idx = 0

        # Find system message start (if including)
        history_start = 0
        if not self.config.include_system_prompt:
            for i, msg in enumerate(messages):
                if msg.get("role") != "system":
                    history_start = i
                    break

        i = 0
        while i < len(messages):
            msg = messages[i]

            if msg.get("role") == "assistant":
                # This is an action step
                history_msgs = messages[history_start:i]
                history_text, history_tokens = self._build_history_text(history_msgs)
                action_text = msg.get("content", "")
                action_tokens = self._estimate_tokens(action_text)

                # Extract model info from response
                model_name, step_cost, prompt_tokens, completion_tokens = (
                    self._extract_step_info(msg)
                )

                # Get observation (next user message if exists)
                observation_text = ""
                if i + 1 < len(messages) and messages[i + 1].get("role") == "user":
                    observation_text = messages[i + 1].get("content", "")

                # Extract propensity if recorded
                propensity = self._extract_propensity(msg)

                # Get model index
                model_idx = self._get_model_idx(model_name)

                steps.append(
                    StepFeature(
                        step_idx=step_idx,
                        history_text=history_text,
                        history_tokens=history_tokens,
                        history_messages=history_msgs,
                        action_text=action_text,
                        action_tokens=action_tokens,
                        model_name=model_name,
                        model_idx=model_idx,
                        step_cost=step_cost,
                        observation_text=observation_text,
                        propensity=propensity,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
                )

                step_idx += 1

            i += 1

        return steps

    def _build_history_text(
        self,
        messages: list[dict],
    ) -> tuple[str, int]:
        """Build history text from messages.

        Args:
            messages: List of message dicts.

        Returns:
            Tuple of (history_text, estimated_tokens).
        """
        parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            parts.append(f"[{role}]\n{content}")

        text = "\n\n".join(parts)

        # Truncate if needed
        if len(text) > self.config.max_history_chars:
            text = text[-self.config.max_history_chars :]

        tokens = self._estimate_tokens(text)
        return text, tokens

    def _estimate_tokens(self, text: str) -> int:
        """Estimate number of tokens in text.

        Args:
            text: Text string.

        Returns:
            Estimated token count.
        """
        return int(len(text) / self.config.chars_per_token)

    def _extract_step_info(
        self,
        message: dict,
    ) -> tuple[str, float, int, int]:
        """Extract model name and cost from assistant message.

        Args:
            message: Assistant message dict.

        Returns:
            Tuple of (model_name, step_cost, prompt_tokens, completion_tokens).
        """
        extra = message.get("extra", {})
        response = extra.get("response", {})

        # Extract model name - prioritize router's selection over provider's response
        # Router stores model_name at message top level or in router_info
        model_name = message.get("model_name", "")
        if not model_name:
            router_info = message.get("router_info", {}) or extra.get("router_info", {})
            model_name = router_info.get("selected_model", "")
        if not model_name:
            # Fallback to provider's response (less reliable for routing)
            model_name = response.get("model", "unknown")

        # Extract cost from usage
        usage = response.get("usage", {})
        step_cost = usage.get("cost", 0.0)
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        return model_name, step_cost, prompt_tokens, completion_tokens

    def _extract_propensity(self, message: dict) -> float | None:
        """Extract propensity (selection probability) from message.

        Args:
            message: Assistant message dict.

        Returns:
            Propensity value or None if not recorded.
        """
        # Check message top level first (where agent stores response fields)
        if "propensity" in message:
            return message["propensity"]

        router_info = message.get("router_info", {})
        if "propensity" in router_info:
            return router_info["propensity"]

        # Fallback to extra (legacy format)
        extra = message.get("extra", {})
        if "propensity" in extra:
            return extra["propensity"]

        extra_router_info = extra.get("router_info", {})
        if "propensity" in extra_router_info:
            return extra_router_info["propensity"]

        return None

    def _get_model_idx(self, model_name: str) -> int:
        """Get model index from name.

        Args:
            model_name: Model name string.

        Returns:
            Model index or -1 if not found.
        """
        # Try exact match first
        if model_name in self._name_to_idx:
            return self._name_to_idx[model_name]

        # Try lowercase
        lower_name = model_name.lower()
        if lower_name in self._name_to_idx:
            return self._name_to_idx[lower_name]

        # Try without provider prefix
        if "/" in model_name:
            short_name = model_name.split("/")[-1]
            if short_name in self._name_to_idx:
                return self._name_to_idx[short_name]
            if short_name.lower() in self._name_to_idx:
                return self._name_to_idx[short_name.lower()]

        return -1


def find_trajectory_files(
    directory: Path | str,
    pattern: str = "**/*.traj.json",
) -> list[Path]:
    """Find all trajectory files in a directory.

    Args:
        directory: Directory to search.
        pattern: Glob pattern for matching files.

    Returns:
        List of trajectory file paths.
    """
    directory = Path(directory)
    return sorted(directory.glob(pattern))
