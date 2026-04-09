"""Error feature detection for trajectory observations.

This module provides an extensible error detection system for identifying
errors in agent trajectory observations. It supports:
- YAML-based rule configuration
- Per-benchmark rule sets (HLE, ScienceWorld)
- Runtime rule addition for customization
- Severity-based penalty calculation (for future use)

Example:
    >>> from miniagenticrouter.research.training.error_detector import (
    ...     ErrorDetector, ErrorDetectorConfig
    ... )
    >>> detector = ErrorDetector()
    >>> result = detector.detect("NameError: name 'x' is not defined", "hle")
    >>> print(result.has_error)  # True
    >>> print(result.errors)     # ["python_name_error"]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ErrorRule:
    """A single error detection rule.

    Attributes:
        name: Unique identifier for this rule (e.g., "python_name_error").
        patterns: List of string patterns to match (case-sensitive).
        severity: Error severity level ("high", "medium", "low").
        description: Human-readable description of the error.
    """

    name: str
    patterns: list[str]
    severity: str = "medium"
    description: str = ""


@dataclass
class ErrorResult:
    """Result of error detection on a single observation.

    Attributes:
        errors: List of error rule names that matched.
        penalty: Total penalty value based on severity weights.
        has_error: Whether any error was detected.
    """

    errors: list[str] = field(default_factory=list)
    penalty: float = 0.0

    @property
    def has_error(self) -> bool:
        """Whether any error was detected."""
        return len(self.errors) > 0


@dataclass
class ErrorDetectorConfig:
    """Configuration for ErrorDetector.

    Attributes:
        enabled: Whether error detection is enabled.
        severity_weights: Mapping from severity level to penalty weight.
        rules_path: Path to custom rules YAML file (None uses default).
    """

    enabled: bool = True
    severity_weights: dict[str, float] = field(
        default_factory=lambda: {"high": 0.3, "medium": 0.1, "low": 0.05}
    )
    rules_path: Path | None = None


@dataclass
class AECConfig:
    """Configuration for Annealed Error Cost (AEC) penalty.

    AEC applies time-dependent penalties: lighter early, heavier late.
    Penalty formula: pen_t = max(base * m(severity) * w(progress), min_penalty)
    where base = 1/N, progress = t/N, w() is the warmup function.

    Attributes:
        enabled: Whether AEC is enabled.
        expected_steps: Mapping benchmark -> task_category -> N (expected steps).
        severity_coefficients: Mapping severity -> penalty coefficient m(k).
        p0: Progress threshold where warmup ramp starts.
        p1: Progress threshold where warmup ramp ends.
        w_min: Minimum warmup weight (used for progress <= p0).
        w_max: Maximum warmup weight (used for progress >= p1).
        min_penalty_factor: Minimum penalty as fraction of base.
    """

    enabled: bool = True
    expected_steps: dict[str, dict[str, int]] = field(default_factory=dict)
    severity_coefficients: dict[str, float] = field(
        default_factory=lambda: {"high": 2.0, "medium": 1.0, "low": 0.25}
    )
    p0: float = 0.3
    p1: float = 0.7
    w_min: float = 0.3
    w_max: float = 1.0
    min_penalty_factor: float = 0.2

    def get_expected_steps(self, benchmark: str, task_category: str) -> int:
        """Get expected steps N for a task category.

        Args:
            benchmark: Benchmark name ("hle" or "scienceworld").
            task_category: Task category (e.g., "Math", "boil").

        Returns:
            Expected steps N, falling back to default if category not found.
        """
        benchmark_steps = self.expected_steps.get(benchmark, {})
        return benchmark_steps.get(
            task_category,
            benchmark_steps.get("default", 20),
        )


# =============================================================================
# Default Rules Path
# =============================================================================


def _get_default_rules_path() -> Path:
    """Get path to the default error_rules.yaml file."""
    # Navigate from this file to config/research/error_rules.yaml
    current_dir = Path(__file__).parent
    config_path = current_dir.parent.parent / "config" / "research" / "error_rules.yaml"
    return config_path


# =============================================================================
# ErrorDetector Class
# =============================================================================


class ErrorDetector:
    """Extensible error feature detector for trajectory observations.

    This class detects errors in observation text using pattern matching.
    Rules are loaded from YAML configuration and can be extended at runtime.

    Supports rule hierarchy: when a child rule matches, the parent rule is
    excluded from results to avoid double-counting (e.g., Traceback + NameError).

    Example:
        >>> detector = ErrorDetector()
        >>> result = detector.detect("Traceback (most recent call last)", "hle")
        >>> result.has_error
        True
        >>> "python_traceback" in result.errors
        True

        >>> # Add custom rule at runtime
        >>> from miniagenticrouter.research.training.error_detector import ErrorRule
        >>> detector.add_rule("hle", ErrorRule(
        ...     name="custom_error",
        ...     patterns=["MyCustomError:"],
        ...     severity="high"
        ... ))
    """

    def __init__(self, config: ErrorDetectorConfig | None = None):
        """Initialize ErrorDetector.

        Args:
            config: Configuration for the detector. Uses defaults if None.
        """
        self.config = config or ErrorDetectorConfig()
        self._rules: dict[str, list[ErrorRule]] = {}
        self._hierarchy: dict[str, list[str]] = {}  # parent -> [children]
        self._child_to_parent: dict[str, str] = {}  # child -> parent
        self._aec_config: AECConfig | None = None
        self._load_rules()

    def _load_rules(self) -> None:
        """Load rules from YAML configuration file."""
        rules_path = self.config.rules_path or _get_default_rules_path()

        if not rules_path.exists():
            # No config file, use empty rules
            self._rules = {"hle": [], "scienceworld": []}
            return

        with open(rules_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Load defaults from config if present
        defaults = data.get("defaults", {})
        if "enabled" in defaults:
            self.config.enabled = defaults["enabled"]
        if "severity_weights" in defaults:
            self.config.severity_weights.update(defaults["severity_weights"])

        # Load rule hierarchy
        hierarchy = data.get("rule_hierarchy", {})
        for parent, children in hierarchy.items():
            self._hierarchy[parent] = children
            for child in children:
                self._child_to_parent[child] = parent

        # Load rules for each benchmark
        for benchmark in ["hle", "scienceworld"]:
            self._rules[benchmark] = []
            benchmark_rules = data.get(benchmark, {})

            for rule_name, rule_data in benchmark_rules.items():
                if rule_name == "custom_rules":
                    # Skip the custom_rules placeholder
                    continue
                if not isinstance(rule_data, dict):
                    continue

                patterns = rule_data.get("patterns", [])
                if not patterns:
                    continue

                rule = ErrorRule(
                    name=rule_name,
                    patterns=patterns,
                    severity=rule_data.get("severity", "medium"),
                    description=rule_data.get("description", ""),
                )
                self._rules[benchmark].append(rule)

        # Load AEC configuration
        aec_data = data.get("aec", {})
        if aec_data:
            warmup = aec_data.get("warmup", {})
            self._aec_config = AECConfig(
                enabled=aec_data.get("enabled", True),
                expected_steps=aec_data.get("expected_steps", {}),
                severity_coefficients=aec_data.get(
                    "severity_coefficients",
                    {"high": 2.0, "medium": 1.0, "low": 0.25},
                ),
                p0=warmup.get("p0", 0.3),
                p1=warmup.get("p1", 0.7),
                w_min=warmup.get("w_min", 0.3),
                w_max=warmup.get("w_max", 1.0),
                min_penalty_factor=aec_data.get("min_penalty_factor", 0.2),
            )

    def detect(self, content: str, benchmark: str = "hle") -> ErrorResult:
        """Detect errors in observation content.

        Applies rule hierarchy: when a child rule matches, the parent rule
        is excluded from results to avoid double-counting.

        Args:
            content: Text content to check for errors (typically observation).
            benchmark: Benchmark type ("hle" or "scienceworld").

        Returns:
            ErrorResult with matched errors and penalty.
        """
        if not self.config.enabled or not content:
            return ErrorResult()

        rules = self._rules.get(benchmark, [])
        matched_rules: list[str] = []
        rule_severities: dict[str, str] = {}

        # First pass: collect all matching rules
        for rule in rules:
            for pattern in rule.patterns:
                if pattern in content:
                    matched_rules.append(rule.name)
                    rule_severities[rule.name] = rule.severity
                    break  # Only count each rule once per observation

        # Second pass: apply hierarchy (exclude parents when children match)
        errors: list[str] = []
        for rule_name in matched_rules:
            # Check if this rule is a parent and any of its children matched
            if rule_name in self._hierarchy:
                children = self._hierarchy[rule_name]
                if any(child in matched_rules for child in children):
                    # Skip parent, child will be counted instead
                    continue
            errors.append(rule_name)

        # Calculate penalty only for non-excluded rules
        total_penalty = 0.0
        for rule_name in errors:
            severity = rule_severities.get(rule_name, "medium")
            total_penalty += self.config.severity_weights.get(severity, 0.1)

        return ErrorResult(errors=errors, penalty=total_penalty)

    def detect_messages(
        self, messages: list[dict[str, Any]], benchmark: str = "hle"
    ) -> ErrorResult:
        """Detect errors in a list of messages.

        Checks all user messages in the list for errors.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            benchmark: Benchmark type ("hle" or "scienceworld").

        Returns:
            ErrorResult with all matched errors and total penalty.
        """
        if not self.config.enabled:
            return ErrorResult()

        all_errors: list[str] = []
        total_penalty = 0.0

        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                result = self.detect(content, benchmark)
                all_errors.extend(result.errors)
                total_penalty += result.penalty

        # Deduplicate errors while preserving order
        seen = set()
        unique_errors = []
        for err in all_errors:
            if err not in seen:
                seen.add(err)
                unique_errors.append(err)

        return ErrorResult(errors=unique_errors, penalty=total_penalty)

    def add_rule(self, benchmark: str, rule: ErrorRule) -> None:
        """Add a custom rule at runtime.

        Args:
            benchmark: Benchmark to add the rule to ("hle" or "scienceworld").
            rule: The ErrorRule to add.
        """
        if benchmark not in self._rules:
            self._rules[benchmark] = []
        self._rules[benchmark].append(rule)

    def get_rules(self, benchmark: str) -> list[ErrorRule]:
        """Get all rules for a benchmark.

        Args:
            benchmark: Benchmark type ("hle" or "scienceworld").

        Returns:
            List of ErrorRule objects.
        """
        return list(self._rules.get(benchmark, []))

    def get_rule_names(self, benchmark: str) -> list[str]:
        """Get all rule names for a benchmark.

        Args:
            benchmark: Benchmark type ("hle" or "scienceworld").

        Returns:
            List of rule name strings.
        """
        return [rule.name for rule in self._rules.get(benchmark, [])]

    def __repr__(self) -> str:
        """String representation of the detector."""
        hle_count = len(self._rules.get("hle", []))
        sw_count = len(self._rules.get("scienceworld", []))
        return f"ErrorDetector(hle_rules={hle_count}, sw_rules={sw_count})"

    def get_aec_config(self) -> AECConfig | None:
        """Get the AEC configuration loaded from YAML.

        Returns:
            AECConfig if AEC is configured, None otherwise.
        """
        return self._aec_config

    def get_severity_map(self, benchmark: str) -> dict[str, str]:
        """Get mapping from rule name to severity for a benchmark.

        Args:
            benchmark: Benchmark type ("hle" or "scienceworld").

        Returns:
            Dict mapping rule_name -> severity ("high", "medium", "low").
        """
        return {rule.name: rule.severity for rule in self._rules.get(benchmark, [])}
