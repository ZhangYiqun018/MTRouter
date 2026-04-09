"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path | str) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_default_config_path() -> Path:
    """Get the default path to data_split.yaml."""
    # Traverse up to find the config directory
    current = Path(__file__).resolve()
    for _ in range(5):  # Max 5 levels up
        current = current.parent
        config_path = current / "config" / "research" / "data_split.yaml"
        if config_path.exists():
            return config_path

    # Fallback to relative path
    return Path("config/research/data_split.yaml")


def get_training_config_path() -> Path:
    """Get the default path to training.yaml."""
    current = Path(__file__).resolve()
    for _ in range(5):  # Max 5 levels up
        current = current.parent
        config_path = current / "config" / "research" / "training.yaml"
        if config_path.exists():
            return config_path

    # Fallback to relative path
    return Path("config/research/training.yaml")


def get_default_hle_config_path() -> Path:
    """Get the default path to hle_data_split.yaml."""
    current = Path(__file__).resolve()
    for _ in range(5):  # Max 5 levels up
        current = current.parent
        config_path = current / "config" / "research" / "hle_data_split.yaml"
        if config_path.exists():
            return config_path

    # Fallback to relative path
    return Path("config/research/hle_data_split.yaml")


def load_training_config() -> dict[str, Any]:
    """Load the training configuration.

    Returns:
        Training configuration dictionary.

    Raises:
        FileNotFoundError: If training.yaml is not found.
    """
    return load_config(get_training_config_path())


def get_model_pool(yaml_config: dict[str, Any]) -> list[str]:
    """Extract model_pool from training YAML config.

    This is the single source of truth for model names used in training,
    data collection, and inference.

    Args:
        yaml_config: Config dict loaded from training.yaml.

    Returns:
        List of model names.

    Raises:
        ValueError: If model_pool is missing, empty, or invalid.

    Example:
        >>> config = load_training_config()
        >>> model_names = get_model_pool(config)
        >>> print(model_names)
        ['openai/gpt-5', 'deepseek/deepseek-v3.2', 'minimax/minimax-m2']
    """
    model_pool = yaml_config.get("model_pool")
    if not model_pool:
        raise ValueError(
            "model_pool not found or empty in training.yaml. "
            "Please add a 'model_pool' section with at least one model name. "
            "Example:\n"
            "  model_pool:\n"
            '    - "openai/gpt-5"\n'
            '    - "deepseek/deepseek-v3.2"'
        )
    if not isinstance(model_pool, list):
        raise ValueError(
            f"model_pool must be a list, got {type(model_pool).__name__}"
        )
    if not all(isinstance(m, str) for m in model_pool):
        raise ValueError("All entries in model_pool must be strings")
    return model_pool
