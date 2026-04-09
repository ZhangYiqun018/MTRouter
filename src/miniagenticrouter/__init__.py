"""
Mini Agentic Router - Multi-turn benchmark evaluation framework with model routing support.

This project is built on top of mini-swe-agent (https://github.com/SWE-agent/mini-swe-agent).

This file provides:
- Path settings for global config file & relative directories
- Version numbering
- Protocols for the core components.
  By the magic of protocols & duck typing, you can pretty much ignore them,
  unless you want the static type checking.
"""

__version__ = "0.1.0b1"

import os
from pathlib import Path
from typing import Any, Protocol

import dotenv
from platformdirs import user_config_dir
from rich.console import Console

from miniagenticrouter.utils.log import logger

package_dir = Path(__file__).resolve().parent

global_config_dir = Path(os.getenv("MAR_GLOBAL_CONFIG_DIR") or user_config_dir("mini-agentic-router"))
global_config_dir.mkdir(parents=True, exist_ok=True)
global_config_file = Path(global_config_dir) / ".env"

if not os.getenv("MAR_SILENT_STARTUP"):
    Console().print(
        f"👋 This is [bold green]mini-agentic-router[/bold green] version [bold green]{__version__}[/bold green].\n"
        f"Loading global config from [bold green]'{global_config_file}'[/bold green]"
    )
dotenv.load_dotenv(dotenv_path=global_config_file)


# === Custom model and provider configuration paths ===
# These are loaded lazily by the registry classes to avoid circular imports
_models_config_dir = package_dir / "config" / "models"
_custom_models_paths = [
    _models_config_dir / "custom_models.yaml",
    global_config_dir / "custom_models.yaml",
]
_providers_paths = [
    _models_config_dir / "providers.yaml",
    global_config_dir / "providers.yaml",
]


# === Protocols ===
# You can ignore them unless you want static type checking.


class Model(Protocol):
    """Protocol for language models."""

    config: Any
    cost: float
    n_calls: int

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict: ...

    def get_template_vars(self) -> dict[str, Any]: ...


class Environment(Protocol):
    """Protocol for execution environments."""

    config: Any

    def execute(self, command: str, cwd: str = "") -> dict[str, str]: ...

    def get_template_vars(self) -> dict[str, Any]: ...


class Agent(Protocol):
    """Protocol for agents."""

    model: Model
    env: Environment
    messages: list[dict[str, str]]
    config: Any

    def run(self, task: str, **kwargs) -> tuple[str, str]: ...


__all__ = [
    "Agent",
    "Model",
    "Environment",
    "package_dir",
    "__version__",
    "global_config_file",
    "global_config_dir",
    "logger",
]
