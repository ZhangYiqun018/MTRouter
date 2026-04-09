"""Model registry for custom model configurations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import litellm
import yaml

from miniagenticrouter.utils.log import logger

# Suppress LiteLLM's "Provider List" warnings during model registration
litellm.suppress_debug_info = True


@dataclass
class ModelInfo:
    """Model metadata, cost information, and connection settings."""

    name: str
    display_name: str
    provider: str
    max_tokens: int
    max_input_tokens: int
    max_output_tokens: int
    input_cost_per_token: float
    output_cost_per_token: float
    supports_vision: bool = False
    supports_function_calling: bool = True
    # Connection settings
    api_base: str | None = None
    api_key_env: str | None = None  # Environment variable name for API key
    model_kwargs: dict[str, Any] = field(default_factory=dict)  # Additional kwargs for litellm
    extra: dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """Model registry - manages all custom model configurations.

    This is a singleton class that registers custom models to litellm
    for cost tracking and model information lookup.

    Usage:
        registry = ModelRegistry.get_instance()
        registry.load_from_yaml(Path("custom_models.yaml"))

        # Or register programmatically
        registry.register(ModelInfo(
            name="openai/claude-sonnet-4-5-20250929",
            display_name="Claude Sonnet 4.5",
            provider="openai",
            ...
        ))
    """

    _instance: ModelRegistry | None = None
    _initialized: bool = False

    def __init__(self):
        self._models: dict[str, ModelInfo] = {}

    @classmethod
    def get_instance(cls) -> ModelRegistry:
        """Get the singleton instance with auto-loading of config files."""
        if cls._instance is None:
            cls._instance = cls()
        if not cls._initialized:
            cls._initialized = True
            cls._instance._load_default_configs()
        return cls._instance

    def _load_default_configs(self) -> None:
        """Load default configuration files."""
        # Import here to avoid circular imports
        from miniagenticrouter import _custom_models_paths

        for path in _custom_models_paths:
            if path.exists():
                self.load_from_yaml(path)

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None
        cls._initialized = False

    def register(self, model_info: ModelInfo) -> None:
        """Register a model to the registry and litellm."""
        self._models[model_info.name] = model_info

        # Register to litellm for cost tracking
        litellm_config = {
            model_info.name: {
                "max_tokens": model_info.max_tokens,
                "max_input_tokens": model_info.max_input_tokens,
                "max_output_tokens": model_info.max_output_tokens,
                "input_cost_per_token": model_info.input_cost_per_token,
                "output_cost_per_token": model_info.output_cost_per_token,
                "litellm_provider": model_info.provider,
            }
        }
        litellm.register_model(litellm_config)
        logger.debug(f"Registered model: {model_info.name}")

    def load_from_yaml(self, path: Path) -> None:
        """Load model configurations from a YAML file.

        YAML format:
            models:
              openai/claude-sonnet-4-5-20250929:
                display_name: "Claude Sonnet 4.5 (OpenAI Compatible)"
                provider: openai
                max_tokens: 8192
                max_input_tokens: 200000
                max_output_tokens: 8192
                input_cost_per_million: 3.0
                output_cost_per_million: 15.0
        """
        if not path.exists():
            logger.warning(f"Model config file not found: {path}")
            return

        try:
            config = yaml.safe_load(path.read_text())
        except Exception as e:
            logger.error(f"Failed to load model config from {path}: {e}")
            return

        if not config:
            return

        models = config.get("models") or {}
        for name, info in models.items():
            if info is None:
                continue

            # Convert cost per million to cost per token
            input_cost = info.get("input_cost_per_million", 0) / 1_000_000
            output_cost = info.get("output_cost_per_million", 0) / 1_000_000

            model_info = ModelInfo(
                name=name,
                display_name=info.get("display_name", name),
                provider=info.get("provider", "openai"),
                max_tokens=info.get("max_tokens", 4096),
                max_input_tokens=info.get("max_input_tokens", 128000),
                max_output_tokens=info.get("max_output_tokens", 4096),
                input_cost_per_token=input_cost,
                output_cost_per_token=output_cost,
                supports_vision=info.get("supports_vision", False),
                supports_function_calling=info.get("supports_function_calling", True),
                # Connection settings
                api_base=info.get("api_base"),
                api_key_env=info.get("api_key_env"),
                model_kwargs=info.get("model_kwargs", {}),
                extra=info.get("extra", {}),
            )
            self.register(model_info)

        if models:
            logger.info(f"Loaded {len(models)} models from {path}")

    def get(self, name: str) -> ModelInfo | None:
        """Get model info by name."""
        return self._models.get(name)

    # Standard environment variable names for each provider
    _PROVIDER_ENV_VARS = {
        "openai": {
            "api_base": ["OPENAI_API_BASE", "OPENAI_BASE_URL"],
            "api_key": ["OPENAI_API_KEY"],
        },
        "anthropic": {
            "api_base": ["ANTHROPIC_API_BASE", "ANTHROPIC_BASE_URL"],
            "api_key": ["ANTHROPIC_API_KEY"],
        },
        "azure": {
            "api_base": ["AZURE_API_BASE", "AZURE_OPENAI_ENDPOINT"],
            "api_key": ["AZURE_API_KEY", "AZURE_OPENAI_API_KEY"],
        },
        "gemini": {
            "api_base": ["GEMINI_API_BASE"],
            "api_key": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        },
        "bedrock": {
            "api_base": ["BEDROCK_API_BASE"],
            "api_key": [],  # Bedrock uses AWS credentials
        },
    }

    def get_model_kwargs(self, name: str) -> dict[str, Any]:
        """Get model_kwargs for a model, including api_base and api_key.

        Resolution order:
        1. api_base: config value -> provider-specific env vars -> None
        2. api_key: config-specified env var -> provider-default env vars -> None

        Returns an empty dict if the model is not found.
        """
        import os

        model_info = self.get(name)
        if model_info is None:
            return {}

        kwargs = dict(model_info.model_kwargs)  # Copy to avoid mutation
        provider = model_info.provider.lower()
        provider_env = self._PROVIDER_ENV_VARS.get(provider, {})

        # Resolve api_base: config -> env vars
        if model_info.api_base:
            kwargs["api_base"] = model_info.api_base
        else:
            # Try provider-specific environment variables
            for env_var in provider_env.get("api_base", []):
                api_base = os.getenv(env_var)
                if api_base:
                    kwargs["api_base"] = api_base
                    logger.debug(f"Using api_base from {env_var} for model {name}")
                    break

        # Resolve api_key: specified env var -> provider default env vars
        api_key = None
        if model_info.api_key_env:
            api_key = os.getenv(model_info.api_key_env)
            if not api_key:
                logger.warning(f"Environment variable {model_info.api_key_env} not set for model {name}")
        else:
            # Try provider-specific default environment variables
            for env_var in provider_env.get("api_key", []):
                api_key = os.getenv(env_var)
                if api_key:
                    logger.debug(f"Using api_key from {env_var} for model {name}")
                    break

        if api_key:
            kwargs["api_key"] = api_key

        return kwargs

    def list_models(self) -> list[str]:
        """List all registered model names."""
        return list(self._models.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._models

    def __len__(self) -> int:
        return len(self._models)
