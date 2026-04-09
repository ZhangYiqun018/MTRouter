"""Provider registry for API configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from miniagenticrouter.utils.log import logger


@dataclass
class ProviderConfig:
    """Provider configuration."""

    name: str
    api_base: str
    api_key_env: str
    openai_compatible: bool = False
    extra_headers: dict[str, str] = field(default_factory=dict)


class ProviderRegistry:
    """Provider registry - manages API provider configurations.

    This is a singleton class that stores provider configurations
    including API base URLs and authentication settings.

    Built-in providers:
        - openai: api.openai.com
        - anthropic: api.anthropic.com

    Usage:
        registry = ProviderRegistry.get_instance()
        registry.load_from_yaml(Path("providers.yaml"))

        # Get provider config
        provider = registry.get("openai")
        api_key = registry.get_api_key("openai")
    """

    _instance: ProviderRegistry | None = None
    _initialized: bool = False

    # Built-in providers
    BUILTIN_PROVIDERS = {
        "openai": ProviderConfig(
            name="openai",
            api_base="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
        ),
        "anthropic": ProviderConfig(
            name="anthropic",
            api_base="https://api.anthropic.com",
            api_key_env="ANTHROPIC_API_KEY",
        ),
        "azure": ProviderConfig(
            name="azure",
            api_base="",  # User must configure
            api_key_env="AZURE_API_KEY",
        ),
        "ollama": ProviderConfig(
            name="ollama",
            api_base="http://localhost:11434",
            api_key_env="",  # No API key needed
        ),
    }

    def __init__(self):
        self._providers: dict[str, ProviderConfig] = self.BUILTIN_PROVIDERS.copy()

    @classmethod
    def get_instance(cls) -> ProviderRegistry:
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
        from miniagenticrouter import _providers_paths

        for path in _providers_paths:
            if path.exists():
                self.load_from_yaml(path)

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None
        cls._initialized = False

    def register(self, config: ProviderConfig) -> None:
        """Register a provider configuration."""
        self._providers[config.name] = config
        logger.debug(f"Registered provider: {config.name}")

    def get(self, name: str) -> ProviderConfig | None:
        """Get provider configuration by name."""
        return self._providers.get(name)

    def get_api_key(self, provider_name: str) -> str | None:
        """Get API key for a provider from environment variables."""
        provider = self.get(provider_name)
        if provider and provider.api_key_env:
            return os.getenv(provider.api_key_env)
        return None

    def get_api_base(self, provider_name: str) -> str | None:
        """Get API base URL for a provider."""
        provider = self.get(provider_name)
        if provider:
            return provider.api_base
        return None

    def load_from_yaml(self, path: Path) -> None:
        """Load provider configurations from a YAML file.

        YAML format:
            providers:
              my_proxy:
                api_base: "https://your-proxy.com/v1"
                api_key_env: "MY_PROXY_API_KEY"
                openai_compatible: true
                extra_headers:
                  X-Custom-Header: "value"
        """
        if not path.exists():
            logger.warning(f"Provider config file not found: {path}")
            return

        try:
            config = yaml.safe_load(path.read_text())
        except Exception as e:
            logger.error(f"Failed to load provider config from {path}: {e}")
            return

        if not config:
            return

        providers = config.get("providers") or {}
        for name, info in providers.items():
            if info is None:
                continue

            provider_config = ProviderConfig(
                name=name,
                api_base=info.get("api_base", ""),
                api_key_env=info.get("api_key_env", f"{name.upper()}_API_KEY"),
                openai_compatible=info.get("openai_compatible", False),
                extra_headers=info.get("extra_headers", {}),
            )
            self.register(provider_config)

        if providers:
            logger.info(f"Loaded {len(providers)} providers from {path}")

    def list_providers(self) -> list[str]:
        """List all registered provider names."""
        return list(self._providers.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._providers

    def __len__(self) -> int:
        return len(self._providers)
