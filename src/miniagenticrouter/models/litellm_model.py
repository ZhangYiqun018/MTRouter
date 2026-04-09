import json
import logging
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import litellm
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from miniagenticrouter.models import GLOBAL_MODEL_STATS
from miniagenticrouter.models.utils.cache_control import set_cache_control

logger = logging.getLogger("litellm_model")

# Lazy initialization flag for model registry
_registry_initialized = False


def _ensure_registry_initialized():
    """Initialize model registry lazily to avoid circular imports."""
    global _registry_initialized
    if not _registry_initialized:
        _registry_initialized = True
        from miniagenticrouter.models.registry import ModelRegistry
        ModelRegistry.get_instance()


def _get_global_stats():
    """Get GLOBAL_MODEL_STATS lazily to avoid circular imports."""
    from miniagenticrouter.models import GLOBAL_MODEL_STATS
    return GLOBAL_MODEL_STATS


@dataclass
class LitellmModelConfig:
    model_name: str
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    litellm_model_registry: Path | str | None = os.getenv("LITELLM_MODEL_REGISTRY_PATH")
    set_cache_control: Literal["default_end"] | None = None
    """Set explicit cache control markers, for example for Anthropic models"""
    cost_tracking: Literal["default", "ignore_errors"] = os.getenv("MAR_COST_TRACKING", "default")
    """Cost tracking mode for this model. Can be "default" or "ignore_errors" (ignore errors/missing cost info)"""


class LitellmModel:
    def __init__(self, *, config_class: Callable = LitellmModelConfig, **kwargs):
        # Ensure custom models are registered before use
        _ensure_registry_initialized()

        # Get model_kwargs from registry and merge with provided kwargs
        # Provided kwargs take precedence over registry config
        model_name = kwargs.get("model_name", "")
        if model_name:
            from miniagenticrouter.models.registry import ModelRegistry
            registry = ModelRegistry.get_instance()
            registry_kwargs = registry.get_model_kwargs(model_name)
            if registry_kwargs:
                # Merge: registry config as base, user-provided model_kwargs on top
                user_model_kwargs = kwargs.get("model_kwargs", {})
                merged_model_kwargs = {**registry_kwargs, **user_model_kwargs}
                kwargs["model_kwargs"] = merged_model_kwargs
                logger.debug(f"Applied registry config for {model_name}: {registry_kwargs}")

        self.config = config_class(**kwargs)
        self.cost = 0.0
        self.n_calls = 0
        if self.config.litellm_model_registry and Path(self.config.litellm_model_registry).is_file():
            litellm.utils.register_model(json.loads(Path(self.config.litellm_model_registry).read_text()))

    @retry(
        reraise=True,
        stop=stop_after_attempt(int(os.getenv("MAR_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(
            (
                litellm.exceptions.UnsupportedParamsError,
                litellm.exceptions.NotFoundError,
                litellm.exceptions.PermissionDeniedError,
                litellm.exceptions.ContextWindowExceededError,
                # litellm.exceptions.APIError,
                litellm.exceptions.AuthenticationError,
                KeyboardInterrupt,
            )
        ),
    )
    def _query(self, messages: list[dict[str, str]], **kwargs):
        final_kwargs = self.config.model_kwargs | kwargs

        # Proxy mode: ensure custom_llm_provider is set to prevent model name parsing
        # Without custom_llm_provider, LiteLLM SDK strips prefixes like "openai/" from model names
        # We need to keep the full model name (e.g., "openai/gpt-5") when sending to Proxy
        if os.getenv("MAR_PROXY_MODE"):
            # Extract provider from model name if not already set
            if "custom_llm_provider" not in final_kwargs and "/" in self.config.model_name:
                final_kwargs["custom_llm_provider"] = "openai"  # Use OpenAI handler for Proxy

        try:
            return litellm.completion(
                model=self.config.model_name, messages=messages, **final_kwargs
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mtr-extra config set KEY VALUE`."
            raise e

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        if self.config.set_cache_control:
            messages = set_cache_control(messages, mode=self.config.set_cache_control)
        # Strip trailing whitespace and filter empty assistant messages for Bedrock compatibility
        cleaned_messages = []
        for msg in messages:
            content = msg["content"].rstrip()
            # Skip empty assistant messages (Bedrock doesn't allow them)
            if msg["role"] == "assistant" and not content:
                continue
            cleaned_messages.append({"role": msg["role"], "content": content})
        response = self._query(cleaned_messages, **kwargs)
        cost = self._calculate_cost(response)
        self.n_calls += 1
        self.cost += cost
        _get_global_stats().add(cost)
        return {
            "content": response.choices[0].message.content or "",  # type: ignore
            "extra": {
                "response": response.model_dump(),
            },
        }

    def _calculate_cost(self, response) -> float:
        """Calculate cost for a response with fallback to ModelRegistry.

        Resolution order:
        1. Try litellm's cost calculator
        2. If that fails or returns 0, use ModelRegistry's pricing info
        3. If both fail, either return 0 (ignore_errors) or raise RuntimeError

        Args:
            response: The litellm completion response.

        Returns:
            Calculated cost in USD.

        Raises:
            RuntimeError: If cost calculation fails and cost_tracking != 'ignore_errors'.
        """
        model_name = self.config.model_name
        cost = 0.0
        error_msg = None

        # Step 1: Try litellm's cost calculator
        try:
            cost = litellm.cost_calculator.completion_cost(response, model=model_name)
            if cost > 0.0:
                return cost
            error_msg = f"litellm returned cost={cost}"
        except Exception as e:
            error_msg = str(e)

        # Step 2: Fallback to ModelRegistry's pricing
        try:
            from miniagenticrouter.models.registry import ModelRegistry
            registry = ModelRegistry.get_instance()
            model_info = registry.get(model_name)

            if model_info and response.usage:
                prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0

                if prompt_tokens > 0 or completion_tokens > 0:
                    cost = (
                        prompt_tokens * model_info.input_cost_per_token +
                        completion_tokens * model_info.output_cost_per_token
                    )
                    if cost > 0.0:
                        logger.debug(
                            f"Used ModelRegistry fallback for {model_name}: "
                            f"cost=${cost:.6f} ({prompt_tokens} prompt + {completion_tokens} completion tokens)"
                        )
                        return cost
                    error_msg = f"ModelRegistry cost={cost} (tokens: {prompt_tokens}/{completion_tokens})"
                else:
                    error_msg = f"No token usage in response (prompt={prompt_tokens}, completion={completion_tokens})"
            elif not model_info:
                error_msg = f"Model {model_name} not found in ModelRegistry"
            else:
                error_msg = "No usage info in response"
        except Exception as e:
            error_msg = f"ModelRegistry fallback failed: {e}"

        # Step 3: Handle failure
        if self.config.cost_tracking == "ignore_errors":
            logger.warning(f"Cost calculation failed for {model_name}: {error_msg}, using cost=0.0")
            return 0.0

        msg = (
            f"Error calculating cost for model {model_name}: {error_msg}. "
            "You can ignore this issue from your config file with cost_tracking: 'ignore_errors' or "
            "globally with export MAR_COST_TRACKING='ignore_errors'. "
            "Alternatively check the 'Cost tracking' section in the documentation at "
            "https://klieret.short.gy/mini-local-models. "
            "Still stuck? Please open a github issue at https://github.com/SWE-agent/mini-swe-agent/issues/new/choose!"
        )
        logger.critical(msg)
        raise RuntimeError(msg)

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}
