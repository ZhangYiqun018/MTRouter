"""Collection mode strategies for data collection.

This module provides different strategies for collecting trajectory data:
- BaselineMode: Single model per trajectory
- RouletteMode: Random model selection with propensity recording
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from miniagenticrouter import Model


def _override_vllm_url_from_current_config(config: Any) -> None:
    """Override vllm_base_url in config with current training.yaml value.

    Checkpoint may contain old server addresses that are no longer valid.
    Runtime configuration (vllm_base_url) should come from current environment.
    """
    if config.history_encoder_config is not None:
        from miniagenticrouter.research.utils.config import load_training_config
        current_yaml = load_training_config()
        current_vllm_url = current_yaml.get("history_encoder", {}).get("vllm_base_url")
        if current_vllm_url:
            config.history_encoder_config.vllm_base_url = current_vllm_url


class CollectionMode(ABC):
    """Abstract base class for collection modes.

    Collection modes define how models are selected for each trajectory.

    Attributes:
        fallback_model_name: Model name to use when ContextWindowExceededError
            occurs. If None, the error is raised without fallback.
    """

    # Fallback model for ContextWindowExceededError (can be overridden in subclasses)
    fallback_model_name: str | None = None

    @abstractmethod
    def create_model_or_router(
        self,
        model_configs: list[dict[str, Any]],
    ) -> Model:
        """Create a model or router for data collection.

        Args:
            model_configs: List of model configuration dicts.

        Returns:
            Model or Router instance.
        """
        ...

    @abstractmethod
    def get_mode_name(self) -> str:
        """Get the name of this collection mode.

        Returns:
            Mode name string (used in output paths).
        """
        ...

    @abstractmethod
    def get_model_names(self, fallback_configs: list[dict[str, Any]]) -> list[str]:
        """Get the model names that will be used by this mode.

        Args:
            fallback_configs: Model configs from data_split (used by roulette/baseline modes).

        Returns:
            List of model names that this mode will use.
        """
        ...

    def get_output_subdir(self) -> str:
        """Get the output subdirectory name for this mode.

        Returns:
            Subdirectory name.
        """
        return self.get_mode_name()


class BaselineMode(CollectionMode):
    """Baseline collection mode using a single model.

    This mode creates a single model instance for each trajectory,
    useful for collecting baseline performance data for each model.

    Args:
        model_name: Name of the model to use (e.g., 'deepseek/deepseek-v3.2').

    Example:
        >>> mode = BaselineMode(model_name="deepseek/deepseek-v3.2")
        >>> model = mode.create_model_or_router(model_configs)
        >>> # Now run trajectories with this single model
    """

    def __init__(self, model_name: str):
        """Initialize BaselineMode.

        Args:
            model_name: Name of the model to use.
        """
        self.model_name = model_name

    # Models that bypass model_configs validation (e.g., auto-routing models)
    PASSTHROUGH_MODELS = {"openrouter/auto"}

    def create_model_or_router(
        self,
        model_configs: list[dict[str, Any]],
    ) -> Model:
        """Create a single model instance.

        Args:
            model_configs: List of model configuration dicts.

        Returns:
            Model instance.

        Raises:
            ValueError: If model_name not found in model_configs
                (unless in PASSTHROUGH_MODELS).
        """
        from miniagenticrouter.models import get_model

        # Check if model is in configs
        for config in model_configs:
            if config["model_name"] == self.model_name:
                return get_model(config=config)

        # Allow passthrough models without validation
        if self.model_name in self.PASSTHROUGH_MODELS:
            return get_model(config={"model_name": self.model_name})

        available = [c["model_name"] for c in model_configs]
        raise ValueError(
            f"Model '{self.model_name}' not found. Available: {available}"
        )

    def get_mode_name(self) -> str:
        """Get mode name including model name."""
        # Sanitize model name for directory (replace / with _)
        safe_name = self.model_name.replace("/", "_")
        return f"baseline_{safe_name}"

    def get_model_names(self, fallback_configs: list[dict[str, Any]]) -> list[str]:
        """Get model names (single model for baseline)."""
        return [self.model_name]

    def __repr__(self) -> str:
        return f"BaselineMode(model_name={self.model_name!r})"


class RouletteMode(CollectionMode):
    """Roulette collection mode with random model selection.

    This mode uses a roulette router to randomly select models,
    optionally recording the selection propensity for offline evaluation.

    Args:
        record_propensity: Whether to record selection probabilities.

    Example:
        >>> mode = RouletteMode(record_propensity=True)
        >>> router = mode.create_model_or_router(model_configs)
        >>> # Router will randomly select models and record propensities
    """

    def __init__(
        self,
        record_propensity: bool = True,
        fallback_model_name: str | None = None,
    ):
        """Initialize RouletteMode.

        Args:
            record_propensity: Whether to record selection probabilities.
            fallback_model_name: Model name for context window fallback.
        """
        self.record_propensity = record_propensity
        self.fallback_model_name = fallback_model_name

    def create_model_or_router(
        self,
        model_configs: list[dict[str, Any]],
    ) -> Model:
        """Create a roulette router.

        Args:
            model_configs: List of model configuration dicts.

        Returns:
            RouletteRouter or PropensityRouletteRouter instance.
        """
        if self.record_propensity:
            # Use propensity-tracking router
            from miniagenticrouter.research.collection.propensity_router import (
                PropensityRouletteRouter,
            )

            return PropensityRouletteRouter(
                model_kwargs=model_configs,
                fallback_model_name=self.fallback_model_name,
            )
        else:
            # Use standard roulette router
            from miniagenticrouter.routers import RouletteRouter

            return RouletteRouter(
                model_kwargs=model_configs,
                fallback_model_name=self.fallback_model_name,
            )

    def get_mode_name(self) -> str:
        """Get mode name."""
        if self.record_propensity:
            return "roulette_propensity"
        return "roulette"

    def get_model_names(self, fallback_configs: list[dict[str, Any]]) -> list[str]:
        """Get model names from data_split configs (roulette uses all available models)."""
        return [cfg.get("model_name", "") for cfg in fallback_configs]

    def __repr__(self) -> str:
        return f"RouletteMode(record_propensity={self.record_propensity})"


class InterleavingMode(CollectionMode):
    """Interleaving collection mode with round-robin model selection.

    This mode uses an interleaving router to cycle through models
    in a fixed sequence.

    Args:
        sequence: Optional custom sequence of model indices.
    """

    def __init__(
        self,
        sequence: list[int] | None = None,
        fallback_model_name: str | None = None,
    ):
        """Initialize InterleavingMode.

        Args:
            sequence: Optional sequence of model indices to follow.
            fallback_model_name: Model name for context window fallback.
        """
        self.sequence = sequence
        self.fallback_model_name = fallback_model_name

    def create_model_or_router(
        self,
        model_configs: list[dict[str, Any]],
    ) -> Model:
        """Create an interleaving router.

        Args:
            model_configs: List of model configuration dicts.

        Returns:
            InterleavingRouter instance.
        """
        from miniagenticrouter.routers import InterleavingRouter

        return InterleavingRouter(
            model_kwargs=model_configs,
            sequence=self.sequence,
            fallback_model_name=self.fallback_model_name,
        )

    def get_mode_name(self) -> str:
        """Get mode name."""
        return "interleaving"

    def get_model_names(self, fallback_configs: list[dict[str, Any]]) -> list[str]:
        """Get model names from data_split configs (interleaving uses all available models)."""
        return [cfg.get("model_name", "") for cfg in fallback_configs]

    def __repr__(self) -> str:
        return f"InterleavingMode(sequence={self.sequence})"


class LearnedMode(CollectionMode):
    """Learned collection mode using trained Q-function.

    This mode uses a trained Q-function to select models based on
    predicted Q-values for each conversation state.

    Args:
        checkpoint: Path to trained Q-function checkpoint.
        model_names: List of model names (must match training order).
        lambda_: Cost penalty coefficient for Q = score - lambda * cost.
            If None, uses the value from checkpoint.
        use_batching: Whether to enable dynamic batching for inference.
        batch_size: Maximum batch size (when use_batching=True).
        timeout: Maximum time to wait for batch to fill in seconds.

    Example:
        >>> mode = LearnedMode(
        ...     checkpoint="outputs/q_function/q_function_best.pt",
        ...     model_names=["openai/gpt-5", "deepseek/deepseek-v3.2"],
        ...     lambda_=1.0,
        ...     use_batching=True,
        ...     batch_size=8,
        ... )
        >>> router = mode.create_model_or_router(model_configs)
    """

    def __init__(
        self,
        checkpoint: Path | str,
        model_names: list[str],
        lambda_: float | None = None,
        use_batching: bool = False,
        batch_size: int = 8,
        timeout: float = 0.02,
        fallback_model_name: str | None = None,
    ):
        """Initialize LearnedMode.

        Args:
            checkpoint: Path to Q-function checkpoint file.
            model_names: Model names in training order.
            lambda_: Cost penalty coefficient. If None, uses checkpoint value.
            use_batching: Whether to enable dynamic batching.
            batch_size: Maximum batch size for batched inference.
            timeout: Maximum seconds to wait for batch to fill.
            fallback_model_name: Model name for context window fallback.
        """
        self.checkpoint = Path(checkpoint)
        self.model_names = model_names
        self.lambda_ = lambda_
        self.use_batching = use_batching
        self.batch_size = batch_size
        self.timeout = timeout
        self.fallback_model_name = fallback_model_name

        self._cached_q_func = None
        self._cached_device: str | None = None
        self._batched_q_func = None

        import threading

        self._q_func_lock = threading.Lock()

        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint}")

    def create_model_or_router(
        self,
        model_configs: list[dict[str, Any]],
    ) -> Model:
        """Create a LearnedRouter with loaded Q-function.

        Args:
            model_configs: List of model configuration dicts.

        Returns:
            LearnedRouter instance with trained Q-function.
        """
        import torch
        from miniagenticrouter.research.routers.learned import LearnedRouter
        from miniagenticrouter.research.training.q_network import NeuralQFunction

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load trained Q-function once and cache it to avoid repeated checkpoint
        # loads (which are expensive and can spam meta-tensor warnings).
        with self._q_func_lock:
            if self._cached_q_func is None or self._cached_device != device:
                # Load config from checkpoint (includes max_tokens, lambda_, etc.)
                ckpt_model_names, config, ckpt_lambda = NeuralQFunction.load_config_from_checkpoint(self.checkpoint)

                # Override vllm_base_url with current config (checkpoint may have stale URL)
                _override_vllm_url_from_current_config(config)

                if ckpt_model_names != self.model_names:
                    raise ValueError(
                        "Model names mismatch between checkpoint and runtime configuration.\n"
                        f"  checkpoint: {ckpt_model_names}\n"
                        f"  runtime   : {self.model_names}\n"
                        "These must match (including order) for correct routing/Q-values."
                    )

                # Use provided lambda_ or fallback to checkpoint value
                effective_lambda = self.lambda_ if self.lambda_ is not None else ckpt_lambda

                q_func = NeuralQFunction(
                    model_names=self.model_names,
                    config=config,
                    lambda_=effective_lambda,
                )
                q_func.load(self.checkpoint)
                # Override lambda_ if provided (load may have overwritten it)
                if self.lambda_ is not None:
                    q_func.set_lambda(self.lambda_)
                q_func.to(device)
                q_func.eval()
                self._cached_q_func = q_func
                self._cached_device = device

                # Create batched wrapper if enabled
                if self.use_batching:
                    from miniagenticrouter.research.training.batched_inference import (
                        BatchedQFunction,
                    )
                    self._batched_q_func = BatchedQFunction(
                        q_func=q_func,
                        batch_size=self.batch_size,
                        timeout=self.timeout,
                    )

            # Use batched or direct Q-function
            q_function = self._batched_q_func if self.use_batching else self._cached_q_func

        # Build model_configs directly from model_pool (self.model_names)
        # This ensures we use training.yaml's model_pool, not data_split's configs
        model_configs_from_pool = [{"model_name": name} for name in self.model_names]

        # Create LearnedRouter with model_pool models
        return LearnedRouter(
            model_kwargs=model_configs_from_pool,
            q_function=q_function,
            fallback_model_name=self.fallback_model_name,
        )

    def shutdown(self) -> None:
        """Shutdown batched inference resources.

        Call this method after data collection is complete to cleanly
        stop the background worker thread.
        """
        if self._batched_q_func is not None:
            self._batched_q_func.shutdown()
            self._batched_q_func = None

    def get_mode_name(self) -> str:
        """Get mode name including checkpoint name."""
        suffix = "_batched" if self.use_batching else ""
        return f"learned_{self.checkpoint.stem}{suffix}"

    def get_model_names(self, fallback_configs: list[dict[str, Any]]) -> list[str]:
        """Get model names from model_pool (ignores fallback_configs)."""
        return list(self.model_names)

    def __repr__(self) -> str:
        return (
            f"LearnedMode(checkpoint={self.checkpoint!r}, "
            f"lambda_={self.lambda_}, use_batching={self.use_batching})"
        )


class MixedMode(CollectionMode):
    """Mixed collection mode combining learned policy with roulette.

    This mode implements the β-mixed strategy for Stage B data collection:
    - With probability β: use learned policy (greedy on Q-values)
    - With probability (1-β): use uniform random (roulette)

    The propensity is correctly recorded for offline policy evaluation:
        propensity(a) = β * I[a = greedy_choice] + (1-β) / n

    Args:
        checkpoint: Path to trained Q-function checkpoint.
        model_names: List of model names (must match training order).
        beta: Probability of using learned policy (0-1). Default: 0.5.
        use_batching: Whether to enable dynamic batching for inference.
        batch_size: Maximum batch size (when use_batching=True).
        timeout: Maximum time to wait for batch to fill in seconds.

    Example:
        >>> mode = MixedMode(
        ...     checkpoint="outputs/q_function/q_function_best.pt",
        ...     model_names=["openai/gpt-5", "deepseek/deepseek-v3.2"],
        ...     beta=0.5,  # 50% learned + 50% roulette
        ... )
        >>> router = mode.create_model_or_router(model_configs)
    """

    def __init__(
        self,
        checkpoint: Path | str,
        model_names: list[str],
        beta: float = 0.5,
        use_batching: bool = False,
        batch_size: int = 8,
        timeout: float = 0.02,
        lambda_: float | None = None,
        fallback_model_name: str | None = None,
    ):
        """Initialize MixedMode.

        Args:
            checkpoint: Path to Q-function checkpoint file.
            model_names: Model names in training order.
            beta: Probability of using learned policy (0-1).
            use_batching: Whether to enable dynamic batching.
            batch_size: Maximum batch size for batched inference.
            timeout: Maximum seconds to wait for batch to fill.
            lambda_: Cost penalty coefficient. If None, uses checkpoint value.
            fallback_model_name: Model name for context window fallback.
        """
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {beta}")

        self.checkpoint = Path(checkpoint)
        self.model_names = model_names
        self.beta = beta
        self.use_batching = use_batching
        self.batch_size = batch_size
        self.timeout = timeout
        self.lambda_ = lambda_
        self.fallback_model_name = fallback_model_name

        self._cached_q_func = None
        self._cached_device: str | None = None
        self._batched_q_func = None

        import threading

        self._q_func_lock = threading.Lock()

        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint}")

    def create_model_or_router(
        self,
        model_configs: list[dict[str, Any]],
    ) -> Model:
        """Create a PropensityMixedRouter with loaded Q-function.

        Args:
            model_configs: List of model configuration dicts.

        Returns:
            PropensityMixedRouter instance with trained Q-function.
        """
        import torch
        from miniagenticrouter.research.collection.propensity_mixed_router import (
            PropensityMixedRouter,
        )
        from miniagenticrouter.research.training.q_network import NeuralQFunction

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load trained Q-function once and cache it
        with self._q_func_lock:
            if self._cached_q_func is None or self._cached_device != device:
                # Load config from checkpoint (includes max_tokens, lambda_, etc.)
                ckpt_model_names, config, ckpt_lambda = NeuralQFunction.load_config_from_checkpoint(self.checkpoint)

                # Override vllm_base_url with current config (checkpoint may have stale URL)
                _override_vllm_url_from_current_config(config)

                if ckpt_model_names != self.model_names:
                    raise ValueError(
                        "Model names mismatch between checkpoint and runtime configuration.\n"
                        f"  checkpoint: {ckpt_model_names}\n"
                        f"  runtime   : {self.model_names}\n"
                        "These must match (including order) for correct routing/Q-values."
                    )
                # Use provided lambda_ or fall back to checkpoint value
                effective_lambda = self.lambda_ if self.lambda_ is not None else ckpt_lambda
                q_func = NeuralQFunction(model_names=self.model_names, config=config, lambda_=effective_lambda)
                q_func.load(self.checkpoint)
                # Override lambda_ if provided (load may have overwritten it)
                if self.lambda_ is not None:
                    q_func.set_lambda(self.lambda_)
                q_func.to(device)
                q_func.eval()
                self._cached_q_func = q_func
                self._cached_device = device

                # Create batched wrapper if enabled
                if self.use_batching:
                    from miniagenticrouter.research.training.batched_inference import (
                        BatchedQFunction,
                    )
                    self._batched_q_func = BatchedQFunction(
                        q_func=q_func,
                        batch_size=self.batch_size,
                        timeout=self.timeout,
                    )

            # Use batched or direct Q-function
            q_function = self._batched_q_func if self.use_batching else self._cached_q_func

        # Build model_configs directly from model_pool (self.model_names)
        # This ensures we use training.yaml's model_pool, not data_split's configs
        model_configs_from_pool = [{"model_name": name} for name in self.model_names]

        # Create PropensityMixedRouter with model_pool models
        return PropensityMixedRouter(
            model_kwargs=model_configs_from_pool,
            q_function=q_function,
            beta=self.beta,
            fallback_model_name=self.fallback_model_name,
        )

    def shutdown(self) -> None:
        """Shutdown batched inference resources.

        Call this method after data collection is complete to cleanly
        stop the background worker thread.
        """
        if self._batched_q_func is not None:
            self._batched_q_func.shutdown()
            self._batched_q_func = None

    def get_mode_name(self) -> str:
        """Get mode name including checkpoint and beta."""
        beta_str = f"beta{int(self.beta * 100)}"
        suffix = "_batched" if self.use_batching else ""
        return f"mixed_{self.checkpoint.stem}_{beta_str}{suffix}"

    def get_model_names(self, fallback_configs: list[dict[str, Any]]) -> list[str]:
        """Get model names from model_pool (ignores fallback_configs)."""
        return list(self.model_names)

    def __repr__(self) -> str:
        return (
            f"MixedMode(checkpoint={self.checkpoint!r}, "
            f"beta={self.beta}, lambda_={self.lambda_}, use_batching={self.use_batching})"
        )


class PerModelRidgeMode(CollectionMode):
    """Per-model Ridge collection mode using independent Ridge regressors.

    Unlike RidgeMode which uses shared weights [z, m(a)], this mode trains
    a separate Ridge regressor for each model using only history embedding z:
        score_hat_a(z) = w_a · z + b_a
        cost_hat_a(z)  = v_a · z + c_a

    This enables context-aware model selection:
        score_hat(z, a1) - score_hat(z, a2) = (w_{a1} - w_{a2}) · z
    which depends on the history embedding z.

    Args:
        model_dir: Directory containing per_model_ridge.pkl.
        lambda_: Cost penalty coefficient (can be adjusted without retraining).
        encoder_model: HuggingFace model for history encoding. If None, reads
            from training.yaml config file.
        use_batching: Whether to enable dynamic batching for inference.
        batch_size: Maximum batch size (when use_batching=True).
        timeout: Maximum time to wait for batch to fill in seconds.

    Example:
        >>> mode = PerModelRidgeMode(
        ...     model_dir="outputs/per_model_ridge",
        ...     lambda_=1.0,
        ...     use_batching=True,
        ... )
        >>> router = mode.create_model_or_router(model_configs)
    """

    def __init__(
        self,
        model_dir: Path | str,
        lambda_: float = 1.0,
        encoder_model: str | None = None,
        use_batching: bool = False,
        batch_size: int = 16,
        timeout: float = 0.02,
        fallback_model_name: str | None = None,
    ):
        """Initialize PerModelRidgeMode.

        Args:
            model_dir: Directory containing per-model Ridge model files.
            lambda_: Cost penalty coefficient.
            encoder_model: HuggingFace model for history encoding.
                If None, reads from training.yaml.
            use_batching: Whether to enable dynamic batching.
            batch_size: Maximum batch size for batched inference.
            timeout: Maximum seconds to wait for batch to fill.
            fallback_model_name: Model name for context window fallback.
        """
        self.model_dir = Path(model_dir)
        self.lambda_ = lambda_
        self.encoder_model = encoder_model
        self.use_batching = use_batching
        self.batch_size = batch_size
        self.timeout = timeout
        self.fallback_model_name = fallback_model_name

        self._cached_q_func = None
        self._cached_device: str | None = None
        self._batched_q_func = None

        import threading
        self._q_func_lock = threading.Lock()

        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

    def create_model_or_router(
        self,
        model_configs: list[dict[str, Any]],
    ) -> Model:
        """Create a LearnedRouter with per-model Ridge Q-function.

        Args:
            model_configs: List of model configuration dicts.

        Returns:
            LearnedRouter instance with per-model Ridge Q-function.
        """
        import torch
        from miniagenticrouter.research.routers.learned import LearnedRouter
        from miniagenticrouter.research.routers.per_model_ridge_q import (
            BatchedPerModelRidgeQFunction,
            PerModelRidgeQFunction,
        )

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load per-model Ridge Q-function once and cache it
        with self._q_func_lock:
            if self._cached_q_func is None or self._cached_device != device:
                q_func = PerModelRidgeQFunction(
                    model_dir=self.model_dir,
                    lambda_=self.lambda_,
                    encoder_model=self.encoder_model,
                    device=device,
                )
                self._cached_q_func = q_func
                self._cached_device = device

                # Create batched wrapper if enabled
                if self.use_batching:
                    self._batched_q_func = BatchedPerModelRidgeQFunction(
                        ridge_q_func=q_func,
                        batch_size=self.batch_size,
                        timeout=self.timeout,
                    )

            # Use batched or direct Q-function
            q_function = self._batched_q_func if self.use_batching else self._cached_q_func

        # Build model_configs directly from trained model names
        # This ensures we use the models from the trained model, not data_split's configs
        trained_model_names = self._cached_q_func._model_names
        model_configs_from_trained = [{"model_name": name} for name in trained_model_names]

        # Create LearnedRouter with trained models
        return LearnedRouter(
            model_kwargs=model_configs_from_trained,
            q_function=q_function,
            fallback_model_name=self.fallback_model_name,
        )

    def shutdown(self) -> None:
        """Shutdown batched inference resources.

        Call this method after data collection is complete to cleanly
        stop the background worker thread.
        """
        if self._batched_q_func is not None:
            self._batched_q_func.shutdown()
            self._batched_q_func = None

    def get_mode_name(self) -> str:
        """Get mode name including lambda."""
        lambda_str = f"lambda{int(self.lambda_ * 10)}"
        suffix = "_batched" if self.use_batching else ""
        return f"per_model_ridge_{lambda_str}{suffix}"

    def get_model_names(self, fallback_configs: list[dict[str, Any]]) -> list[str]:
        """Get model names from training.yaml model_pool (ignores fallback_configs)."""
        from miniagenticrouter.research.utils.config import get_model_pool, load_training_config
        config = load_training_config()
        return get_model_pool(config)

    def __repr__(self) -> str:
        return (
            f"PerModelRidgeMode(model_dir={self.model_dir!r}, "
            f"lambda_={self.lambda_}, use_batching={self.use_batching})"
        )


class ClusterMode(CollectionMode):
    """Cluster-based collection mode using KMeans clustering.

    Uses KMeans clustering to partition history embedding space into K regions.
    Each region maintains per-model statistics (score_mean, cost_mean).

    At inference:
        Q = score_mean - λ * cost_mean

    This provides state-dependent routing where different clusters can
    prefer different models based on their statistics.

    Args:
        model_dir: Directory containing cluster_kmeans.pkl and cluster_stats.json.
        lambda_: Cost penalty coefficient (can be adjusted without retraining).
        encoder_model: HuggingFace model for history encoding. If None, reads
            from training.yaml config file.
        use_batching: Whether to enable dynamic batching for inference.
        batch_size: Maximum batch size (when use_batching=True).
        timeout: Maximum time to wait for batch to fill in seconds.

    Example:
        >>> mode = ClusterMode(
        ...     model_dir="outputs/cluster_baseline",
        ...     lambda_=1.0,
        ...     use_batching=True,
        ... )
        >>> router = mode.create_model_or_router(model_configs)
    """

    def __init__(
        self,
        model_dir: Path | str,
        lambda_: float = 1.0,
        encoder_model: str | None = None,
        use_batching: bool = False,
        batch_size: int = 16,
        timeout: float = 0.02,
        fallback_model_name: str | None = None,
    ):
        """Initialize ClusterMode.

        Args:
            model_dir: Directory containing cluster model files.
            lambda_: Cost penalty coefficient.
            encoder_model: HuggingFace model for history encoding.
                If None, reads from training.yaml.
            use_batching: Whether to enable dynamic batching.
            batch_size: Maximum batch size for batched inference.
            timeout: Maximum seconds to wait for batch to fill.
            fallback_model_name: Model name for context window fallback.
        """
        self.model_dir = Path(model_dir)
        self.lambda_ = lambda_
        self.encoder_model = encoder_model
        self.use_batching = use_batching
        self.batch_size = batch_size
        self.timeout = timeout
        self.fallback_model_name = fallback_model_name

        self._cached_q_func = None
        self._cached_device: str | None = None
        self._batched_q_func = None

        import threading
        self._q_func_lock = threading.Lock()

        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

    def create_model_or_router(
        self,
        model_configs: list[dict[str, Any]],
    ) -> Model:
        """Create a LearnedRouter with cluster Q-function.

        Args:
            model_configs: List of model configuration dicts.

        Returns:
            LearnedRouter instance with cluster Q-function.
        """
        import torch
        from miniagenticrouter.research.routers.learned import LearnedRouter
        from miniagenticrouter.research.routers.cluster_q import (
            BatchedClusterQFunction,
            ClusterQFunction,
        )

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load cluster Q-function once and cache it
        with self._q_func_lock:
            if self._cached_q_func is None or self._cached_device != device:
                q_func = ClusterQFunction(
                    model_dir=self.model_dir,
                    lambda_=self.lambda_,
                    encoder_model=self.encoder_model,
                    device=device,
                )
                self._cached_q_func = q_func
                self._cached_device = device

                # Create batched wrapper if enabled
                if self.use_batching:
                    self._batched_q_func = BatchedClusterQFunction(
                        cluster_q_func=q_func,
                        batch_size=self.batch_size,
                        timeout=self.timeout,
                    )

            # Use batched or direct Q-function
            q_function = self._batched_q_func if self.use_batching else self._cached_q_func

        # Build model_configs directly from trained model names
        # This ensures we use the models from the trained model, not data_split's configs
        trained_model_names = self._cached_q_func._model_names
        model_configs_from_trained = [{"model_name": name} for name in trained_model_names]

        # Create LearnedRouter with trained models
        return LearnedRouter(
            model_kwargs=model_configs_from_trained,
            q_function=q_function,
            fallback_model_name=self.fallback_model_name,
        )

    def shutdown(self) -> None:
        """Shutdown batched inference resources.

        Call this method after data collection is complete to cleanly
        stop the background worker thread.
        """
        if self._batched_q_func is not None:
            self._batched_q_func.shutdown()
            self._batched_q_func = None

    def get_mode_name(self) -> str:
        """Get mode name including lambda."""
        lambda_str = f"lambda{int(self.lambda_ * 10)}"
        suffix = "_batched" if self.use_batching else ""
        return f"cluster_{lambda_str}{suffix}"

    def get_model_names(self, fallback_configs: list[dict[str, Any]]) -> list[str]:
        """Get model names from training.yaml model_pool (ignores fallback_configs)."""
        from miniagenticrouter.research.utils.config import get_model_pool, load_training_config
        config = load_training_config()
        return get_model_pool(config)

    def __repr__(self) -> str:
        return (
            f"ClusterMode(model_dir={self.model_dir!r}, "
            f"lambda_={self.lambda_}, use_batching={self.use_batching})"
        )


class PerModelXGBoostMode(CollectionMode):
    """Per-model XGBoost collection mode using independent XGBoost regressors.

    Unlike shared-weight approaches, this mode trains a separate XGBRegressor
    for each model using only history embedding z:
        score_hat_a(z) = XGB_score_a(z)
        cost_hat_a(z)  = XGB_cost_a(z)

    This enables context-aware model selection where different models can
    be preferred for different conversation contexts.

    Args:
        model_dir: Directory containing per_model_xgboost.pkl.
        lambda_: Cost penalty coefficient (can be adjusted without retraining).
        encoder_model: HuggingFace model for history encoding. If None, reads
            from training.yaml config file.
        use_batching: Whether to enable dynamic batching for inference.
        batch_size: Maximum batch size (when use_batching=True).
        timeout: Maximum time to wait for batch to fill in seconds.

    Example:
        >>> mode = PerModelXGBoostMode(
        ...     model_dir="outputs/per_model_xgboost",
        ...     lambda_=1.0,
        ...     use_batching=True,
        ... )
        >>> router = mode.create_model_or_router(model_configs)
    """

    def __init__(
        self,
        model_dir: Path | str,
        lambda_: float = 1.0,
        encoder_model: str | None = None,
        use_batching: bool = False,
        batch_size: int = 16,
        timeout: float = 0.02,
        fallback_model_name: str | None = None,
    ):
        """Initialize PerModelXGBoostMode.

        Args:
            model_dir: Directory containing per-model XGBoost model files.
            lambda_: Cost penalty coefficient.
            encoder_model: HuggingFace model for history encoding.
                If None, reads from training.yaml.
            use_batching: Whether to enable dynamic batching.
            batch_size: Maximum batch size for batched inference.
            timeout: Maximum seconds to wait for batch to fill.
            fallback_model_name: Model name for context window fallback.
        """
        self.model_dir = Path(model_dir)
        self.lambda_ = lambda_
        self.encoder_model = encoder_model
        self.use_batching = use_batching
        self.batch_size = batch_size
        self.timeout = timeout
        self.fallback_model_name = fallback_model_name

        self._cached_q_func = None
        self._cached_device: str | None = None
        self._batched_q_func = None

        import threading
        self._q_func_lock = threading.Lock()

        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

    def create_model_or_router(
        self,
        model_configs: list[dict[str, Any]],
    ) -> Model:
        """Create a LearnedRouter with per-model XGBoost Q-function.

        Args:
            model_configs: List of model configuration dicts.

        Returns:
            LearnedRouter instance with per-model XGBoost Q-function.
        """
        import torch
        from miniagenticrouter.research.routers.learned import LearnedRouter
        from miniagenticrouter.research.routers.per_model_xgboost_q import (
            BatchedPerModelXGBoostQFunction,
            PerModelXGBoostQFunction,
        )

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load per-model XGBoost Q-function once and cache it
        with self._q_func_lock:
            if self._cached_q_func is None or self._cached_device != device:
                q_func = PerModelXGBoostQFunction(
                    model_dir=self.model_dir,
                    lambda_=self.lambda_,
                    encoder_model=self.encoder_model,
                    device=device,
                )
                self._cached_q_func = q_func
                self._cached_device = device

                # Create batched wrapper if enabled
                if self.use_batching:
                    self._batched_q_func = BatchedPerModelXGBoostQFunction(
                        xgb_q_func=q_func,
                        batch_size=self.batch_size,
                        timeout=self.timeout,
                    )

            # Use batched or direct Q-function
            q_function = self._batched_q_func if self.use_batching else self._cached_q_func

        # Build model_configs directly from trained model names
        # This ensures we use the models from the trained model, not data_split's configs
        trained_model_names = self._cached_q_func._model_names
        model_configs_from_trained = [{"model_name": name} for name in trained_model_names]

        # Create LearnedRouter with trained models
        return LearnedRouter(
            model_kwargs=model_configs_from_trained,
            q_function=q_function,
            fallback_model_name=self.fallback_model_name,
        )

    def shutdown(self) -> None:
        """Shutdown batched inference resources.

        Call this method after data collection is complete to cleanly
        stop the background worker thread.
        """
        if self._batched_q_func is not None:
            self._batched_q_func.shutdown()
            self._batched_q_func = None

    def get_mode_name(self) -> str:
        """Get mode name including lambda."""
        lambda_str = f"lambda{int(self.lambda_ * 10)}"
        suffix = "_batched" if self.use_batching else ""
        return f"per_model_xgboost_{lambda_str}{suffix}"

    def get_model_names(self, fallback_configs: list[dict[str, Any]]) -> list[str]:
        """Get model names from training.yaml model_pool (ignores fallback_configs)."""
        from miniagenticrouter.research.utils.config import get_model_pool, load_training_config
        config = load_training_config()
        return get_model_pool(config)

    def __repr__(self) -> str:
        return (
            f"PerModelXGBoostMode(model_dir={self.model_dir!r}, "
            f"lambda_={self.lambda_}, use_batching={self.use_batching})"
        )


class HeuristicMode(CollectionMode):
    """Heuristic collection mode using rule-based model selection.

    This mode uses simple heuristic rules for model routing:
    - First step: strong model
    - After tool error: strong model
    - Late phase: value model
    - Otherwise: random cheap model

    Args:
        task_type: Task type ("hle" or "scienceworld").
        config_path: Path to heuristic.yaml config file (optional).
        strong_model: Override strong model name.
        value_model: Override value model name.
        cheap_pool: Override cheap model pool.
        max_steps: Override max steps for task.

    Example:
        >>> mode = HeuristicMode(
        ...     task_type="hle",
        ...     max_steps=30,
        ... )
        >>> router = mode.create_model_or_router(model_configs)
    """

    def __init__(
        self,
        task_type: str = "hle",
        config_path: Path | str | None = None,
        strong_model: str | None = None,
        value_model: str | None = None,
        cheap_pool: list[str] | None = None,
        max_steps: int | None = None,
        fallback_model_name: str | None = None,
    ):
        """Initialize HeuristicMode.

        Args:
            task_type: Task type ("hle" or "scienceworld").
            config_path: Path to heuristic.yaml config file.
            strong_model: Override strong model name.
            value_model: Override value model name.
            cheap_pool: Override cheap model pool.
            max_steps: Override max steps.
            fallback_model_name: Model name for context window fallback.
        """
        self.task_type = task_type
        self.fallback_model_name = fallback_model_name
        self.config_path = Path(config_path) if config_path else None

        # Load config from YAML
        self._config = self._load_config()

        # Override with explicit parameters
        task_config = self._config.get("tasks", {}).get(task_type, {})
        self.strong_model = strong_model or task_config.get(
            "strong_model", "openai/gpt-5"
        )
        self.value_model = value_model or task_config.get(
            "value_model", "deepseek/deepseek-v3.2"
        )
        self.cheap_pool = cheap_pool or task_config.get(
            "cheap_pool",
            [
                "minimax/minimax-m2",
                "moonshotai/kimi-k2-0905",
                "x-ai/grok-4.1-fast",
            ],
        )
        self.max_steps = max_steps or task_config.get(
            "max_steps", 30 if task_type == "hle" else 50
        )

        # Get error detection rules
        error_rules = self._config.get("tool_error_rules", {})
        self.error_patterns = error_rules.get(
            "error_patterns",
            ["unknown_tool:", "validation_error:", "execution_error:"],
        )
        self.check_returncode = error_rules.get("check_returncode", True)
        self.late_phase_fraction = self._config.get("late_phase_fraction", 0.667)

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        import yaml

        if self.config_path and self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f) or {}

        # Try default location
        from miniagenticrouter.config import builtin_config_dir

        default_path = builtin_config_dir / "research" / "heuristic.yaml"
        if default_path.exists():
            with open(default_path) as f:
                return yaml.safe_load(f) or {}

        return {}

    def create_model_or_router(
        self,
        model_configs: list[dict[str, Any]],
    ) -> Model:
        """Create a HeuristicRouter.

        Args:
            model_configs: List of model configuration dicts (from data_split).

        Returns:
            HeuristicRouter instance.
        """
        from miniagenticrouter.routers.heuristic import HeuristicRouter

        # Build model_kwargs from all required models (sorted for deterministic order)
        required_models = sorted(
            {self.strong_model, self.value_model} | set(self.cheap_pool)
        )
        model_kwargs = [{"model_name": name} for name in required_models]

        return HeuristicRouter(
            model_kwargs=model_kwargs,
            task_type=self.task_type,
            max_steps=self.max_steps,
            strong_model=self.strong_model,
            value_model=self.value_model,
            cheap_pool=self.cheap_pool,
            late_phase_fraction=self.late_phase_fraction,
            error_patterns=self.error_patterns,
            check_returncode=self.check_returncode,
            fallback_model_name=self.fallback_model_name,
        )

    def get_mode_name(self) -> str:
        """Get mode name for output directory."""
        return f"heuristic_{self.task_type}"

    def get_model_names(self, fallback_configs: list[dict[str, Any]]) -> list[str]:
        """Get all model names used by this mode (sorted for deterministic order)."""
        return sorted({self.strong_model, self.value_model} | set(self.cheap_pool))

    def __repr__(self) -> str:
        return (
            f"HeuristicMode(task_type={self.task_type!r}, "
            f"max_steps={self.max_steps}, "
            f"strong_model={self.strong_model!r})"
        )


class LLMRouterMode(CollectionMode):
    """LLM-based routing mode.

    Uses an LLM policy to decide which model to use for each query.
    The LLM analyzes the task and selects the best model based on
    task complexity, model strengths, and cost-effectiveness.

    Example:
        >>> mode = LLMRouterMode(
        ...     policy_model_name="openai/gpt-5",
        ...     history_turns=3,
        ...     model_descriptors={
        ...         "deepseek/deepseek-v3.2": "Low cost, simple tasks",
        ...         "openai/gpt-5": "High quality, complex reasoning",
        ...     },
        ... )
        >>> router = mode.create_model_or_router(model_configs)
    """

    def __init__(
        self,
        policy_model_name: str = "openai/gpt-5",
        history_turns: int = 3,
        model_descriptors: dict[str, str] | None = None,
        fallback_model_name: str | None = None,
    ):
        """Initialize LLMRouterMode.

        Args:
            policy_model_name: Name of the LLM to use for routing decisions.
            history_turns: Number of conversation turns to show (0 = all).
            model_descriptors: Optional descriptions of each model's strengths.
            fallback_model_name: Model name for context window fallback.
        """
        self.policy_model_name = policy_model_name
        self.history_turns = history_turns
        self.model_descriptors = model_descriptors or {}
        self.fallback_model_name = fallback_model_name

    def create_model_or_router(
        self,
        model_configs: list[dict[str, Any]],
    ) -> Model:
        """Create an LLMRouter.

        Args:
            model_configs: List of model configuration dicts (from data_split).

        Returns:
            LLMRouter instance.
        """
        from miniagenticrouter.research.routers.llmrouter import LLMRouter

        return LLMRouter(
            model_kwargs=model_configs,
            policy_model_name=self.policy_model_name,
            history_turns=self.history_turns,
            model_descriptors=self.model_descriptors,
            fallback_model_name=self.fallback_model_name,
        )

    def get_mode_name(self) -> str:
        """Get mode name for output directory."""
        policy_short = self.policy_model_name.split("/")[-1]
        return f"llmrouter_{policy_short}"

    def get_model_names(self, fallback_configs: list[dict[str, Any]]) -> list[str]:
        """Get all model names used by this mode."""
        return [cfg.get("model_name", "") for cfg in fallback_configs]

    def __repr__(self) -> str:
        return (
            f"LLMRouterMode(policy_model_name={self.policy_model_name!r}, "
            f"model_descriptors={len(self.model_descriptors)} entries)"
        )
