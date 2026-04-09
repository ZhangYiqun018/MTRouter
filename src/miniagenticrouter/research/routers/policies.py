"""Selection policies for learned routers.

This module provides different strategies for selecting models based on
Q-values or other scores. Policies can be used with LearnedRouter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class SelectionPolicy(ABC):
    """Abstract base class for model selection policies.

    A policy takes Q-values (or scores) for each model and returns
    the index of the selected model.
    """

    @abstractmethod
    def select(
        self,
        q_values: np.ndarray,
        model_indices: list[int] | None = None,
    ) -> int:
        """Select a model based on Q-values.

        Args:
            q_values: Array of Q-values for each model.
            model_indices: Optional list of valid model indices.
                If None, all indices are valid.

        Returns:
            Index of the selected model.
        """
        ...

    def reset(self) -> None:
        """Reset policy state (e.g., for new episode)."""
        pass

    def get_config(self) -> dict[str, Any]:
        """Get policy configuration for logging."""
        return {"policy_type": self.__class__.__name__}


class GreedyPolicy(SelectionPolicy):
    """Greedy policy: always select the model with highest Q-value.

    This is the simplest exploitation policy with no exploration.

    Example:
        >>> policy = GreedyPolicy()
        >>> q_values = np.array([0.5, 0.8, 0.3])
        >>> policy.select(q_values)
        1  # Index of highest value
    """

    def select(
        self,
        q_values: np.ndarray,
        model_indices: list[int] | None = None,
    ) -> int:
        """Select model with highest Q-value.

        Args:
            q_values: Array of Q-values.
            model_indices: Optional valid indices (uses all if None).

        Returns:
            Index of model with highest Q-value.
        """
        if model_indices is None:
            return int(np.argmax(q_values))

        # Filter to valid indices
        valid_q = np.array([q_values[i] for i in model_indices])
        best_idx = int(np.argmax(valid_q))
        return model_indices[best_idx]

    def __repr__(self) -> str:
        return "GreedyPolicy()"


class EpsilonGreedyPolicy(SelectionPolicy):
    """Epsilon-greedy policy: explore with probability epsilon.

    With probability epsilon, select a random model.
    With probability 1-epsilon, select the model with highest Q-value.

    Supports epsilon decay for gradually reducing exploration.

    Args:
        epsilon: Initial exploration probability (0-1).
        decay: Decay rate per selection (multiplicative).
        min_epsilon: Minimum epsilon value after decay.

    Example:
        >>> policy = EpsilonGreedyPolicy(epsilon=0.1, decay=0.99)
        >>> q_values = np.array([0.5, 0.8, 0.3])
        >>> policy.select(q_values)  # 10% chance of random, 90% greedy
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        decay: float = 1.0,
        min_epsilon: float = 0.01,
    ):
        """Initialize EpsilonGreedyPolicy.

        Args:
            epsilon: Initial exploration probability.
            decay: Decay rate per selection.
            min_epsilon: Minimum epsilon value.
        """
        if not 0 <= epsilon <= 1:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        if not 0 < decay <= 1:
            raise ValueError(f"decay must be in (0, 1], got {decay}")
        if not 0 <= min_epsilon <= epsilon:
            raise ValueError(
                f"min_epsilon must be in [0, epsilon], got {min_epsilon}"
            )

        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon
        self._step = 0

    def select(
        self,
        q_values: np.ndarray,
        model_indices: list[int] | None = None,
    ) -> int:
        """Select model with epsilon-greedy strategy.

        Args:
            q_values: Array of Q-values.
            model_indices: Optional valid indices.

        Returns:
            Selected model index.
        """
        # Calculate current epsilon with decay
        current_eps = max(
            self.min_epsilon,
            self.epsilon * (self.decay**self._step),
        )
        self._step += 1

        if model_indices is None:
            model_indices = list(range(len(q_values)))

        # Explore: random selection
        if np.random.random() < current_eps:
            return np.random.choice(model_indices)

        # Exploit: greedy selection
        valid_q = np.array([q_values[i] for i in model_indices])
        best_idx = int(np.argmax(valid_q))
        return model_indices[best_idx]

    def reset(self) -> None:
        """Reset step counter (but keep epsilon)."""
        self._step = 0

    def get_current_epsilon(self) -> float:
        """Get current epsilon value after decay."""
        return max(self.min_epsilon, self.epsilon * (self.decay**self._step))

    def get_config(self) -> dict[str, Any]:
        """Get policy configuration."""
        return {
            "policy_type": self.__class__.__name__,
            "epsilon": self.epsilon,
            "decay": self.decay,
            "min_epsilon": self.min_epsilon,
            "current_epsilon": self.get_current_epsilon(),
        }

    def __repr__(self) -> str:
        return (
            f"EpsilonGreedyPolicy(epsilon={self.epsilon}, "
            f"decay={self.decay}, min_epsilon={self.min_epsilon})"
        )


class SoftmaxPolicy(SelectionPolicy):
    """Softmax (Boltzmann) policy: select proportionally to Q-values.

    Selection probability is proportional to exp(Q/temperature).
    Higher temperature = more random, lower temperature = more greedy.

    Args:
        temperature: Temperature parameter (>0).

    Example:
        >>> policy = SoftmaxPolicy(temperature=1.0)
        >>> q_values = np.array([0.5, 0.8, 0.3])
        >>> # Higher Q-values have higher selection probability
        >>> policy.select(q_values)
    """

    def __init__(self, temperature: float = 1.0):
        """Initialize SoftmaxPolicy.

        Args:
            temperature: Temperature parameter.
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.temperature = temperature

    def select(
        self,
        q_values: np.ndarray,
        model_indices: list[int] | None = None,
    ) -> int:
        """Select model with softmax probability.

        Args:
            q_values: Array of Q-values.
            model_indices: Optional valid indices.

        Returns:
            Selected model index.
        """
        if model_indices is None:
            model_indices = list(range(len(q_values)))

        valid_q = np.array([q_values[i] for i in model_indices])

        # Compute softmax with numerical stability
        q_scaled = valid_q / self.temperature
        q_scaled = q_scaled - np.max(q_scaled)  # Prevent overflow
        exp_q = np.exp(q_scaled)
        probs = exp_q / np.sum(exp_q)

        # Sample from distribution
        selected_idx = np.random.choice(len(model_indices), p=probs)
        return model_indices[selected_idx]

    def get_probabilities(
        self,
        q_values: np.ndarray,
        model_indices: list[int] | None = None,
    ) -> np.ndarray:
        """Get selection probabilities for each model.

        Args:
            q_values: Array of Q-values.
            model_indices: Optional valid indices.

        Returns:
            Array of selection probabilities.
        """
        if model_indices is None:
            model_indices = list(range(len(q_values)))

        valid_q = np.array([q_values[i] for i in model_indices])

        q_scaled = valid_q / self.temperature
        q_scaled = q_scaled - np.max(q_scaled)
        exp_q = np.exp(q_scaled)
        probs = exp_q / np.sum(exp_q)

        # Map back to full array if needed
        if len(model_indices) == len(q_values):
            return probs

        full_probs = np.zeros(len(q_values))
        for i, idx in enumerate(model_indices):
            full_probs[idx] = probs[i]
        return full_probs

    def get_config(self) -> dict[str, Any]:
        """Get policy configuration."""
        return {
            "policy_type": self.__class__.__name__,
            "temperature": self.temperature,
        }

    def __repr__(self) -> str:
        return f"SoftmaxPolicy(temperature={self.temperature})"


class UCBPolicy(SelectionPolicy):
    """Upper Confidence Bound (UCB) policy.

    Selects based on Q-value plus exploration bonus:
    score = Q + c * sqrt(log(N) / n)

    where:
    - Q is the estimated value
    - c is the exploration coefficient
    - N is total number of selections
    - n is number of times this model was selected

    Args:
        c: Exploration coefficient (higher = more exploration).

    Example:
        >>> policy = UCBPolicy(c=2.0)
        >>> q_values = np.array([0.5, 0.8, 0.3])
        >>> policy.select(q_values)
    """

    def __init__(self, c: float = 2.0):
        """Initialize UCBPolicy.

        Args:
            c: Exploration coefficient.
        """
        if c < 0:
            raise ValueError(f"c must be >= 0, got {c}")

        self.c = c
        self._total_selections = 0
        self._model_selections: dict[int, int] = {}

    def select(
        self,
        q_values: np.ndarray,
        model_indices: list[int] | None = None,
    ) -> int:
        """Select model using UCB.

        Args:
            q_values: Array of Q-values.
            model_indices: Optional valid indices.

        Returns:
            Selected model index.
        """
        if model_indices is None:
            model_indices = list(range(len(q_values)))

        self._total_selections += 1

        # Compute UCB scores
        ucb_scores = []
        for idx in model_indices:
            n = self._model_selections.get(idx, 0)
            if n == 0:
                # Unselected models get infinite bonus (select them first)
                ucb_scores.append(float("inf"))
            else:
                bonus = self.c * np.sqrt(np.log(self._total_selections) / n)
                ucb_scores.append(q_values[idx] + bonus)

        # Select highest UCB score
        best_idx = int(np.argmax(ucb_scores))
        selected = model_indices[best_idx]

        # Update selection count
        self._model_selections[selected] = (
            self._model_selections.get(selected, 0) + 1
        )

        return selected

    def reset(self) -> None:
        """Reset selection counts."""
        self._total_selections = 0
        self._model_selections.clear()

    def get_config(self) -> dict[str, Any]:
        """Get policy configuration."""
        return {
            "policy_type": self.__class__.__name__,
            "c": self.c,
            "total_selections": self._total_selections,
            "model_selections": dict(self._model_selections),
        }

    def __repr__(self) -> str:
        return f"UCBPolicy(c={self.c})"


class ThompsonSamplingPolicy(SelectionPolicy):
    """Thompson Sampling policy for model selection.

    Maintains Beta distributions for each model's success rate,
    samples from posteriors, and selects the model with highest sample.

    Assumes binary rewards (success/failure). For continuous rewards,
    use UCB or other policies.

    Args:
        alpha_prior: Prior alpha parameter (successes).
        beta_prior: Prior beta parameter (failures).

    Example:
        >>> policy = ThompsonSamplingPolicy()
        >>> q_values = np.array([0.5, 0.8, 0.3])  # Treated as success rates
        >>> policy.select(q_values)
        >>> policy.update(selected_idx=1, reward=1.0)  # Update after observing
    """

    def __init__(
        self,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
    ):
        """Initialize ThompsonSamplingPolicy.

        Args:
            alpha_prior: Prior successes.
            beta_prior: Prior failures.
        """
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior

        # Posterior parameters for each model
        self._alphas: dict[int, float] = {}
        self._betas: dict[int, float] = {}

    def select(
        self,
        q_values: np.ndarray,
        model_indices: list[int] | None = None,
    ) -> int:
        """Select model using Thompson Sampling.

        Args:
            q_values: Array of Q-values (used as prior mean if no data).
            model_indices: Optional valid indices.

        Returns:
            Selected model index.
        """
        if model_indices is None:
            model_indices = list(range(len(q_values)))

        # Sample from posterior for each model
        samples = []
        for idx in model_indices:
            alpha = self._alphas.get(idx, self.alpha_prior)
            beta = self._betas.get(idx, self.beta_prior)
            sample = np.random.beta(alpha, beta)
            samples.append(sample)

        # Select highest sample
        best_idx = int(np.argmax(samples))
        return model_indices[best_idx]

    def update(self, model_idx: int, reward: float) -> None:
        """Update posterior after observing reward.

        Args:
            model_idx: Index of the model that was used.
            reward: Observed reward (0 or 1 for binary, or in [0,1]).
        """
        # Clamp reward to [0, 1]
        reward = max(0.0, min(1.0, reward))

        if model_idx not in self._alphas:
            self._alphas[model_idx] = self.alpha_prior
            self._betas[model_idx] = self.beta_prior

        self._alphas[model_idx] += reward
        self._betas[model_idx] += 1 - reward

    def reset(self) -> None:
        """Reset posteriors to priors."""
        self._alphas.clear()
        self._betas.clear()

    def get_config(self) -> dict[str, Any]:
        """Get policy configuration."""
        return {
            "policy_type": self.__class__.__name__,
            "alpha_prior": self.alpha_prior,
            "beta_prior": self.beta_prior,
            "posteriors": {
                idx: {"alpha": self._alphas.get(idx), "beta": self._betas.get(idx)}
                for idx in set(self._alphas.keys()) | set(self._betas.keys())
            },
        }

    def __repr__(self) -> str:
        return (
            f"ThompsonSamplingPolicy(alpha_prior={self.alpha_prior}, "
            f"beta_prior={self.beta_prior})"
        )
