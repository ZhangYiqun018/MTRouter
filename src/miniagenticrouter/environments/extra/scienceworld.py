"""ScienceWorld environment adapter for mini-SWE-agent.

This environment wraps the ScienceWorld simulation for text-based
science experiment tasks.

!!! warning
    This environment requires the `scienceworld` package to be installed:
    `pip install mtrouter[scienceworld]`
"""

from dataclasses import asdict, dataclass, field
from typing import Any


# Action categories for query commands (prefixes to match valid actions)
ACTION_CATEGORIES = {
    "navigation": ["go ", "go to ", "go through ", "walk ", "move ", "teleport "],
    "object": ["pick up ", "get ", "take ", "put ", "drop ", "pour ", "dunk "],
    "observation": ["look", "examine ", "inventory", "read "],
    "device": ["activate ", "turn on ", "turn off ", "deactivate ", "use "],
    "door": ["open ", "close "],
    "electrical": ["connect ", "disconnect "],
    "interaction": ["mix ", "stir ", "eat ", "consume ", "flush ", "focus "],
}


@dataclass
class ScienceWorldEnvironmentConfig:
    """Configuration for ScienceWorld environment.

    Attributes:
        task_name: Name of the ScienceWorld task (e.g., "boil", "melt-ice").
        variation_idx: Task variation index (default 0).
        simplification_str: Simplification options string.
    """

    task_name: str = ""
    variation_idx: int = 0
    simplification_str: str = ""


class ScienceWorldEnvironment:
    """ScienceWorld environment adapter.

    This class wraps the ScienceWorld simulation environment, providing
    a compatible interface for use with FlexibleAgent.

    Example:
        >>> env = ScienceWorldEnvironment(task_name="boil", variation_idx=0)
        >>> result = env.step("look around")
        >>> print(result["observation"])
        'You are in a kitchen...'
    """

    def __init__(
        self,
        *,
        config_class: type = ScienceWorldEnvironmentConfig,
        **kwargs,
    ):
        """Initialize the ScienceWorld environment.

        Args:
            config_class: Configuration dataclass to use.
            **kwargs: Configuration options passed to config_class.

        Raises:
            ImportError: If scienceworld package is not installed.
        """
        self.config = config_class(**kwargs)
        self._env = None
        self._task_description = ""
        self._valid_actions: list[str] = []
        self._last_score: float = 0.0

    def _ensure_env(self):
        """Lazily initialize the ScienceWorld environment."""
        if self._env is None:
            try:
                from scienceworld import ScienceWorldEnv
            except ImportError as e:
                raise ImportError(
                    "scienceworld package is required. "
                    "Install with: pip install mtrouter[scienceworld]"
                ) from e

            self._env = ScienceWorldEnv("")
            if self.config.task_name:
                self.load_task(self.config.task_name, self.config.variation_idx)

    def load_task(self, task_name: str, variation_idx: int = 0) -> dict:
        """Load a specific task and variation.

        Args:
            task_name: Name of the task (e.g., "boil", "melt-ice").
            variation_idx: Task variation index.

        Returns:
            Dict with initial observation and task info.
        """
        self._ensure_env()
        self.config.task_name = task_name
        self.config.variation_idx = variation_idx

        # Load the task
        self._env.load(
            task_name,
            variation_idx,
            simplificationStr=self.config.simplification_str,
        )

        # Get initial observation
        obs, info = self._env.reset()
        self._task_description = info.get("taskDesc", "")
        self._valid_actions = info.get("valid", [])

        return {
            "observation": obs,
            "task_description": self._task_description,
            "valid_actions": self._valid_actions,
        }

    def step(self, command: str) -> dict:
        """Execute a command in the ScienceWorld environment.

        Args:
            command: Text command to execute. Commands starting with '?' are
                     query commands that return action space info without
                     consuming a game step.

        Returns:
            Dict containing:
            - "observation": Text observation from the environment
            - "reward": Reward signal
            - "done": Whether the episode is finished
            - "score": Current score
            - "valid_actions": List of valid actions
        """
        self._ensure_env()

        # Intercept query commands (free actions, no game step consumed)
        if command.startswith("?"):
            return self._handle_query(command[1:].strip())

        # Execute the action
        obs, reward, done, info = self._env.step(command)

        # Update state from info
        self._valid_actions = info.get("valid", [])
        self._last_score = info.get("score", 0)

        return {
            "observation": obs,
            "reward": reward,
            "done": done,
            "score": self._last_score,
            "valid_actions": self._valid_actions,
        }

    def _handle_query(self, query: str) -> dict:
        """Handle action space query commands.

        These are meta-commands that return information about available actions
        without consuming a game step. Useful for agents to explore the action
        space on demand.

        Args:
            query: Query string (without the leading '?').

        Returns:
            Dict with observation containing query results.
        """
        valid_actions = self._valid_actions
        query_lower = query.lower()

        if query_lower in ACTION_CATEGORIES:
            # Return actions matching the specified category
            prefixes = ACTION_CATEGORIES[query_lower]
            filtered = [
                a for a in valid_actions
                if any(a.lower().startswith(p) for p in prefixes)
            ]
            if filtered:
                obs = f"[Query Result] {query} actions ({len(filtered)} found):\n"
                obs += "\n".join(f"  - {a}" for a in filtered[:30])
                if len(filtered) > 30:
                    obs += f"\n  ... and {len(filtered) - 30} more"
            else:
                obs = f"[Query Result] No {query} actions available in current state."

        elif query_lower == "all":
            obs = f"[Query Result] All valid actions ({len(valid_actions)}):\n"
            obs += "\n".join(f"  - {a}" for a in valid_actions[:50])
            if len(valid_actions) > 50:
                obs += f"\n  ... and {len(valid_actions) - 50} more"

        elif query_lower in ("categories", "help"):
            obs = "[Query Help] Available query commands:\n"
            obs += "  ?categories - Show this help\n"
            obs += "  ?all - Show all valid actions\n"
            for cat in ACTION_CATEGORIES:
                obs += f"  ?{cat} - Show {cat} actions\n"

        else:
            obs = f"[Query Error] Unknown query: '{query}'. Use ?categories for help."

        return {
            "observation": obs,
            "reward": 0,
            "done": False,
            "score": self._last_score,
            "valid_actions": valid_actions,  # Unchanged
        }

    def execute(self, command: str, cwd: str = "") -> dict:
        """Execute method for Environment protocol compatibility.

        This allows ScienceWorldEnvironment to be used with tools
        that expect the standard Environment interface.

        Args:
            command: Text command to execute.
            cwd: Ignored for ScienceWorld.

        Returns:
            Dict with output and returncode.
        """
        result = self.step(command)
        return {
            "output": result["observation"],
            "returncode": 0,
            "done": result["done"],
        }

    def get_task_names(self) -> list[str]:
        """Get list of available task names."""
        self._ensure_env()
        return self._env.get_task_names()

    def get_task_description(self) -> str:
        """Get the current task description."""
        return self._task_description

    def get_valid_actions(self) -> list[str]:
        """Get list of currently valid actions."""
        return self._valid_actions

    def get_score(self) -> float:
        """Get current score (from last step)."""
        return self._last_score

    def get_template_vars(self) -> dict[str, Any]:
        """Return environment variables for template rendering."""
        return asdict(self.config) | {
            "task_description": self._task_description,
            "valid_actions": self._valid_actions,
        }

    def reset(self) -> dict:
        """Reset the current task to initial state."""
        if self.config.task_name:
            return self.load_task(self.config.task_name, self.config.variation_idx)
        return {"observation": "", "task_description": "", "valid_actions": []}

    def close(self):
        """Close the environment."""
        if self._env is not None:
            self._env.close()
            self._env = None

    def __enter__(self):
        """Enter context manager."""
        self._ensure_env()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager, ensuring cleanup."""
        self.close()
        return False

    def __del__(self):
        """Backup cleanup in case close() is not called."""
        try:
            if self._env is not None:
                self._env.close()
        except Exception:
            pass
