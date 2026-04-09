"""Data split management for research experiments.

This module provides the DataSplit class for managing train/val/test splits
of ScienceWorld tasks and variations.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from miniagenticrouter.research.utils.config import get_default_config_path, load_config


@dataclass
class VariationSplit:
    """Variation-level split for a single task.

    Attributes:
        task_name: Name of the task.
        train_variations: Variation indices for training.
        val_variations: Variation indices for validation.
        test_id_variations: Variation indices for in-distribution testing.
    """

    task_name: str
    train_variations: list[int]
    val_variations: list[int]
    test_id_variations: list[int]

    @property
    def all_variations(self) -> list[int]:
        """Get all variations."""
        return self.train_variations + self.val_variations + self.test_id_variations


@dataclass
class DataSplitConfig:
    """Configuration for data splitting.

    Attributes:
        split_seed: Random seed for reproducibility.
        split_version: Version string for the split.
        train_ratio: Fraction of variations for training.
        val_ratio: Fraction of variations for validation.
        test_id_ratio: Fraction of variations for ID testing.
        max_variations_per_task: Maximum variations to use per task.
        min_variations_per_task: Minimum variations required per task.
    """

    split_seed: int = 42
    split_version: str = "v1.0"
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_id_ratio: float = 0.2
    max_variations_per_task: int = 30
    min_variations_per_task: int = 10

    def __post_init__(self) -> None:
        """Validate ratios sum to 1."""
        total = self.train_ratio + self.val_ratio + self.test_id_ratio
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")


class DataSplit:
    """Manage train/val/test splits for research experiments.

    This class reads from data_split.yaml and provides:
    - Task-level splits (train tasks vs OOD test tasks)
    - Variation-level splits within training tasks

    Example:
        >>> split = DataSplit.from_yaml("config/research/data_split.yaml")
        >>> split.get_train_tasks()
        ['boil', 'melt', 'chemistry-mix', ...]
        >>> split.get_variations("boil", "train")
        [0, 1, 2, 3, 4, 5]
        >>> tasks = split.enumerate_tasks("train")
        >>> len(tasks)
        162
    """

    def __init__(
        self,
        train_tasks: list[str],
        ood_test_tasks: list[str],
        config: DataSplitConfig | None = None,
        models_config: dict[str, Any] | None = None,
        use_first_try: bool = False,
    ):
        """Initialize DataSplit.

        Args:
            train_tasks: List of training task names.
            ood_test_tasks: List of OOD test task names.
            config: Split configuration.
            models_config: Model tier configurations.
            use_first_try: Whether to use first_try subset.
        """
        self.config = config or DataSplitConfig()
        self.train_tasks = train_tasks
        self.ood_test_tasks = ood_test_tasks
        self._models_config = models_config or {}
        self._use_first_try = use_first_try
        self._variation_splits: dict[str, VariationSplit] = {}
        self._initialized = False

    @classmethod
    def from_yaml(
        cls,
        path: Path | str | None = None,
        use_first_try: bool = False,
    ) -> DataSplit:
        """Load DataSplit from a YAML configuration file.

        Args:
            path: Path to the YAML file. If None, uses default path.
            use_first_try: Whether to use first_try subset.

        Returns:
            DataSplit instance.
        """
        if path is None:
            path = get_default_config_path()
        path = Path(path)

        data = load_config(path)

        # Determine which task lists to use
        if use_first_try and "first_try" in data:
            train_tasks = data["first_try"].get("train_tasks", [])
            ood_test_tasks = data["first_try"].get("ood_test_tasks", [])
        else:
            train_tasks = data.get("train_tasks", [])
            ood_test_tasks = data.get("ood_test_tasks", [])

        # Parse configuration
        variation_split = data.get("variation_split", {})
        config = DataSplitConfig(
            split_seed=data.get("split_seed", 42),
            split_version=data.get("split_version", "v1.0"),
            train_ratio=variation_split.get("train_ratio", 0.6),
            val_ratio=variation_split.get("val_ratio", 0.2),
            test_id_ratio=variation_split.get("test_id_ratio", 0.2),
            max_variations_per_task=variation_split.get("max_variations_per_task", 30),
            min_variations_per_task=variation_split.get("min_variations_per_task", 10),
        )

        # Parse models config
        models_config = data.get("models", {})

        return cls(
            train_tasks=train_tasks,
            ood_test_tasks=ood_test_tasks,
            config=config,
            models_config=models_config,
            use_first_try=use_first_try,
        )

    def initialize_variations(self, task_variations: dict[str, int] | None = None) -> None:
        """Initialize variation splits for all training tasks.

        This method computes the train/val/test_id split for each task's
        variations. It can either accept a pre-computed dict of task variations
        or query the ScienceWorld environment.

        Args:
            task_variations: Optional dict mapping task names to number of
                variations. If None, queries ScienceWorld.
        """
        if self._initialized:
            return

        if task_variations is None:
            task_variations = self._get_task_variations_from_env()

        rng = random.Random(self.config.split_seed)

        for task_name in self.train_tasks:
            if task_name not in task_variations:
                raise ValueError(f"Task '{task_name}' not found in task_variations")

            n_vars = task_variations[task_name]

            # Cap at max_variations_per_task
            n_vars = min(n_vars, self.config.max_variations_per_task)

            # Skip if too few variations
            if n_vars < self.config.min_variations_per_task:
                continue

            # Create shuffled list of variation indices
            variations = list(range(n_vars))
            rng.shuffle(variations)

            # Split according to ratios
            n_train = int(n_vars * self.config.train_ratio)
            n_val = int(n_vars * self.config.val_ratio)

            train_vars = sorted(variations[:n_train])
            val_vars = sorted(variations[n_train : n_train + n_val])
            test_id_vars = sorted(variations[n_train + n_val :])

            self._variation_splits[task_name] = VariationSplit(
                task_name=task_name,
                train_variations=train_vars,
                val_variations=val_vars,
                test_id_variations=test_id_vars,
            )

        self._initialized = True

    def _get_task_variations_from_env(self) -> dict[str, int]:
        """Query ScienceWorld for task variation counts.

        Returns:
            Dict mapping task names to number of variations.
        """
        try:
            from scienceworld import ScienceWorldEnv
        except ImportError:
            raise ImportError(
                "ScienceWorld is not installed. Either install it with "
                "'pip install scienceworld' or provide task_variations dict."
            )

        env = ScienceWorldEnv("", serverPath=None)
        try:
            result = {}
            for task_name in env.get_task_names():
                result[task_name] = env.get_max_variations(task_name)
            return result
        finally:
            env.close()

    def get_train_tasks(self) -> list[str]:
        """Get list of training task names."""
        return self.train_tasks.copy()

    def get_ood_test_tasks(self) -> list[str]:
        """Get list of OOD test task names."""
        return self.ood_test_tasks.copy()

    def get_variations(
        self,
        task_name: str,
        split: Literal["train", "val", "test_id", "all"],
    ) -> list[int]:
        """Get variation indices for a task and split.

        Args:
            task_name: Name of the task.
            split: Which split to get variations for.

        Returns:
            List of variation indices.

        Raises:
            ValueError: If task is not in training tasks.
        """
        if not self._initialized:
            self.initialize_variations()

        if task_name not in self._variation_splits:
            if task_name in self.ood_test_tasks:
                raise ValueError(
                    f"Task '{task_name}' is an OOD test task. Use enumerate_tasks('ood_test')."
                )
            raise ValueError(f"Task '{task_name}' not found in training tasks.")

        vs = self._variation_splits[task_name]

        if split == "train":
            return vs.train_variations.copy()
        elif split == "val":
            return vs.val_variations.copy()
        elif split == "test_id":
            return vs.test_id_variations.copy()
        elif split == "all":
            return vs.all_variations.copy()
        else:
            raise ValueError(f"Unknown split: {split}")

    def enumerate_tasks(
        self,
        split: Literal["train", "val", "test_id", "ood_test"],
        simplification_str: str = "",
    ) -> list[dict[str, Any]]:
        """Enumerate all task instances for a given split.

        Returns a list compatible with enumerate_scienceworld_tasks().

        Args:
            split: Which split to enumerate.
            simplification_str: Simplification string for ScienceWorld.

        Returns:
            List of task info dicts with keys:
            - task_id: Unique identifier (e.g., "boil_var0")
            - task_name: Task name
            - variation_idx: Variation index
            - simplification_str: Simplification string
            - split: The split name
        """
        if not self._initialized:
            self.initialize_variations()

        tasks = []

        if split == "ood_test":
            # For OOD test, we need to get variations from the environment
            task_variations = self._get_ood_variations()
            for task_name in self.ood_test_tasks:
                n_vars = task_variations.get(task_name, 0)
                n_vars = min(n_vars, self.config.max_variations_per_task)
                for var_idx in range(n_vars):
                    task_id = f"{task_name}_var{var_idx}"
                    tasks.append(
                        {
                            "task_id": task_id,
                            "task_name": task_name,
                            "variation_idx": var_idx,
                            "simplification_str": simplification_str,
                            "split": split,
                        }
                    )
        else:
            # For train/val/test_id, use the computed splits
            for task_name in self.train_tasks:
                if task_name not in self._variation_splits:
                    continue
                variations = self.get_variations(task_name, split)
                for var_idx in variations:
                    task_id = f"{task_name}_var{var_idx}"
                    tasks.append(
                        {
                            "task_id": task_id,
                            "task_name": task_name,
                            "variation_idx": var_idx,
                            "simplification_str": simplification_str,
                            "split": split,
                        }
                    )

        return tasks

    def _get_ood_variations(self) -> dict[str, int]:
        """Get variation counts for OOD test tasks.

        Returns:
            Dict mapping OOD task names to number of variations.
        """
        try:
            from scienceworld import ScienceWorldEnv
        except ImportError:
            # Fallback: assume max_variations_per_task
            return {t: self.config.max_variations_per_task for t in self.ood_test_tasks}

        env = ScienceWorldEnv("", serverPath=None)
        try:
            result = {}
            for task_name in self.ood_test_tasks:
                result[task_name] = env.get_max_variations(task_name)
            return result
        finally:
            env.close()

    def get_model_configs(self) -> list[dict[str, Any]]:
        """Get model configurations for roulette/baseline data collection.

        Returns:
            List of model config dicts with 'model_name'.
        """
        pool_config = self._models_config.get("roulette_model_pool", [])
        if not isinstance(pool_config, list):
            return []

        result = []
        for item in pool_config:
            if isinstance(item, str):
                model_name = item
            elif isinstance(item, dict):
                model_name = item.get("model_id") or item.get("model_name", "")
            else:
                continue

            # Just pass the model name - get_model() will read the full config
            # from custom_models.yaml including the correct custom_llm_provider
            model_config: dict[str, Any] = {"model_name": model_name}
            result.append(model_config)

        return result

    def summary(self) -> str:
        """Generate a summary of the data split.

        Returns:
            Human-readable summary string.
        """
        if not self._initialized:
            self.initialize_variations()

        lines = [
            f"DataSplit (version={self.config.split_version}, seed={self.config.split_seed})",
            f"  First-try mode: {self._use_first_try}",
            "",
            f"Training tasks: {len(self.train_tasks)}",
        ]

        total_train = total_val = total_test = 0
        for task_name in self.train_tasks:
            if task_name in self._variation_splits:
                vs = self._variation_splits[task_name]
                total_train += len(vs.train_variations)
                total_val += len(vs.val_variations)
                total_test += len(vs.test_id_variations)

        lines.extend(
            [
                f"  Train variations: {total_train}",
                f"  Val variations: {total_val}",
                f"  Test-ID variations: {total_test}",
                "",
                f"OOD test tasks: {len(self.ood_test_tasks)}",
            ]
        )

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"DataSplit(train_tasks={len(self.train_tasks)}, "
            f"ood_test_tasks={len(self.ood_test_tasks)}, "
            f"first_try={self._use_first_try})"
        )
