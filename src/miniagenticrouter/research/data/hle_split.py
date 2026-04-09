"""Data split management for HLE (Humanity's Last Exam) research experiments.

This module provides the HLEDataSplit class for managing train/val/test splits
of HLE tasks by category.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from miniagenticrouter.research.utils.config import load_config


@dataclass
class HLEDataSplitConfig:
    """Configuration for HLE data splitting.

    Attributes:
        split_seed: Random seed for reproducibility.
        split_version: Version string for the split.
        train_ratio: Fraction of tasks for training within train categories.
        val_ratio: Fraction of tasks for validation within train categories.
        test_id_ratio: Fraction of tasks for ID testing within train categories.
        skip_multimodal: Whether to skip tasks with images.
    """

    split_seed: int = 42
    split_version: str = "v1.0"
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_id_ratio: float = 0.2
    skip_multimodal: bool = True

    def __post_init__(self) -> None:
        """Validate ratios sum to 1."""
        total = self.train_ratio + self.val_ratio + self.test_id_ratio
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")


class HLEDataSplit:
    """Manage train/val/test splits for HLE research experiments.

    HLE has no variation concept like ScienceWorld. Instead, we split by:
    - Category-level: train_categories vs ood_test_categories
    - Task-level: Within train categories, split tasks into train/val/test_id

    Example:
        >>> split = HLEDataSplit.from_yaml("config/research/hle_data_split.yaml")
        >>> split.initialize_splits()
        >>> tasks = split.enumerate_tasks("train")
        >>> len(tasks)
        1073
    """

    def __init__(
        self,
        train_categories: list[str],
        ood_test_categories: list[str],
        config: HLEDataSplitConfig | None = None,
        models_config: dict[str, Any] | None = None,
    ):
        """Initialize HLEDataSplit.

        Args:
            train_categories: List of category names for training.
            ood_test_categories: List of category names for OOD testing.
            config: Split configuration.
            models_config: Model configurations.
        """
        self.config = config or HLEDataSplitConfig()
        self.train_categories = train_categories
        self.ood_test_categories = ood_test_categories
        self._models_config = models_config or {}
        self._splits: dict[str, list[dict[str, Any]]] = {
            "train": [],
            "val": [],
            "test_id": [],
            "ood_test": [],
        }
        self._initialized = False

    @classmethod
    def from_yaml(
        cls,
        path: Path | str | None = None,
    ) -> HLEDataSplit:
        """Load HLEDataSplit from a YAML configuration file.

        Args:
            path: Path to the YAML file. If None, uses default path.

        Returns:
            HLEDataSplit instance.
        """
        if path is None:
            from miniagenticrouter.research.utils.config import get_default_hle_config_path
            path = get_default_hle_config_path()
        path = Path(path)

        data = load_config(path)

        train_categories = data.get("train_categories", [])
        ood_test_categories = data.get("ood_test_categories", [])

        # Parse configuration
        task_split = data.get("task_split", {})
        config = HLEDataSplitConfig(
            split_seed=data.get("split_seed", 42),
            split_version=data.get("split_version", "v1.0"),
            train_ratio=task_split.get("train_ratio", 0.6),
            val_ratio=task_split.get("val_ratio", 0.2),
            test_id_ratio=task_split.get("test_id_ratio", 0.2),
            skip_multimodal=data.get("skip_multimodal", True),
        )

        # Parse models config
        models_config = data.get("models", {})

        return cls(
            train_categories=train_categories,
            ood_test_categories=ood_test_categories,
            config=config,
            models_config=models_config,
        )

    def initialize_splits(self, tasks: list[dict[str, Any]] | None = None) -> None:
        """Initialize task splits from HuggingFace dataset.

        This method loads the HLE dataset and splits tasks by category.
        Tasks in train_categories are further split into train/val/test_id.
        Tasks in ood_test_categories go to ood_test.

        Args:
            tasks: Optional pre-loaded list of tasks. If None, loads from HuggingFace.
        """
        if self._initialized:
            return

        if tasks is None:
            tasks = self._load_hle_tasks()

        # Separate tasks by category type
        train_category_tasks: list[dict[str, Any]] = []
        ood_test_tasks: list[dict[str, Any]] = []

        for task in tasks:
            category = task.get("category", "")
            if category in self.train_categories:
                train_category_tasks.append(task)
            elif category in self.ood_test_categories:
                ood_test_tasks.append(task)
            # Tasks not in either category are ignored

        # Split train_category_tasks into train/val/test_id
        rng = random.Random(self.config.split_seed)
        rng.shuffle(train_category_tasks)

        n_total = len(train_category_tasks)
        n_train = int(n_total * self.config.train_ratio)
        n_val = int(n_total * self.config.val_ratio)

        self._splits["train"] = train_category_tasks[:n_train]
        self._splits["val"] = train_category_tasks[n_train : n_train + n_val]
        self._splits["test_id"] = train_category_tasks[n_train + n_val :]
        self._splits["ood_test"] = ood_test_tasks

        # Add split field to each task
        for split_name, split_tasks in self._splits.items():
            for task in split_tasks:
                task["split"] = split_name

        self._initialized = True

    def _load_hle_tasks(self) -> list[dict[str, Any]]:
        """Load HLE tasks from HuggingFace.

        Returns:
            List of task dictionaries with category field.
        """
        from miniagenticrouter.data.hle_tasks import load_hle_dataset

        dataset = load_hle_dataset("test")
        tasks = []

        for item in dataset:
            # Skip multimodal tasks if configured
            image = item.get("image", "")
            if self.config.skip_multimodal and image:
                continue

            task_id = item.get("id", "")
            category = item.get("category", "")
            subject = item.get("raw_subject", item.get("subject", ""))

            tasks.append({
                "task_id": task_id,
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "image": image,
                "subject": subject,
                "category": category,
                "question_type": item.get("answer_type", "short_answer"),
                "_original": dict(item),
            })

        return tasks

    def get_train_categories(self) -> list[str]:
        """Get list of training category names."""
        return self.train_categories.copy()

    def get_ood_test_categories(self) -> list[str]:
        """Get list of OOD test category names."""
        return self.ood_test_categories.copy()

    def enumerate_tasks(
        self,
        split: Literal["train", "val", "test_id", "ood_test"],
    ) -> list[dict[str, Any]]:
        """Enumerate all task instances for a given split.

        Args:
            split: Which split to enumerate.

        Returns:
            List of task info dicts with keys:
            - task_id: Unique identifier
            - question: Question text
            - answer: Ground truth answer
            - subject: Subject area
            - category: Category name
            - question_type: Type of question
            - split: The split name
            - _original: Original HLE data
        """
        if not self._initialized:
            self.initialize_splits()

        if split not in self._splits:
            raise ValueError(f"Unknown split: {split}. Valid splits: {list(self._splits.keys())}")

        return [task.copy() for task in self._splits[split]]

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

            model_config: dict[str, Any] = {"model_name": model_name}
            result.append(model_config)

        return result

    def get_split_stats(self) -> dict[str, dict[str, int]]:
        """Get statistics for each split.

        Returns:
            Dictionary mapping split name to category counts.
        """
        if not self._initialized:
            self.initialize_splits()

        stats: dict[str, dict[str, int]] = {}
        for split_name, tasks in self._splits.items():
            category_counts: dict[str, int] = {}
            for task in tasks:
                category = task.get("category", "unknown")
                category_counts[category] = category_counts.get(category, 0) + 1
            stats[split_name] = category_counts

        return stats

    def summary(self) -> str:
        """Generate a summary of the data split.

        Returns:
            Human-readable summary string.
        """
        if not self._initialized:
            self.initialize_splits()

        lines = [
            f"HLEDataSplit (version={self.config.split_version}, seed={self.config.split_seed})",
            f"  Skip multimodal: {self.config.skip_multimodal}",
            "",
            f"Training categories: {len(self.train_categories)}",
        ]
        for cat in self.train_categories:
            lines.append(f"  - {cat}")

        lines.extend([
            "",
            f"  Train tasks: {len(self._splits['train'])}",
            f"  Val tasks: {len(self._splits['val'])}",
            f"  Test-ID tasks: {len(self._splits['test_id'])}",
            "",
            f"OOD test categories: {len(self.ood_test_categories)}",
        ])
        for cat in self.ood_test_categories:
            lines.append(f"  - {cat}")

        lines.append(f"  OOD test tasks: {len(self._splits['ood_test'])}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"HLEDataSplit(train_categories={len(self.train_categories)}, "
            f"ood_test_categories={len(self.ood_test_categories)})"
        )
