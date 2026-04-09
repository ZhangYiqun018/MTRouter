"""ScienceWorld task enumeration utilities."""

from __future__ import annotations

import fnmatch
from typing import Any


def enumerate_scienceworld_tasks(
    task_filter: str | None = None,
    variation_filter: int | None = None,
    simplification_str: str = "",
) -> list[dict[str, Any]]:
    """Enumerate ScienceWorld tasks and variations.

    Args:
        task_filter: Glob pattern to filter task names (e.g., "boil*", "*melt*").
        variation_filter: If set, only include this specific variation index.
        simplification_str: Simplification string to pass to ScienceWorld.

    Returns:
        List of task dictionaries with keys:
        - task_id: Unique identifier (e.g., "boil_var0")
        - task_name: Name of the task
        - variation_idx: Variation index
        - simplification_str: Simplification string

    Example:
        >>> tasks = enumerate_scienceworld_tasks(task_filter="boil*")
        >>> len(tasks) > 0
        True
        >>> tasks[0]["task_name"].startswith("boil")
        True
    """
    try:
        from scienceworld import ScienceWorldEnv
    except ImportError as e:
        raise ImportError(
            "ScienceWorld is not installed. Install from a repo checkout with: pip install -e '.[scienceworld]'"
        ) from e

    # Create temporary environment to enumerate tasks
    env = ScienceWorldEnv("", serverPath=None)

    tasks = []
    try:
        for task_name in env.get_task_names():
            # Apply task filter
            if task_filter and not fnmatch.fnmatch(task_name, task_filter):
                continue

            # Get number of variations for this task
            num_variations = env.get_max_variations(task_name)

            for var_idx in range(num_variations):
                # Apply variation filter
                if variation_filter is not None and var_idx != variation_filter:
                    continue

                task_id = f"{task_name}_var{var_idx}"
                tasks.append({
                    "task_id": task_id,
                    "task_name": task_name,
                    "variation_idx": var_idx,
                    "simplification_str": simplification_str,
                })
    finally:
        env.close()

    return tasks


def get_scienceworld_task_names() -> list[str]:
    """Get all available ScienceWorld task names.

    Returns:
        List of task names.
    """
    try:
        from scienceworld import ScienceWorldEnv
    except ImportError as e:
        raise ImportError(
            "ScienceWorld is not installed. Install from a repo checkout with: pip install -e '.[scienceworld]'"
        ) from e

    env = ScienceWorldEnv("", serverPath=None)
    try:
        return list(env.get_task_names())
    finally:
        env.close()


def get_scienceworld_task_info(task_name: str, simplification_str: str = "") -> dict[str, Any]:
    """Get information about a specific ScienceWorld task.

    Args:
        task_name: Name of the task.
        simplification_str: Simplification string.

    Returns:
        Dictionary with task information:
        - task_name: Name of the task
        - num_variations: Number of variations
        - simplification_str: Simplification string
    """
    try:
        from scienceworld import ScienceWorldEnv
    except ImportError as e:
        raise ImportError(
            "ScienceWorld is not installed. Install from a repo checkout with: pip install -e '.[scienceworld]'"
        ) from e

    env = ScienceWorldEnv("", serverPath=None)
    try:
        num_variations = env.get_max_variations(task_name)
        return {
            "task_name": task_name,
            "num_variations": num_variations,
            "simplification_str": simplification_str,
        }
    finally:
        env.close()
