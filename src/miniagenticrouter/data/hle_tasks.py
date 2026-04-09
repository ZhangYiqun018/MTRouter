"""HLE (Humanity's Last Exam) task enumeration utilities."""

from __future__ import annotations

import fnmatch
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def load_hle_dataset(split: str = "test") -> tuple[dict, ...]:
    """Load HLE dataset from Hugging Face.

    Args:
        split: Dataset split to load (default: "test").

    Returns:
        Tuple of task dictionaries (cached for efficiency).

    Raises:
        ImportError: If datasets library is not installed.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "datasets library is not installed. Install with: pip install datasets"
        ) from e

    dataset = load_dataset("cais/hle", split=split)
    # Convert to tuple for caching (lists are not hashable)
    return tuple(dict(item) for item in dataset)


def enumerate_hle_tasks(
    subject_filter: str | None = None,
    slice_spec: str | None = None,
    ids: list[str] | None = None,
    skip_multimodal: bool = True,
    split: str = "test",
) -> list[dict[str, Any]]:
    """Enumerate HLE tasks with filtering options.

    Args:
        subject_filter: Glob pattern to filter by subject (e.g., "math*", "*physics*").
        slice_spec: Slice specification (e.g., "0:100", "50:150", ":10").
        ids: List of specific task IDs to include.
        skip_multimodal: If True, skip tasks with images (default: True).
        split: Dataset split to use (default: "test").

    Returns:
        List of task dictionaries with keys:
        - task_id: Unique identifier (from dataset "id" field)
        - question: Question text
        - answer: Ground truth answer
        - image: Image URL (empty string if no image)
        - subject: Subject category (if available)
        - question_type: "multiple_choice" or "short_answer" (inferred)

    Example:
        >>> tasks = enumerate_hle_tasks(slice_spec="0:10")
        >>> len(tasks) <= 10
        True
    """
    dataset = load_hle_dataset(split)
    tasks = []

    for item in dataset:
        task_id = item.get("id", item.get("question_id", str(len(tasks))))
        image = item.get("image", "")

        # Skip multimodal tasks if requested
        if skip_multimodal and image:
            continue

        # Apply ID filter
        if ids is not None and task_id not in ids:
            continue

        # Apply subject filter
        subject = item.get("subject", item.get("category", ""))
        if subject_filter and not fnmatch.fnmatch(subject.lower(), subject_filter.lower()):
            continue

        # Infer question type
        answer = item.get("answer", "")
        question_type = _infer_question_type(item)

        tasks.append({
            "task_id": task_id,
            "question": item.get("question", ""),
            "answer": answer,
            "image": image,
            "subject": subject,
            "question_type": question_type,
            # Preserve original fields for benchmark_data
            "_original": item,
        })

    # Apply slice
    if slice_spec:
        tasks = _apply_slice(tasks, slice_spec)

    return tasks


def get_hle_task_by_id(task_id: str, split: str = "test") -> dict[str, Any] | None:
    """Get a specific HLE task by ID.

    Args:
        task_id: The task ID to retrieve.
        split: Dataset split to use (default: "test").

    Returns:
        Task dictionary or None if not found.
    """
    dataset = load_hle_dataset(split)

    for item in dataset:
        item_id = item.get("id", item.get("question_id", ""))
        if item_id == task_id:
            return {
                "task_id": item_id,
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "image": item.get("image", ""),
                "subject": item.get("subject", item.get("category", "")),
                "question_type": _infer_question_type(item),
                "_original": item,
            }

    return None


def get_hle_stats(skip_multimodal: bool = True, split: str = "test") -> dict[str, Any]:
    """Get statistics about the HLE dataset.

    Args:
        skip_multimodal: If True, exclude multimodal tasks from stats.
        split: Dataset split to use (default: "test").

    Returns:
        Dictionary with statistics:
        - total: Total number of tasks
        - multimodal: Number of tasks with images
        - text_only: Number of text-only tasks
        - subjects: Dictionary of subject counts
        - question_types: Dictionary of question type counts
    """
    dataset = load_hle_dataset(split)

    stats = {
        "total": len(dataset),
        "multimodal": 0,
        "text_only": 0,
        "subjects": {},
        "question_types": {"multiple_choice": 0, "short_answer": 0},
    }

    for item in dataset:
        image = item.get("image", "")

        if image:
            stats["multimodal"] += 1
            if skip_multimodal:
                continue
        else:
            stats["text_only"] += 1

        # Count subjects
        subject = item.get("subject", item.get("category", "unknown"))
        stats["subjects"][subject] = stats["subjects"].get(subject, 0) + 1

        # Count question types
        q_type = _infer_question_type(item)
        stats["question_types"][q_type] += 1

    return stats


def _infer_question_type(item: dict) -> str:
    """Infer question type from item fields.

    Returns "multiple_choice" if options/choices are present, otherwise "short_answer".
    """
    # Check for multiple choice indicators
    if item.get("options") or item.get("choices"):
        return "multiple_choice"

    # Check if answer looks like a choice letter
    answer = str(item.get("answer", "")).strip()
    if len(answer) == 1 and answer.upper() in "ABCDEFGH":
        return "multiple_choice"

    return "short_answer"


def _apply_slice(tasks: list[dict], slice_spec: str) -> list[dict]:
    """Apply slice specification to task list.

    Args:
        tasks: List of tasks.
        slice_spec: Slice specification (e.g., "0:100", ":10", "50:").

    Returns:
        Sliced task list.
    """
    parts = slice_spec.split(":")
    if len(parts) == 1:
        # Single index
        idx = int(parts[0])
        return [tasks[idx]] if 0 <= idx < len(tasks) else []
    elif len(parts) == 2:
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else len(tasks)
        return tasks[start:end]
    else:
        raise ValueError(f"Invalid slice specification: {slice_spec}")


def list_hle_subjects(split: str = "test") -> list[str]:
    """List all unique subjects in the HLE dataset.

    Args:
        split: Dataset split to use (default: "test").

    Returns:
        Sorted list of unique subjects.
    """
    dataset = load_hle_dataset(split)
    subjects = set()

    for item in dataset:
        subject = item.get("subject", item.get("category", ""))
        if subject:
            subjects.add(subject)

    return sorted(subjects)
