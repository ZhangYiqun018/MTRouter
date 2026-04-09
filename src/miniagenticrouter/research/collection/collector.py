"""Data collector for research experiments.

This module provides the Collector class for systematic trajectory collection
across different data splits and collection modes.

This is a thin wrapper around the existing `process_task` function from
run/extra/scienceworld.py, adding research-specific features:
- DataSplit integration for train/val/test splits
- CollectionMode for model selection (baseline, roulette, etc.)
- runs_per_variation for multiple runs
- skip_existing for incremental collection
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from miniagenticrouter.research.collection.modes import CollectionMode
    from miniagenticrouter.research.data.split import DataSplit


@dataclass
class CollectionConfig:
    """Configuration for data collection.

    Attributes:
        output_dir: Base directory for saving trajectories.
        runs_per_variation: Number of runs per task variation.
        step_limit: Maximum steps per episode.
        cost_limit: Maximum cost (USD) per episode.
        workers: Number of parallel workers.
        record_propensity: Whether to record selection probabilities.
        simplification_str: Simplification string for ScienceWorld.
        agent_config: Optional dict of agent configuration (templates, etc.).
        skip_existing: Whether to skip already collected trajectories.
    """

    output_dir: Path
    runs_per_variation: int = 3
    step_limit: int = 50
    cost_limit: float = 5.0
    workers: int = 4
    record_propensity: bool = False
    simplification_str: str = ""
    agent_config: dict | None = None
    skip_existing: bool = True

    def __post_init__(self) -> None:
        """Ensure output_dir is a Path."""
        self.output_dir = Path(self.output_dir)


@dataclass
class CollectionResult:
    """Result of a single trajectory collection.

    Attributes:
        task_id: Task identifier.
        run_id: Run number (0-indexed).
        traj_path: Path to the saved trajectory file.
        final_score: Final environment score.
        total_cost: Total cost (USD) for the episode.
        n_steps: Number of steps taken.
        exit_status: Exit status string.
        model_usage: Dict mapping model names to call counts.
        error: Error message if collection failed.
    """

    task_id: str
    run_id: int
    traj_path: Path | None
    final_score: float
    total_cost: float
    n_steps: int
    exit_status: str
    model_usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    @property
    def success(self) -> bool:
        """Whether collection succeeded."""
        return self.error is None


class Collector:
    """Systematic data collector for research experiments.

    This class coordinates trajectory collection across:
    - Different data splits (train/val/test)
    - Different collection modes (baseline/roulette)
    - Multiple runs per variation

    It delegates the actual task execution to `process_task` from
    run/extra/scienceworld.py, adding research-specific features.

    Example:
        >>> split = DataSplit.from_yaml("config/research/data_split.yaml")
        >>> mode = RouletteMode(record_propensity=True)
        >>> config = CollectionConfig(output_dir=Path("./trajectories"))
        >>> collector = Collector(data_split=split, mode=mode, config=config)
        >>> results = collector.collect(split="train")
    """

    def __init__(
        self,
        data_split: DataSplit,
        mode: CollectionMode,
        config: CollectionConfig,
    ):
        """Initialize Collector.

        Args:
            data_split: Data split configuration.
            mode: Collection mode (BaselineMode, RouletteMode, etc.).
            config: Collection configuration.
        """
        self.data_split = data_split
        self.mode = mode
        self.config = config
        self._model_configs = data_split.get_model_configs()
        self._last_skipped: int = 0

    def _get_agent_config(self) -> dict:
        """Get agent configuration (templates, parser, etc.).

        Returns config from CollectionConfig.agent_config if provided,
        otherwise loads from default scienceworld.yaml.
        """
        if self.config.agent_config is not None:
            agent_config = dict(self.config.agent_config)
        else:
            # Load default ScienceWorld config
            import yaml

            from miniagenticrouter.config import builtin_config_dir

            config_path = builtin_config_dir / "extra" / "scienceworld.yaml"
            with open(config_path) as f:
                full_config = yaml.safe_load(f)
            agent_config = dict(full_config.get("agent", {}))

        # Remove class spec from kwargs (not a FlexibleAgent parameter)
        agent_config.pop("agent_class", None)
        # Set step/cost limits from CollectionConfig
        agent_config["step_limit"] = self.config.step_limit
        agent_config["cost_limit"] = self.config.cost_limit

        return agent_config

    def _build_process_task_config(self) -> dict:
        """Build config dict for process_task.

        Returns:
            Config dict compatible with process_task.
        """
        return {"agent": self._get_agent_config()}

    def _get_config_signature(self) -> str:
        """Generate config signature for skip-existing validation.

        The signature includes parameters that affect trajectory results.
        If the signature changes, old trajectories should be re-run.

        Returns:
            8-character hex signature.
        """
        import hashlib

        sig_data = {
            "mode": self.mode.get_mode_name(),
            "models": self.mode.get_model_names(self._model_configs),
            "step_limit": self.config.step_limit,
            "cost_limit": self.config.cost_limit,
        }
        sig_json = json.dumps(sig_data, sort_keys=True)
        return hashlib.md5(sig_json.encode()).hexdigest()[:8]

    def collect(
        self,
        split: Literal["train", "val", "test_id", "ood_test"] = "train",
        task_filter: str | None = None,
        dry_run: bool = False,
    ) -> list[CollectionResult]:
        """Collect trajectories for a data split.

        Args:
            split: Which split to collect for.
            task_filter: Optional glob pattern to filter tasks.
            dry_run: If True, only print what would be done.

        Returns:
            List of collection results.

        Note:
            This method assumes single-writer access to output_dir.
            Running multiple collect processes to the same output_dir
            concurrently may cause race conditions.
        """
        import fnmatch

        from rich.live import Live

        from miniagenticrouter.run.extra.scienceworld import process_task
        from miniagenticrouter.run.extra.utils.batch_progress import (
            RunBatchProgressManager,
        )

        # Get tasks for this split
        tasks = self.data_split.enumerate_tasks(
            split=split,
            simplification_str=self.config.simplification_str,
        )

        # Apply task filter
        if task_filter:
            tasks = [
                t for t in tasks if fnmatch.fnmatch(t["task_name"], task_filter)
            ]

        if dry_run:
            self._print_collection_plan(tasks)
            return []

        # Generate all (task, run_id) pairs, expanding runs_per_variation
        work_items = []
        skipped = 0
        for task in tasks:
            for run_id in range(self.config.runs_per_variation):
                # Skip if already exists and is valid
                if self.config.skip_existing:
                    output_path = self.get_output_path(task["task_id"], run_id)
                    if self._is_valid_trajectory(output_path):
                        skipped += 1
                        continue

                # Create task_info with run_id in task_id
                task_with_run = dict(task)
                task_with_run["original_task_id"] = task["task_id"]
                task_with_run["run_id"] = run_id
                task_with_run["task_id"] = f"{task['task_id']}_run{run_id}"
                work_items.append(task_with_run)

        if skipped > 0:
            print(f"Skipping {skipped} already collected trajectories")

        # Store for summary
        self._last_skipped = skipped

        if not work_items:
            print("No tasks to collect (all already done)")
            return []

        # Build config for process_task
        config = self._build_process_task_config()

        # Compute output directory with mode subdirectory
        mode_name = self.mode.get_output_subdir()
        output_dir = self.config.output_dir / mode_name

        # Create progress manager
        progress_manager = RunBatchProgressManager(
            num_instances=len(work_items),
            yaml_report_path=output_dir / f"collection_report_{split}.yaml",
        )

        results = []

        # Pre-compute config signature for all tasks
        config_signature = self._get_config_signature()

        def process_one(task_info: dict) -> CollectionResult:
            """Process a single task and return CollectionResult."""
            task_id = task_info["task_id"]
            original_task_id = task_info["original_task_id"]
            run_id = task_info["run_id"]

            # Create fresh model for each task (thread-safe)
            model = self.mode.create_model_or_router(self._model_configs)

            try:
                result_data = process_task(
                    task_info=task_info,
                    output_dir=output_dir,
                    config=config,
                    progress_manager=progress_manager,
                    model=model,
                    extra_info_override={"config_signature": config_signature},
                )

                # Get model usage
                model_usage = {}
                if hasattr(model, "models"):
                    for m in model.models:
                        if m.n_calls > 0:
                            model_usage[m.config.model_name] = m.n_calls
                else:
                    model_usage[model.config.model_name] = model.n_calls

                return CollectionResult(
                    task_id=original_task_id,
                    run_id=run_id,
                    traj_path=output_dir / task_id / f"{task_id}.traj.json",
                    final_score=result_data["score"],
                    total_cost=result_data["cost"],
                    n_steps=result_data["steps"],
                    exit_status=result_data["exit_status"],
                    model_usage=model_usage,
                )
            except Exception as e:
                # Still collect model usage even on error
                error_model_usage = {}
                if model and hasattr(model, "models"):
                    for m in model.models:
                        if m.n_calls > 0:
                            error_model_usage[m.config.model_name] = m.n_calls
                elif model and hasattr(model, "config"):
                    error_model_usage[model.config.model_name] = model.n_calls

                return CollectionResult(
                    task_id=original_task_id,
                    run_id=run_id,
                    traj_path=None,
                    final_score=0.0,
                    total_cost=model.cost if model else 0.0,
                    n_steps=model.n_calls if model else 0,
                    exit_status="Error",
                    error=str(e),
                    model_usage=error_model_usage,
                )

        # Execute in parallel
        try:
            with Live(progress_manager.render_group, refresh_per_second=4):
                with ThreadPoolExecutor(max_workers=self.config.workers) as executor:
                    futures = {
                        executor.submit(process_one, task_info): task_info
                        for task_info in work_items
                    }

                    for future in as_completed(futures):
                        task_info = futures[future]
                        try:
                            result = future.result()
                        except Exception as e:
                            result = CollectionResult(
                                task_id=task_info["original_task_id"],
                                run_id=task_info["run_id"],
                                traj_path=None,
                                final_score=0.0,
                                total_cost=0.0,
                                n_steps=0,
                                exit_status="Error",
                                error=str(e),
                            )

                        # Update model distribution display (for multi-model routers)
                        if result.model_usage:
                            progress_manager.update_model_usage(result.model_usage)

                        results.append(result)
        finally:
            # Shutdown mode resources (e.g., batched inference worker)
            if hasattr(self.mode, "shutdown"):
                self.mode.shutdown()

        return results

    def get_output_path(self, task_id: str, run_id: int) -> Path:
        """Get the output path for a trajectory.

        Format: {output_dir}/{mode}/{task_id}_run{run_id}/{task_id}_run{run_id}.traj.json

        Args:
            task_id: Task identifier.
            run_id: Run number.

        Returns:
            Path to the trajectory file.
        """
        mode_name = self.mode.get_output_subdir()
        instance_id = f"{task_id}_run{run_id}"
        return (
            self.config.output_dir
            / mode_name
            / instance_id
            / f"{instance_id}.traj.json"
        )

    # Exit statuses that indicate a completed run (should skip on resume)
    # All other statuses (APIError, InternalServerError, etc.) should be retried
    COMPLETED_EXIT_STATUSES = {"Submitted", "ForcedAnswer", "LimitsExceeded"}

    def _is_valid_trajectory(self, path: Path) -> bool:
        """Check if a trajectory file exists and is valid.

        A valid trajectory file must:
        - Exist and be parseable as JSON
        - Have a completed exit_status (Submitted, ForcedAnswer, LimitsExceeded)
        - Have matching config_signature

        Trajectories with error exit_status (APIError, InternalServerError, etc.)
        are considered invalid and will be retried.

        Args:
            path: Path to the trajectory file.

        Returns:
            True if valid, False otherwise.
        """
        if not path.exists():
            return False

        try:
            with open(path) as f:
                data = json.load(f)
            # Check for completed exit_status
            info = data.get("info", {})
            exit_status = info.get("exit_status")
            if exit_status not in self.COMPLETED_EXIT_STATUSES:
                return False
            # Check config signature - missing signature means invalid (needs re-run)
            if "config_signature" not in info:
                return False
            return info["config_signature"] == self._get_config_signature()
        except (json.JSONDecodeError, OSError):
            return False

    def _print_collection_plan(self, tasks: list[dict]) -> None:
        """Print what would be collected (dry run).

        Args:
            tasks: List of task info dicts.
        """
        n_tasks = len(tasks)
        n_runs = self.config.runs_per_variation
        n_total = n_tasks * n_runs

        print(f"Collection Plan (dry run):")
        print(f"  Mode: {self.mode.get_mode_name()}")
        print(f"  Output dir: {self.config.output_dir}")
        print(f"  Tasks: {n_tasks}")
        print(f"  Runs per variation: {n_runs}")
        print(f"  Total trajectories: {n_total}")
        print(f"  Workers: {self.config.workers}")
        print()
        print(f"First 10 tasks:")
        for task in tasks[:10]:
            print(f"  - {task['task_id']}")
        if n_tasks > 10:
            print(f"  ... and {n_tasks - 10} more")

    def summary(self, results: list[CollectionResult]) -> str:
        """Generate a summary of collection results.

        Args:
            results: List of collection results.

        Returns:
            Human-readable summary string.
        """
        n_total = len(results)
        n_success = sum(1 for r in results if r.success)
        n_failed = n_total - n_success

        total_cost = sum(r.total_cost for r in results)
        # avg_score and avg_steps: all tasks (LimitsExceeded tasks also have scores and steps)
        avg_score = (
            sum(r.final_score for r in results) / n_total
            if n_total > 0
            else 0.0
        )
        avg_steps = (
            sum(r.n_steps for r in results) / n_total
            if n_total > 0
            else 0.0
        )

        lines = [
            f"Collection Summary:",
            f"  Skipped (existing): {self._last_skipped}",
            f"  Collected: {n_total}",
            f"  Success: {n_success}",
            f"  Failed: {n_failed}",
            f"  Total cost: ${total_cost:.2f}",
            f"  Avg score: {avg_score:.2f}",
            f"  Avg steps: {avg_steps:.1f}",
        ]

        # Model usage distribution
        model_totals: dict[str, int] = {}
        for r in results:
            for model, count in r.model_usage.items():
                model_totals[model] = model_totals.get(model, 0) + count

        if model_totals:
            lines.append("  Model usage:")
            for model, count in sorted(model_totals.items()):
                lines.append(f"    {model}: {count}")

        return "\n".join(lines)

    def _load_results_from_json(self, results_path: Path) -> list[CollectionResult]:
        """Load all results from results.json file.

        Converts the JSON format to CollectionResult objects.

        Args:
            results_path: Path to results.json file.

        Returns:
            List of CollectionResult objects.
        """
        if not results_path.exists():
            return []

        try:
            with open(results_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

        results = []
        for task_id, info in data.items():
            # Parse run_id from task_id (e.g., "boil_var0_run1" -> 1)
            run_id = 0
            match = re.search(r"_run(\d+)$", task_id)
            if match:
                run_id = int(match.group(1))
                # Extract original task_id without _runN suffix
                original_task_id = task_id[: match.start()]
            else:
                original_task_id = task_id

            results.append(
                CollectionResult(
                    task_id=original_task_id,
                    run_id=run_id,
                    traj_path=results_path.parent / task_id / f"{task_id}.traj.json",
                    final_score=info.get("score", 0.0),
                    total_cost=info.get("cost", 0.0),
                    n_steps=info.get("steps", 0),
                    exit_status=info.get("exit_status", "Unknown"),
                    model_usage=info.get("model_usage", {}),
                    error=None if info.get("done") else "Not completed",
                )
            )

        return results

    def save_summary(
        self,
        results: list[CollectionResult],
        split: str,
        output_path: Path | None = None,
    ) -> Path:
        """Save collection summary to JSON file.

        This method loads ALL results from results.json (not just newly collected ones)
        to ensure the summary reflects the complete state.

        Args:
            results: List of newly collected results (used for skipped count only).
            split: Data split name (train, val, test_id, ood_test).
            output_path: Optional custom output path. If None, uses default.

        Returns:
            Path to the saved summary file.
        """
        if output_path is None:
            mode_name = self.mode.get_output_subdir()
            output_path = self.config.output_dir / mode_name / f"summary_{split}.json"

        # Load ALL results from results.json (not just newly collected ones)
        # This ensures the summary reflects the complete state including previously
        # collected trajectories that were skipped in this run
        results_json_path = output_path.parent / "results.json"
        all_results = self._load_results_from_json(results_json_path)

        # If we have results from JSON, use those; otherwise fall back to passed results
        if all_results:
            results = all_results

        # Helper to compute group stats
        def compute_stats(group: list[CollectionResult]) -> dict:
            total = len(group)
            success = sum(1 for r in group if r.success)
            # avg_score and avg_steps: all tasks (LimitsExceeded tasks also have scores and steps)
            all_scores = [r.final_score for r in group]
            all_steps = [r.n_steps for r in group]
            # avg_steps_success: only successful tasks
            success_steps = [r.n_steps for r in group if r.success]
            return {
                "total": total,
                "success": success,
                "failed": total - success,
                "success_rate": round(success / total, 4) if total > 0 else 0.0,
                "total_cost": round(sum(r.total_cost for r in group), 6),
                "avg_score": round(sum(all_scores) / len(all_scores), 2) if all_scores else 0.0,
                "avg_steps": round(sum(all_steps) / len(all_steps), 1) if all_steps else 0.0,
                "avg_steps_success": round(sum(success_steps) / len(success_steps), 1) if success_steps else 0.0,
            }

        # Helper to extract task_name from task_id (e.g., "boil_var16" -> "boil")
        def extract_task_name(task_id: str) -> str:
            match = re.match(r"^(.+?)_var\d+", task_id)
            return match.group(1) if match else task_id

        # Group by run_id
        by_run: dict[int, list[CollectionResult]] = defaultdict(list)
        for r in results:
            by_run[r.run_id].append(r)

        # Group by task_name
        by_task: dict[str, list[CollectionResult]] = defaultdict(list)
        for r in results:
            task_name = extract_task_name(r.task_id)
            by_task[task_name].append(r)

        # Group by exit_status
        by_exit_status: dict[str, int] = defaultdict(int)
        for r in results:
            by_exit_status[r.exit_status] += 1

        # Model usage
        model_usage: dict[str, dict[str, Any]] = defaultdict(lambda: {"calls": 0, "cost": 0.0})
        for r in results:
            for model, count in r.model_usage.items():
                model_usage[model]["calls"] += count
            # Approximate cost per model (assume proportional to calls)
            total_calls = sum(r.model_usage.values()) if r.model_usage else 1
            for model, count in r.model_usage.items():
                model_usage[model]["cost"] += r.total_cost * (count / total_calls)

        # Round model costs
        for model in model_usage:
            model_usage[model]["cost"] = round(model_usage[model]["cost"], 6)

        # Build summary dict
        summary_data = {
            "metadata": {
                "split": split,
                "mode": self.mode.get_mode_name(),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "config": {
                    "runs_per_variation": self.config.runs_per_variation,
                    "step_limit": self.config.step_limit,
                    "cost_limit": self.config.cost_limit,
                    "workers": self.config.workers,
                },
            },
            "overall": {
                **compute_stats(results),
                "skipped": self._last_skipped,
            },
            "by_run": {
                str(run_id): compute_stats(group)
                for run_id, group in sorted(by_run.items())
            },
            "by_task": {
                task_name: compute_stats(group)
                for task_name, group in sorted(by_task.items())
            },
            "by_exit_status": dict(sorted(by_exit_status.items())),
            "model_usage": dict(model_usage),
        }

        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary_data, f, indent=2)

        return output_path


def regenerate_summary(
    results_dir: Path,
    split: str = "test_id",
    output_path: Path | None = None,
) -> Path:
    """Regenerate summary from results.json without needing a Collector.

    This is useful for fixing summaries that were generated incorrectly,
    or for generating summaries from manually collected results.

    Args:
        results_dir: Directory containing results.json (e.g., trajectories/test/baseline_xxx).
        split: Data split name for the output filename (default: test_id).
        output_path: Optional custom output path. If None, uses default.

    Returns:
        Path to the saved summary file.

    Example:
        >>> from miniagenticrouter.research.collection import regenerate_summary
        >>> regenerate_summary(Path("trajectories/test/baseline_deepseek_deepseek-v3.2"))
    """
    results_dir = Path(results_dir)
    results_json_path = results_dir / "results.json"

    if not results_json_path.exists():
        raise FileNotFoundError(f"results.json not found in {results_dir}")

    # Load results from JSON
    with open(results_json_path) as f:
        data = json.load(f)

    # Convert to CollectionResult objects
    results = []
    for task_id, info in data.items():
        # Parse run_id from task_id (e.g., "boil_var0_run1" -> 1)
        run_id = 0
        match = re.search(r"_run(\d+)$", task_id)
        if match:
            run_id = int(match.group(1))
            original_task_id = task_id[: match.start()]
        else:
            original_task_id = task_id

        results.append(
            CollectionResult(
                task_id=original_task_id,
                run_id=run_id,
                traj_path=results_dir / task_id / f"{task_id}.traj.json",
                final_score=info.get("score", 0.0),
                total_cost=info.get("cost", 0.0),
                n_steps=info.get("steps", 0),
                exit_status=info.get("exit_status", "Unknown"),
                model_usage=info.get("model_usage", {}),
                error=None if info.get("done") else "Not completed",
            )
        )

    if not results:
        raise ValueError(f"No results found in {results_json_path}")

    # Helper to compute group stats
    def compute_stats(group: list[CollectionResult]) -> dict:
        total = len(group)
        success = sum(1 for r in group if r.success)
        # avg_score and avg_steps: all tasks (LimitsExceeded tasks also have scores and steps)
        all_scores = [r.final_score for r in group]
        all_steps = [r.n_steps for r in group]
        # avg_steps_success: only successful tasks
        success_steps = [r.n_steps for r in group if r.success]
        return {
            "total": total,
            "success": success,
            "failed": total - success,
            "success_rate": round(success / total, 4) if total > 0 else 0.0,
            "total_cost": round(sum(r.total_cost for r in group), 6),
            "avg_score": round(sum(all_scores) / len(all_scores), 2) if all_scores else 0.0,
            "avg_steps": round(sum(all_steps) / len(all_steps), 1) if all_steps else 0.0,
            "avg_steps_success": round(sum(success_steps) / len(success_steps), 1) if success_steps else 0.0,
        }

    # Helper to extract task_name from task_id
    def extract_task_name(task_id: str) -> str:
        match = re.match(r"^(.+?)_var\d+", task_id)
        return match.group(1) if match else task_id

    # Group by run_id
    by_run: dict[int, list[CollectionResult]] = defaultdict(list)
    for r in results:
        by_run[r.run_id].append(r)

    # Group by task_name
    by_task: dict[str, list[CollectionResult]] = defaultdict(list)
    for r in results:
        task_name = extract_task_name(r.task_id)
        by_task[task_name].append(r)

    # Group by exit_status
    by_exit_status: dict[str, int] = defaultdict(int)
    for r in results:
        by_exit_status[r.exit_status] += 1

    # Model usage
    model_usage: dict[str, dict[str, Any]] = defaultdict(lambda: {"calls": 0, "cost": 0.0})
    for r in results:
        for model, count in r.model_usage.items():
            model_usage[model]["calls"] += count
        total_calls = sum(r.model_usage.values()) if r.model_usage else 1
        for model, count in r.model_usage.items():
            model_usage[model]["cost"] += r.total_cost * (count / total_calls)

    for model in model_usage:
        model_usage[model]["cost"] = round(model_usage[model]["cost"], 6)

    # Build summary dict
    summary_data = {
        "metadata": {
            "split": split,
            "mode": results_dir.name,  # Use directory name as mode
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "regenerated": True,
        },
        "overall": compute_stats(results),
        "by_run": {
            str(run_id): compute_stats(group)
            for run_id, group in sorted(by_run.items())
        },
        "by_task": {
            task_name: compute_stats(group)
            for task_name, group in sorted(by_task.items())
        },
        "by_exit_status": dict(sorted(by_exit_status.items())),
        "model_usage": dict(model_usage),
    }

    # Determine output path
    if output_path is None:
        output_path = results_dir / f"summary_{split}.json"

    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"Regenerated summary: {output_path}")
    print(f"  Total results: {len(results)}")
    print(f"  Runs: {sorted(by_run.keys())}")
    print(f"  Tasks: {len(by_task)}")

    return output_path
