"""HLE data collector for research experiments.

This module provides the HLECollector class for systematic trajectory collection
on the HLE (Humanity's Last Exam) benchmark.

This is parallel to collector.py (for ScienceWorld) but adapted for HLE:
- Uses HLEDataSplit instead of DataSplit
- No simplification_str parameter
- Different task_id format (no _varN suffix)
- Uses LLM Judge for scoring instead of environment scores
"""

from __future__ import annotations

import json
import re
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml

if TYPE_CHECKING:
    from miniagenticrouter.research.collection.modes import CollectionMode
    from miniagenticrouter.research.data.hle_split import HLEDataSplit


# Default template for final step warning
DEFAULT_FINAL_STEP_TEMPLATE = """
⚠️ [Step {current_step}/{step_limit}] This is your FINAL step.
You MUST call the answer tool NOW with your best answer based on all information gathered so far.
Synthesize your findings and provide your final answer immediately.
"""


@dataclass
class HLECollectionConfig:
    """Configuration for HLE data collection.

    Attributes:
        output_dir: Base directory for saving trajectories.
        runs_per_task: Number of runs per task.
        step_limit: Maximum steps per episode.
        cost_limit: Maximum cost (USD) per episode.
        workers: Number of parallel workers.
        record_propensity: Whether to record selection probabilities.
        agent_config: Optional dict of agent configuration.
        skip_existing: Whether to skip already collected trajectories.
        enable_judge: Whether to run LLM Judge for evaluation.
        judge_model: Model name for the Judge.
        final_step_warning: Whether to warn agent on final step.
        final_step_template: Template for final step warning message.
        force_answer_on_limit: Whether to force extract answer when step limit is reached.
    """

    output_dir: Path
    runs_per_task: int = 3
    step_limit: int | None = None  # None = use hle.yaml default
    cost_limit: float | None = None  # None = use hle.yaml default
    workers: int = 4
    record_propensity: bool = False
    agent_config: dict | None = None
    skip_existing: bool = True
    enable_judge: bool = True
    judge_model: str | None = None
    # Final step handling
    final_step_warning: bool = True
    final_step_template: str = DEFAULT_FINAL_STEP_TEMPLATE
    force_answer_on_limit: bool = True

    def __post_init__(self) -> None:
        """Ensure output_dir is a Path."""
        self.output_dir = Path(self.output_dir)


@dataclass
class HLECollectionResult:
    """Result of a single HLE trajectory collection.

    Attributes:
        task_id: Task identifier.
        run_id: Run number (0-indexed).
        traj_path: Path to the saved trajectory file.
        judge_score: LLM Judge score (0 or 1).
        total_cost: Total cost (USD) for the episode.
        n_steps: Number of steps taken.
        exit_status: Exit status string.
        model_usage: Dict mapping model names to call counts.
        error: Error message if collection failed.
    """

    task_id: str
    run_id: int
    traj_path: Path | None
    judge_score: float  # 0 or 1 from Judge
    total_cost: float
    n_steps: int
    exit_status: str
    model_usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    @property
    def success(self) -> bool:
        """Whether collection succeeded."""
        return self.error is None


class HLECollector:
    """Systematic data collector for HLE research experiments.

    This class coordinates trajectory collection for HLE benchmark:
    - Different data splits (train/val/test_id/ood_test)
    - Different collection modes (baseline/roulette)
    - Multiple runs per task

    Example:
        >>> from miniagenticrouter.research.data import HLEDataSplit
        >>> from miniagenticrouter.research.collection import BaselineMode
        >>> split = HLEDataSplit.from_yaml()
        >>> mode = BaselineMode(model_name="openai/gpt-5")
        >>> config = HLECollectionConfig(output_dir=Path("./hle_trajectories"))
        >>> collector = HLECollector(data_split=split, mode=mode, config=config)
        >>> results = collector.collect(split="train")
    """

    def __init__(
        self,
        data_split: HLEDataSplit,
        mode: CollectionMode,
        config: HLECollectionConfig,
    ):
        """Initialize HLECollector.

        Args:
            data_split: HLE data split configuration.
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

        Returns config from HLECollectionConfig.agent_config if provided,
        otherwise loads from default hle.yaml.
        """
        if self.config.agent_config is not None:
            agent_config = dict(self.config.agent_config)
        else:
            # Load default HLE config
            from miniagenticrouter.config import builtin_config_dir

            config_path = builtin_config_dir / "extra" / "hle.yaml"
            with open(config_path) as f:
                full_config = yaml.safe_load(f)
            agent_config = dict(full_config.get("agent", {}))

        # Remove class spec from kwargs (not a FlexibleAgent parameter)
        agent_config.pop("agent_class", None)

        # Override step/cost limits only if explicitly set in config
        # (None means use yaml default)
        if self.config.step_limit is not None:
            agent_config["step_limit"] = self.config.step_limit
        if self.config.cost_limit is not None:
            agent_config["cost_limit"] = self.config.cost_limit

        return agent_config

    def _get_judge_model(self) -> str | None:
        """Get judge model name.

        Returns config value if set, otherwise reads from hle.yaml.
        """
        if self.config.judge_model is not None:
            return self.config.judge_model

        # Load from hle.yaml
        from miniagenticrouter.config import builtin_config_dir

        config_path = builtin_config_dir / "extra" / "hle.yaml"
        with open(config_path) as f:
            full_config = yaml.safe_load(f)
        judge_config = full_config.get("judge", {})
        return judge_config.get("model_name")

    def _get_config_signature(self) -> str:
        """Generate config signature for skip-existing validation.

        Uses effective (resolved) limits to ensure trajectories collected under
        different YAML defaults are not incorrectly reused.
        """
        import hashlib

        # Get effective limits (resolved from yaml if not explicitly set)
        agent_config = self._get_agent_config()
        effective_step_limit = agent_config.get("step_limit")
        effective_cost_limit = agent_config.get("cost_limit")

        sig_data = {
            "mode": self.mode.get_mode_name(),
            "models": self.mode.get_model_names(self._model_configs),
            "step_limit": effective_step_limit,
            "cost_limit": effective_cost_limit,
            "benchmark": "hle",
        }
        sig_json = json.dumps(sig_data, sort_keys=True)
        return hashlib.md5(sig_json.encode()).hexdigest()[:8]

    def collect(
        self,
        split: Literal["train", "val", "test_id", "ood_test"] = "train",
        category_filter: str | None = None,
        task_id: str | None = None,
        limit: int | None = None,
        dry_run: bool = False,
    ) -> list[HLECollectionResult]:
        """Collect trajectories for a data split.

        Args:
            split: Which split to collect for.
            category_filter: Optional category to filter tasks.
            task_id: Optional specific task ID to collect.
            limit: Optional limit on number of tasks to collect.
            dry_run: If True, only print what would be done.

        Returns:
            List of collection results.
        """
        import fnmatch

        from rich.live import Live

        from miniagenticrouter.run.extra.utils.batch_progress import (
            RunBatchProgressManager,
        )

        # Get tasks for this split
        tasks = self.data_split.enumerate_tasks(split=split)

        # Apply task_id filter (single task)
        if task_id:
            tasks = [t for t in tasks if t["task_id"] == task_id]
            if not tasks:
                print(f"Warning: Task ID '{task_id}' not found in split '{split}'")
                return []

        # Apply category filter
        if category_filter:
            tasks = [
                t for t in tasks if fnmatch.fnmatch(t.get("category", ""), category_filter)
            ]

        # Apply limit
        if limit is not None and limit > 0:
            tasks = tasks[:limit]

        if dry_run:
            self._print_collection_plan(tasks)
            return []

        # Generate all (task, run_id) pairs
        work_items = []
        skipped = 0
        for task in tasks:
            for run_id in range(self.config.runs_per_task):
                # Skip if already exists and is valid
                if self.config.skip_existing:
                    output_path = self.get_output_path(task["task_id"], run_id)
                    if self._is_valid_trajectory(output_path):
                        skipped += 1
                        continue

                # Create task_info with run_id
                task_with_run = dict(task)
                task_with_run["original_task_id"] = task["task_id"]
                task_with_run["run_id"] = run_id
                task_with_run["task_id"] = f"{task['task_id']}_run{run_id}"
                # Add compatibility fields for TrajectoryParser
                task_with_run["task_name"] = task.get("category", task["task_id"])
                task_with_run["variation_idx"] = 0
                work_items.append(task_with_run)

        if skipped > 0:
            print(f"Skipping {skipped} already collected trajectories")

        self._last_skipped = skipped

        if not work_items:
            print("No tasks to collect (all already done)")
            return []

        # Compute output directory with mode subdirectory
        mode_name = self.mode.get_output_subdir()
        output_dir = self.config.output_dir / mode_name

        # Create progress manager
        progress_manager = RunBatchProgressManager(
            num_instances=len(work_items),
            yaml_report_path=output_dir / f"collection_report_{split}.yaml",
        )

        results = []
        config_signature = self._get_config_signature()
        agent_config = self._get_agent_config()

        def process_one(task_info: dict) -> HLECollectionResult:
            """Process a single HLE task and return result."""
            return self._process_task(
                task_info=task_info,
                output_dir=output_dir,
                agent_config=agent_config,
                progress_manager=progress_manager,
                config_signature=config_signature,
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
                            result = HLECollectionResult(
                                task_id=task_info["original_task_id"],
                                run_id=task_info["run_id"],
                                traj_path=None,
                                judge_score=0.0,
                                total_cost=0.0,
                                n_steps=0,
                                exit_status="Error",
                                error=str(e),
                            )

                        # Update model distribution display
                        if result.model_usage:
                            progress_manager.update_model_usage(result.model_usage)

                        results.append(result)
        finally:
            if hasattr(self.mode, "shutdown"):
                self.mode.shutdown()

        return results

    def _process_task(
        self,
        task_info: dict,
        output_dir: Path,
        agent_config: dict,
        progress_manager: Any,
        config_signature: str,
    ) -> HLECollectionResult:
        """Process a single HLE task.

        This is adapted from run/extra/hle.py process_task but accepts
        an external model for research collection.
        """
        from miniagenticrouter.agents.multitool import MultiToolAgent
        from miniagenticrouter.eval.hle_judge import LLMJudge
        from miniagenticrouter.run.utils.save import save_traj
        from miniagenticrouter.utils.log import logger

        task_id = task_info["task_id"]
        original_task_id = task_info["original_task_id"]
        run_id = task_info["run_id"]
        question = task_info["question"]
        ground_truth = task_info["answer"]
        subject = task_info.get("subject", "")
        category = task_info.get("category", "")
        question_type = task_info.get("question_type", "short_answer")

        task_dir = output_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        # Create model for this task
        model = self.mode.create_model_or_router(self._model_configs)

        progress_manager.on_instance_start(task_id)
        progress_manager.update_instance_status(task_id, "Initializing agent")

        agent = None
        exit_status, result = None, None
        start_time = datetime.now(timezone.utc)
        extra_info: dict[str, Any] = {}

        try:
            # Create agent with progress tracking and final step warning
            agent = _ProgressTrackingMultiToolAgent(
                model=model,
                env=None,
                progress_manager=progress_manager,
                task_id=task_id,
                final_step_warning=self.config.final_step_warning,
                final_step_template=self.config.final_step_template,
                **agent_config,
            )

            # Add task data to template vars
            agent.extra_template_vars["question"] = question
            agent.extra_template_vars["task_id"] = task_id
            agent.extra_template_vars["subject"] = subject

            # Run the agent
            exit_status, result = agent.run(question)

            # Handle forced answer on LimitsExceeded
            if (exit_status == "LimitsExceeded"
                and self.config.force_answer_on_limit
                and agent.messages):
                # Extract last assistant response as forced answer
                for msg in reversed(agent.messages):
                    if msg["role"] == "assistant" and msg.get("content"):
                        result = msg["content"]
                        exit_status = "ForcedAnswer"
                        logger.info(f"Task {task_id}: Forced answer extracted from last response")
                        break

        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}", exc_info=True)
            exit_status, result = type(e).__name__, str(e)
            extra_info["traceback"] = traceback.format_exc()

        finally:
            end_time = datetime.now(timezone.utc)

            # Get model usage first
            model_usage = {}
            if hasattr(model, "models"):
                for m in model.models:
                    if m.n_calls > 0:
                        model_usage[m.config.model_name] = m.n_calls
            else:
                model_usage[model.config.model_name] = model.n_calls

            # Build benchmark_data
            # Include prediction for both Submitted and ForcedAnswer
            has_answer = exit_status in ("Submitted", "ForcedAnswer")
            benchmark_data = {
                "question": question,
                "ground_truth": ground_truth,
                "subject": subject,
                "category": category,
                "question_type": question_type,
                "prediction": result if has_answer else None,
            }

            # Run LLM Judge BEFORE save_traj (so judge_result is included in trajectory)
            judge_score = 0.0
            if self.config.enable_judge and has_answer and result:
                try:
                    progress_manager.update_instance_status(task_id, "Running Judge")
                    judge_model = self._get_judge_model()
                    judge = LLMJudge(model_name=judge_model)
                    judge_result = judge.judge(
                        question=question,
                        response=result,
                        correct_answer=ground_truth,
                        task_id=task_id,
                    )
                    judge_score = 1.0 if judge_result.correct else 0.0
                    benchmark_data["judge_result"] = judge_result.to_dict()
                    # Update progress display with judge result
                    progress_manager.update_judge_result(judge_result.correct)
                except Exception as e:
                    logger.error(f"LLM Judge failed for task {task_id}: {e}")

            extra_info.update({
                "task_id": original_task_id,
                "run_id": run_id,
                "benchmark": "hle",
                "benchmark_data": benchmark_data,
                "config_signature": config_signature,
                "timestamps": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                },
            })

            # Save trajectory (now includes judge_result if available)
            save_traj(
                agent,
                task_dir / f"{task_id}.traj.json",
                exit_status=exit_status,
                result=result,
                extra_info=extra_info,
                print_fct=logger.info,
            )

            # Update results file
            result_data = {
                "task_id": original_task_id,
                "run_id": run_id,
                "category": category,
                "subject": subject,
                "question_type": question_type,
                "prediction": result if has_answer else None,
                "ground_truth": ground_truth,
                "steps": model.n_calls,
                "cost": model.cost,
                "exit_status": exit_status,
                "done": has_answer,  # Both Submitted and ForcedAnswer count as done
                "correct": judge_score == 1.0 if self.config.enable_judge else None,
                "model_usage": model_usage,
            }
            self._update_results_file(output_dir / "results.json", task_id, result_data)

            # Update progress display with turns count
            progress_manager.update_turns(model.n_calls)

            progress_manager.on_instance_end(task_id, f"{exit_status}")

            return HLECollectionResult(
                task_id=original_task_id,
                run_id=run_id,
                traj_path=task_dir / f"{task_id}.traj.json",
                judge_score=judge_score,
                total_cost=model.cost,
                n_steps=model.n_calls,
                exit_status=exit_status or "Unknown",
                model_usage=model_usage,
            )

    def _update_results_file(self, output_path: Path, task_id: str, result_data: dict):
        """Update the output JSON file with results from a single task."""
        import threading

        # Use a class-level lock for thread safety
        if not hasattr(self, "_results_lock"):
            self._results_lock = threading.Lock()

        with self._results_lock:
            output_data = {"metadata": {}, "tasks": {}}
            if output_path.exists():
                output_data = json.loads(output_path.read_text())

            output_data["tasks"][task_id] = result_data

            # Update metadata
            tasks = output_data["tasks"]
            output_data["metadata"]["total"] = len(tasks)
            output_data["metadata"]["completed"] = sum(1 for t in tasks.values() if t.get("done"))
            output_data["metadata"]["timestamp"] = datetime.now(timezone.utc).isoformat()
            output_data["metadata"]["benchmark"] = "hle"

            output_path.write_text(json.dumps(output_data, indent=2))

    def get_output_path(self, task_id: str, run_id: int) -> Path:
        """Get the output path for a trajectory."""
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

        A valid trajectory must:
        - Exist and be parseable as JSON
        - Have a completed exit_status (Submitted, ForcedAnswer, LimitsExceeded)
        - Have matching config_signature

        Trajectories with error exit_status (APIError, InternalServerError, etc.)
        are considered invalid and will be retried.
        """
        if not path.exists():
            return False

        try:
            with open(path) as f:
                data = json.load(f)
            info = data.get("info", {})
            exit_status = info.get("exit_status")
            if exit_status not in self.COMPLETED_EXIT_STATUSES:
                return False
            if "config_signature" not in info:
                return False
            return info["config_signature"] == self._get_config_signature()
        except (json.JSONDecodeError, OSError):
            return False

    def _print_collection_plan(self, tasks: list[dict]) -> None:
        """Print what would be collected (dry run)."""
        n_tasks = len(tasks)
        n_runs = self.config.runs_per_task
        n_total = n_tasks * n_runs

        print(f"HLE Collection Plan (dry run):")
        print(f"  Mode: {self.mode.get_mode_name()}")
        print(f"  Output dir: {self.config.output_dir}")
        print(f"  Tasks: {n_tasks}")
        print(f"  Runs per task: {n_runs}")
        print(f"  Total trajectories: {n_total}")
        print(f"  Workers: {self.config.workers}")
        print()
        print(f"First 10 tasks:")
        for task in tasks[:10]:
            print(f"  - {task['task_id']} ({task.get('category', 'unknown')})")
        if n_tasks > 10:
            print(f"  ... and {n_tasks - 10} more")

    def summary(self, results: list[HLECollectionResult]) -> str:
        """Generate a summary of collection results."""
        n_total = len(results)
        n_success = sum(1 for r in results if r.success)
        n_failed = n_total - n_success

        total_cost = sum(r.total_cost for r in results)
        avg_score = (
            sum(r.judge_score for r in results) / n_total
            if n_total > 0
            else 0.0
        )
        avg_steps = (
            sum(r.n_steps for r in results) / n_total
            if n_total > 0
            else 0.0
        )

        lines = [
            f"HLE Collection Summary:",
            f"  Skipped (existing): {self._last_skipped}",
            f"  Collected: {n_total}",
            f"  Success: {n_success}",
            f"  Failed: {n_failed}",
            f"  Total cost: ${total_cost:.2f}",
            f"  Avg judge score: {avg_score:.2f}",
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

    def _load_results_from_json(self, results_path: Path) -> list[HLECollectionResult]:
        """Load all results from results.json file.

        Converts the HLE JSON format to HLECollectionResult objects.

        Args:
            results_path: Path to results.json file.

        Returns:
            List of HLECollectionResult objects.
        """
        if not results_path.exists():
            return []

        try:
            with open(results_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

        results = []
        tasks = data.get("tasks", {})
        for instance_id, info in tasks.items():
            # Parse run_id from instance_id (e.g., "abc123_run1" -> 1)
            run_id = info.get("run_id", 0)
            original_task_id = info.get("task_id", instance_id)

            results.append(
                HLECollectionResult(
                    task_id=original_task_id,
                    run_id=run_id,
                    traj_path=results_path.parent / instance_id / f"{instance_id}.traj.json",
                    judge_score=1.0 if info.get("correct") else 0.0,
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
        results: list[HLECollectionResult],
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

        # Load ALL results from results.json
        results_json_path = output_path.parent / "results.json"
        all_results = self._load_results_from_json(results_json_path)

        # If we have results from JSON, use those; otherwise fall back to passed results
        if all_results:
            results = all_results

        # Helper to compute group stats
        def compute_stats(group: list[HLECollectionResult]) -> dict:
            total = len(group)
            success = sum(1 for r in group if r.success)
            correct = sum(1 for r in group if r.judge_score == 1.0)
            all_steps = [r.n_steps for r in group]
            success_steps = [r.n_steps for r in group if r.success]
            return {
                "total": total,
                "success": success,
                "failed": total - success,
                "correct": correct,
                "accuracy": round(correct / total, 4) if total > 0 else 0.0,
                "success_rate": round(success / total, 4) if total > 0 else 0.0,
                "total_cost": round(sum(r.total_cost for r in group), 6),
                "avg_steps": round(sum(all_steps) / len(all_steps), 1) if all_steps else 0.0,
                "avg_steps_success": round(sum(success_steps) / len(success_steps), 1) if success_steps else 0.0,
            }

        # Group by run_id
        by_run: dict[int, list[HLECollectionResult]] = defaultdict(list)
        for r in results:
            by_run[r.run_id].append(r)

        # Group by category (extracted from results.json via task info)
        # We need to load category info from results.json
        by_category: dict[str, list[HLECollectionResult]] = defaultdict(list)
        if results_json_path.exists():
            try:
                with open(results_json_path) as f:
                    raw_data = json.load(f)
                tasks_data = raw_data.get("tasks", {})
                for r in results:
                    instance_id = f"{r.task_id}_run{r.run_id}"
                    task_info = tasks_data.get(instance_id, {})
                    category = task_info.get("category", "Unknown")
                    by_category[category].append(r)
            except (json.JSONDecodeError, OSError):
                pass

        # Group by exit_status
        by_exit_status: dict[str, int] = defaultdict(int)
        for r in results:
            by_exit_status[r.exit_status] += 1

        # Model usage
        model_usage: dict[str, dict[str, Any]] = defaultdict(lambda: {"calls": 0, "cost": 0.0})
        for r in results:
            for model, count in r.model_usage.items():
                model_usage[model]["calls"] += count
            total_calls = sum(r.model_usage.values()) or 1  # Avoid division by zero
            for model, count in r.model_usage.items():
                model_usage[model]["cost"] += r.total_cost * (count / total_calls)

        for model in model_usage:
            model_usage[model]["cost"] = round(model_usage[model]["cost"], 6)

        # Build summary dict
        summary_data = {
            "metadata": {
                "split": split,
                "mode": self.mode.get_mode_name(),
                "benchmark": "hle",
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "config": {
                    "runs_per_task": self.config.runs_per_task,
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
            "by_category": {
                category: compute_stats(group)
                for category, group in sorted(by_category.items())
            },
            "by_exit_status": dict(sorted(by_exit_status.items())),
            "model_usage": dict(model_usage),
        }

        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary_data, f, indent=2)

        return output_path


def regenerate_hle_summary(
    results_dir: Path,
    split: str = "train",
    output_path: Path | None = None,
) -> Path:
    """Regenerate HLE summary from results.json without needing a Collector.

    This is useful for fixing summaries that were generated incorrectly,
    or for generating summaries from manually collected results.

    Args:
        results_dir: Directory containing results.json (e.g., hle_trajectories/roulette_propensity).
        split: Data split name for the output filename (default: train).
        output_path: Optional custom output path. If None, uses default.

    Returns:
        Path to the saved summary file.

    Example:
        >>> from miniagenticrouter.research.collection import regenerate_hle_summary
        >>> regenerate_hle_summary(Path("hle_trajectories/roulette_propensity"))
    """
    results_dir = Path(results_dir)
    results_json_path = results_dir / "results.json"

    if not results_json_path.exists():
        raise FileNotFoundError(f"results.json not found in {results_dir}")

    # Load results from JSON
    with open(results_json_path) as f:
        data = json.load(f)

    tasks_data = data.get("tasks", {})
    if not tasks_data:
        raise ValueError(f"No tasks found in {results_json_path}")

    # Convert to HLECollectionResult objects
    results = []
    for instance_id, info in tasks_data.items():
        run_id = info.get("run_id", 0)
        original_task_id = info.get("task_id", instance_id)

        results.append(
            HLECollectionResult(
                task_id=original_task_id,
                run_id=run_id,
                traj_path=results_dir / instance_id / f"{instance_id}.traj.json",
                judge_score=1.0 if info.get("correct") else 0.0,
                total_cost=info.get("cost", 0.0),
                n_steps=info.get("steps", 0),
                exit_status=info.get("exit_status", "Unknown"),
                model_usage=info.get("model_usage", {}),
                error=None if info.get("done") else "Not completed",
            )
        )

    # Helper to compute group stats
    def compute_stats(group: list[HLECollectionResult]) -> dict:
        total = len(group)
        success = sum(1 for r in group if r.success)
        correct = sum(1 for r in group if r.judge_score == 1.0)
        all_steps = [r.n_steps for r in group]
        success_steps = [r.n_steps for r in group if r.success]
        return {
            "total": total,
            "success": success,
            "failed": total - success,
            "correct": correct,
            "accuracy": round(correct / total, 4) if total > 0 else 0.0,
            "success_rate": round(success / total, 4) if total > 0 else 0.0,
            "total_cost": round(sum(r.total_cost for r in group), 6),
            "avg_steps": round(sum(all_steps) / len(all_steps), 1) if all_steps else 0.0,
            "avg_steps_success": round(sum(success_steps) / len(success_steps), 1) if success_steps else 0.0,
        }

    # Group by run_id
    by_run: dict[int, list[HLECollectionResult]] = defaultdict(list)
    for r in results:
        by_run[r.run_id].append(r)

    # Group by category
    by_category: dict[str, list[HLECollectionResult]] = defaultdict(list)
    for r in results:
        instance_id = f"{r.task_id}_run{r.run_id}"
        task_info = tasks_data.get(instance_id, {})
        category = task_info.get("category", "Unknown")
        by_category[category].append(r)

    # Group by exit_status
    by_exit_status: dict[str, int] = defaultdict(int)
    for r in results:
        by_exit_status[r.exit_status] += 1

    # Model usage
    model_usage: dict[str, dict[str, Any]] = defaultdict(lambda: {"calls": 0, "cost": 0.0})
    for r in results:
        for model, count in r.model_usage.items():
            model_usage[model]["calls"] += count
        total_calls = sum(r.model_usage.values()) or 1  # Avoid division by zero
        for model, count in r.model_usage.items():
            model_usage[model]["cost"] += r.total_cost * (count / total_calls)

    for model in model_usage:
        model_usage[model]["cost"] = round(model_usage[model]["cost"], 6)

    # Build summary dict
    summary_data = {
        "metadata": {
            "split": split,
            "mode": results_dir.name,
            "benchmark": "hle",
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "regenerated": True,
        },
        "overall": compute_stats(results),
        "by_run": {
            str(run_id): compute_stats(group)
            for run_id, group in sorted(by_run.items())
        },
        "by_category": {
            category: compute_stats(group)
            for category, group in sorted(by_category.items())
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

    print(f"Regenerated HLE summary: {output_path}")
    print(f"  Total results: {len(results)}")
    print(f"  Runs: {sorted(by_run.keys())}")
    print(f"  Categories: {len(by_category)}")
    print(f"  Accuracy: {summary_data['overall']['accuracy']:.2%}")

    return output_path


class _ProgressTrackingMultiToolAgent:
    """Wrapper around MultiToolAgent that provides progress updates and final step warning."""

    def __init__(
        self,
        *args,
        progress_manager: Any,
        task_id: str = "",
        final_step_warning: bool = True,
        final_step_template: str = DEFAULT_FINAL_STEP_TEMPLATE,
        **kwargs,
    ):
        from miniagenticrouter.agents.multitool import MultiToolAgent

        self._agent = MultiToolAgent(*args, **kwargs)
        self.progress_manager = progress_manager
        self.task_id = task_id
        self.final_step_warning = final_step_warning
        self.final_step_template = final_step_template
        # Forward attributes
        self.model = self._agent.model
        self.extra_template_vars = self._agent.extra_template_vars

    def run(self, *args, **kwargs):
        """Run with progress tracking and final step warning."""
        # Override query method to inject final step warning
        original_query = self._agent.query
        step_limit = self._agent.config.step_limit

        def query_with_warning():
            # Check if this is the final step (step_limit - 1 because n_calls is 0-indexed)
            # Guard against step_limit being None (when using yaml defaults or custom configs)
            if (self.final_step_warning
                and step_limit is not None
                and step_limit > 0
                and self.model.n_calls == step_limit - 1):
                # Inject warning into last user message
                warning = self.final_step_template.format(
                    current_step=self.model.n_calls + 1,
                    step_limit=step_limit,
                )
                if self._agent.messages and self._agent.messages[-1]["role"] == "user":
                    self._agent.messages[-1]["content"] += f"\n\n{warning}"

            return original_query()

        self._agent.query = query_with_warning

        # Override step method for progress tracking
        original_step = self._agent.step

        def tracked_step():
            self.progress_manager.update_instance_status(
                self.task_id, f"Step {self.model.n_calls + 1:3d} (${self.model.cost:.2f})"
            )
            return original_step()

        self._agent.step = tracked_step
        return self._agent.run(*args, **kwargs)

    def __getattr__(self, name):
        """Forward attribute access to wrapped agent."""
        return getattr(self._agent, name)
