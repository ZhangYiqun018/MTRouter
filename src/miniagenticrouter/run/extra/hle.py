#!/usr/bin/env python3

"""Run mini-SWE-agent on HLE (Humanity's Last Exam) tasks in batch mode."""

from __future__ import annotations

import concurrent.futures
import json
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import typer
import yaml
from rich.live import Live

from miniagenticrouter.agents.multitool import MultiToolAgent
from miniagenticrouter.config import builtin_config_dir, get_config_path
from miniagenticrouter.data.hle_tasks import (
    enumerate_hle_tasks,
    get_hle_stats,
    list_hle_subjects,
)
from miniagenticrouter.eval.hle_judge import LLMJudge
from miniagenticrouter.models import get_model
from miniagenticrouter.run.extra.utils.batch_progress import RunBatchProgressManager
from miniagenticrouter.run.utils.router import create_model_or_router
from miniagenticrouter.run.utils.save import save_traj
from miniagenticrouter.utils.log import add_file_handler, logger

_HELP_TEXT = """Run mini-SWE-agent on HLE (Humanity's Last Exam) benchmark tasks.

[not dim]
HLE is a multi-turn benchmark at the frontier of human knowledge,
with 2,500 questions across mathematics, humanities, and natural sciences.
[/not dim]
"""

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

_OUTPUT_FILE_LOCK = threading.Lock()


class ProgressTrackingMultiToolAgent(MultiToolAgent):
    """Simple wrapper around MultiToolAgent that provides progress updates."""

    def __init__(
        self,
        *args,
        progress_manager: RunBatchProgressManager,
        task_id: str = "",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.progress_manager: RunBatchProgressManager = progress_manager
        self.task_id = task_id

    def step(self) -> dict:
        """Override step to provide progress updates."""
        self.progress_manager.update_instance_status(
            self.task_id, f"Step {self.model.n_calls + 1:3d} (${self.model.cost:.2f})"
        )
        return super().step()


def update_results_file(output_path: Path, task_id: str, result_data: dict):
    """Update the output JSON file with results from a single task."""
    with _OUTPUT_FILE_LOCK:
        output_data = {"metadata": {}, "tasks": {}}
        if output_path.exists():
            output_data = json.loads(output_path.read_text())

        output_data["tasks"][task_id] = result_data

        # Update metadata
        tasks = output_data["tasks"]
        output_data["metadata"]["total"] = len(tasks)
        output_data["metadata"]["completed"] = sum(1 for t in tasks.values() if t.get("done"))
        output_data["metadata"]["timestamp"] = datetime.now(timezone.utc).isoformat()

        output_path.write_text(json.dumps(output_data, indent=2))


def remove_from_results_file(output_path: Path, task_id: str):
    """Remove a task from the results file."""
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        output_data = json.loads(output_path.read_text())
        if task_id in output_data.get("tasks", {}):
            del output_data["tasks"][task_id]
            output_path.write_text(json.dumps(output_data, indent=2))


def process_task(
    task_info: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
) -> None:
    """Process a single HLE task."""
    task_id = task_info["task_id"]
    question = task_info["question"]
    ground_truth = task_info["answer"]
    subject = task_info.get("subject", "")
    question_type = task_info.get("question_type", "short_answer")

    task_dir = output_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    # Avoid inconsistent state if something fails
    remove_from_results_file(output_dir / "results.json", task_id)
    (task_dir / f"{task_id}.traj.json").unlink(missing_ok=True)

    # Create model or router based on config
    router_type = config.get("_router_type", "none")
    model_names = config.get("_router_models")
    model = create_model_or_router(model_names, router_type, config.get("model", {}))

    progress_manager.on_instance_start(task_id)
    progress_manager.update_instance_status(task_id, "Initializing agent")

    agent = None
    extra_info = None
    exit_status, result = None, None
    start_time = datetime.now(timezone.utc)

    try:
        # Create agent
        agent_config = dict(config.get("agent", {}))
        agent_config.pop("agent_class", None)  # Remove class spec from kwargs

        agent = ProgressTrackingMultiToolAgent(
            model=model,
            env=None,  # HLE doesn't need an environment
            progress_manager=progress_manager,
            task_id=task_id,
            **agent_config,
        )

        # Add task data to template vars
        agent.extra_template_vars["question"] = question
        agent.extra_template_vars["task_id"] = task_id
        agent.extra_template_vars["subject"] = subject

        # Run the agent
        exit_status, result = agent.run(question)

    except Exception as e:
        logger.error(f"Error processing task {task_id}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}

    finally:
        end_time = datetime.now(timezone.utc)

        # Build benchmark_data for unified trajectory format
        benchmark_data = {
            "question": question,
            "ground_truth": ground_truth,
            "subject": subject,
            "question_type": question_type,
            "prediction": result if exit_status == "Submitted" else None,
        }

        extra_info = (extra_info or {}) | {
            "task_id": task_id,
            "benchmark": "hle",
            "benchmark_data": benchmark_data,
            "timestamps": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
            },
        }

        save_traj(
            agent,
            task_dir / f"{task_id}.traj.json",
            exit_status=exit_status,
            result=result,
            extra_info=extra_info,
            print_fct=logger.info,
        )

        result_data = {
            "task_id": task_id,
            "subject": subject,
            "question_type": question_type,
            "prediction": result if exit_status == "Submitted" else None,
            "ground_truth": ground_truth,
            "steps": model.n_calls,
            "cost": model.cost,
            "exit_status": exit_status,
            "done": exit_status == "Submitted",
            "correct": None,  # Will be set by Judge if enabled
        }

        # Run LLM Judge if enabled and task completed successfully
        judge_config = config.get("judge", {})
        if judge_config.get("enabled", False) and exit_status == "Submitted" and result:
            try:
                progress_manager.update_instance_status(task_id, "Running Judge")
                judge = LLMJudge(
                    model_name=judge_config.get("model_name"),
                    model_kwargs=judge_config.get("model_kwargs", {}),
                )
                judge_result = judge.judge(
                    question=question,
                    response=result,
                    correct_answer=ground_truth,
                    task_id=task_id,
                )
                result_data["correct"] = judge_result.correct
                result_data["judge_reasoning"] = judge_result.reasoning
                benchmark_data["judge_result"] = judge_result.to_dict()
            except Exception as e:
                logger.error(f"LLM Judge failed for task {task_id}: {e}")
                result_data["judge_error"] = str(e)

        update_results_file(output_dir / "results.json", task_id, result_data)
        progress_manager.on_instance_end(task_id, f"{exit_status}")


def update_predictions_file(output_path: Path, results_path: Path):
    """Generate predictions.json in official HLE format for potential submission."""
    if not results_path.exists():
        return

    results = json.loads(results_path.read_text())
    predictions = {}

    for task_id, task_result in results.get("tasks", {}).items():
        predictions[task_id] = {
            "model": results.get("metadata", {}).get("model", "unknown"),
            "response": task_result.get("prediction", ""),
        }

    output_path.write_text(json.dumps(predictions, indent=2))


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    subject_filter: str | None = typer.Option(None, "-s", "--subject", help="Subject filter (glob pattern)", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:100' for first 100 tasks)", rich_help_panel="Data selection"),
    ids_file: Path | None = typer.Option(None, "--ids", help="File containing task IDs (one per line)", rich_help_panel="Data selection"),
    output: str = typer.Option("./hle_results", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads", rich_help_panel="Basic"),
    model_name: str | None = typer.Option(None, "-m", "--model", help="Model(s) to use. For routers, use comma-separated names", rich_help_panel="Basic"),
    router: str = typer.Option("none", "-r", "--router", help="Router type: none, roulette, or interleaving", rich_help_panel="Basic"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing tasks", rich_help_panel="Data selection"),
    config_spec: Path = typer.Option(builtin_config_dir / "extra" / "hle.yaml", "-c", "--config", help="Path to config file", rich_help_panel="Basic"),
    list_subjects: bool = typer.Option(False, "--list-subjects", help="List available subjects and exit"),
    stats: bool = typer.Option(False, "--stats", help="Show dataset statistics and exit"),
) -> None:
    # fmt: on
    if stats:
        try:
            hle_stats = get_hle_stats()
            typer.echo("\nHLE Dataset Statistics\n")
            typer.echo(f"Total questions: {hle_stats['total']}")
            typer.echo(f"Text-only: {hle_stats['text_only']}")
            typer.echo(f"Multimodal: {hle_stats['multimodal']}")
            typer.echo(f"\nQuestion types:")
            for qtype, count in hle_stats["question_types"].items():
                typer.echo(f"  {qtype}: {count}")
        except Exception as e:
            typer.echo(f"Error loading dataset: {e}")
            typer.echo("Make sure you have accepted the dataset terms on Hugging Face")
        raise typer.Exit()

    if list_subjects:
        try:
            subjects = list_hle_subjects()
            typer.echo("\nAvailable HLE Subjects:\n")
            for subject in subjects:
                typer.echo(f"  - {subject}")
        except Exception as e:
            typer.echo(f"Error: {e}")
        raise typer.Exit()

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "miniagenticrouter.log")

    # Load task IDs from file if provided
    task_ids = None
    if ids_file is not None:
        task_ids = [line.strip() for line in ids_file.read_text().splitlines() if line.strip()]
        logger.info(f"Loaded {len(task_ids)} task IDs from {ids_file}")

    # Enumerate tasks
    try:
        tasks = enumerate_hle_tasks(
            subject_filter=subject_filter,
            slice_spec=slice_spec if slice_spec else None,
            ids=task_ids,
            skip_multimodal=True,  # Skip multimodal in first version
        )
    except Exception as e:
        logger.error(f"Error loading HLE dataset: {e}")
        typer.echo("Make sure you have accepted the dataset terms on Hugging Face")
        raise typer.Exit(1)

    if not tasks:
        logger.error("No tasks found matching the filter criteria")
        raise typer.Exit(1)

    # Skip existing
    if not redo_existing and (output_path / "results.json").exists():
        existing_data = json.loads((output_path / "results.json").read_text())
        existing_tasks = list(existing_data.get("tasks", {}).keys())
        logger.info(f"Skipping {len(existing_tasks)} existing tasks")
        tasks = [t for t in tasks if t["task_id"] not in existing_tasks]

    logger.info(f"Running on {len(tasks)} tasks...")

    # Load config
    config_path = get_config_path(config_spec)
    logger.info(f"Loading agent config from '{config_path}'")
    config = yaml.safe_load(config_path.read_text())

    # Store router configuration for workers
    config["_router_type"] = router
    config["_router_models"] = model_name

    if router != "none" and model_name:
        logger.info(f"Using {router} router with models: {model_name}")
    elif model_name is not None:
        config.setdefault("model", {})["model_name"] = model_name
        config["metadata"] = {"model": model_name, "benchmark": "hle"}

    progress_manager = RunBatchProgressManager(
        len(tasks), output_path / f"exit_statuses_{time.time()}.yaml"
    )

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                task_id = futures[future]
                logger.error(f"Error in future for task {task_id}: {e}", exc_info=True)
                progress_manager.on_uncaught_exception(task_id, e)

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_task, task_info, output_path, config, progress_manager
                ): task_info["task_id"]
                for task_info in tasks
            }
            try:
                process_futures(futures)
            except KeyboardInterrupt:
                logger.info("Cancelling all pending jobs. Press ^C again to exit immediately.")
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()
                process_futures(futures)

    # Generate predictions.json for potential submission
    update_predictions_file(output_path / "predictions.json", output_path / "results.json")

    # Print summary
    if (output_path / "results.json").exists():
        results = json.loads((output_path / "results.json").read_text())
        tasks_data = results.get("tasks", {})
        total_tasks = len(tasks_data)
        completed = sum(1 for r in tasks_data.values() if r.get("done"))
        total_cost = sum(r.get("cost", 0) for r in tasks_data.values())

        # Calculate accuracy from Judge results
        judged_tasks = [r for r in tasks_data.values() if r.get("correct") is not None]
        correct_count = sum(1 for r in judged_tasks if r.get("correct"))
        judged_count = len(judged_tasks)

        typer.echo(f"\n{'='*50}")
        typer.echo(f"Summary:")
        typer.echo(f"  Total tasks: {total_tasks}")
        typer.echo(f"  Completed: {completed} ({100*completed/total_tasks:.1f}%)" if total_tasks else "  Completed: 0")
        if judged_count > 0:
            typer.echo(f"  Accuracy: {correct_count}/{judged_count} ({100*correct_count/judged_count:.1f}%)")
        typer.echo(f"  Total cost: ${total_cost:.2f}")
        typer.echo(f"\nOutput files:")
        typer.echo(f"  Results: {output_path / 'results.json'}")
        typer.echo(f"  Predictions: {output_path / 'predictions.json'}")
        typer.echo(f"{'='*50}")


if __name__ == "__main__":
    app()
