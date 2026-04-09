#!/usr/bin/env python3

"""Run mini-SWE-agent on ScienceWorld tasks in batch mode."""

from __future__ import annotations

import concurrent.futures
import json
import threading
import time
import traceback
from pathlib import Path

import typer
import yaml
from rich.live import Live

from miniagenticrouter.agents.flexible import FlexibleAgent
from miniagenticrouter.config import builtin_config_dir, get_config_path
from miniagenticrouter.data.scienceworld_tasks import (
    enumerate_scienceworld_tasks,
    get_scienceworld_task_names,
)
from miniagenticrouter.environments.extra.scienceworld import ScienceWorldEnvironment
from miniagenticrouter.models import get_model
from miniagenticrouter.run.extra.utils.batch_progress import RunBatchProgressManager
from miniagenticrouter.run.utils.router import create_model_or_router
from miniagenticrouter.run.utils.save import save_traj
from miniagenticrouter.utils.log import add_file_handler, logger

_HELP_TEXT = """Run mini-agentic-router on ScienceWorld benchmark tasks.

[not dim]
ScienceWorld is a text-based science simulation environment for evaluating
scientific reasoning capabilities of language models.
[/not dim]
"""

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

_OUTPUT_FILE_LOCK = threading.Lock()


class ProgressTrackingAgent(FlexibleAgent):
    """Simple wrapper around FlexibleAgent that provides progress updates."""

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
        output_data = {}
        if output_path.exists():
            output_data = json.loads(output_path.read_text())
        output_data[task_id] = result_data
        output_path.write_text(json.dumps(output_data, indent=2))


def remove_from_results_file(output_path: Path, task_id: str):
    """Remove a task from the results file."""
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        output_data = json.loads(output_path.read_text())
        if task_id in output_data:
            del output_data[task_id]
            output_path.write_text(json.dumps(output_data, indent=2))


def process_task(
    task_info: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
    model: "Model | None" = None,
    extra_info_override: dict | None = None,
) -> dict:
    """Process a single ScienceWorld task.

    Args:
        task_info: Task information dict with task_id, task_name, variation_idx.
        output_dir: Directory to save results.
        config: Configuration dict with agent settings.
        progress_manager: Progress manager for status updates.
        model: Optional pre-created model/router. If None, creates from config.
        extra_info_override: Optional extra info to merge into trajectory.

    Returns:
        Result dict with task_id, score, steps, cost, exit_status.
    """
    task_id = task_info["task_id"]
    task_name = task_info["task_name"]
    variation_idx = task_info["variation_idx"]
    simplification_str = task_info.get("simplification_str", "")

    task_dir = output_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    # Avoid inconsistent state if something fails
    remove_from_results_file(output_dir / "results.json", task_id)
    (task_dir / f"{task_id}.traj.json").unlink(missing_ok=True)

    # Create model or router if not provided
    if model is None:
        router_type = config.get("_router_type", "none")
        model_names = config.get("_router_models")
        model = create_model_or_router(model_names, router_type, config.get("model", {}))

    progress_manager.on_instance_start(task_id)
    progress_manager.update_instance_status(task_id, "Initializing environment")

    agent = None
    env = None
    extra_info = None
    exit_status, result = None, None
    final_score = 0.0

    try:
        # Create environment
        env = ScienceWorldEnvironment(
            task_name=task_name,
            variation_idx=variation_idx,
            simplification_str=simplification_str,
        )

        # Load the task
        task_data = env.load_task(task_name, variation_idx)
        initial_observation = task_data["observation"]
        task_description = task_data["task_description"]

        # Create agent
        agent_config = dict(config.get("agent", {}))
        agent_config.pop("agent_class", None)  # Remove class spec from kwargs

        agent = ProgressTrackingAgent(
            model=model,
            env=env,
            progress_manager=progress_manager,
            task_id=task_id,
            **agent_config,
        )

        # Add initial observation to template vars
        # Note: task_description is already provided by env.get_template_vars()
        agent.extra_template_vars["initial_observation"] = initial_observation

        # Run the agent
        exit_status, result = agent.run(task_description)
        final_score = env.get_score()

    except Exception as e:
        logger.error(f"Error processing task {task_id}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}

    finally:
        extra_info = (extra_info or {}) | {
            "task_id": task_id,
            "task_name": task_name,
            "variation_idx": variation_idx,
            "final_score": final_score,
        }
        # Merge caller-provided extra info (e.g., config_signature)
        if extra_info_override:
            extra_info = extra_info | extra_info_override

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
            "task_name": task_name,
            "variation_idx": variation_idx,
            "score": final_score,
            "steps": model.n_calls,
            "cost": model.cost,
            "exit_status": exit_status,
            "done": exit_status == "Submitted",
        }
        update_results_file(output_dir / "results.json", task_id, result_data)
        progress_manager.on_instance_end(task_id, f"{exit_status} (score={final_score:.2f})")

        if env is not None:
            env.close()

        return result_data


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    task_filter: str | None = typer.Option(None, "-t", "--task", help="Task name filter (glob pattern, e.g., 'boil*')", rich_help_panel="Data selection"),
    variation: int = typer.Option(-1, "-v", "--variation", help="Variation index (-1 for all)", rich_help_panel="Data selection"),
    simplification: str = typer.Option("", "--simplification", help="Simplification string", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:10' for first 10 tasks)", rich_help_panel="Data selection"),
    output: str = typer.Option("./scienceworld_results", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads", rich_help_panel="Basic"),
    model_name: str | None = typer.Option(None, "-m", "--model", help="Model(s) to use. For routers, use comma-separated names", rich_help_panel="Basic"),
    router: str = typer.Option("none", "-r", "--router", help="Router type: none, roulette, or interleaving", rich_help_panel="Basic"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing tasks", rich_help_panel="Data selection"),
    config_spec: Path = typer.Option(builtin_config_dir / "extra" / "scienceworld.yaml", "-c", "--config", help="Path to config file", rich_help_panel="Basic"),
    list_tasks: bool = typer.Option(False, "--list-tasks", help="List available task names and exit"),
) -> None:
    # fmt: on
    if list_tasks:
        task_names = get_scienceworld_task_names()
        typer.echo("Available ScienceWorld tasks:")
        for name in sorted(task_names):
            typer.echo(f"  - {name}")
        raise typer.Exit()

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "miniagenticrouter.log")

    # Enumerate tasks
    variation_filter = None if variation < 0 else variation
    tasks = enumerate_scienceworld_tasks(
        task_filter=task_filter,
        variation_filter=variation_filter,
        simplification_str=simplification,
    )

    if not tasks:
        logger.error("No tasks found matching the filter criteria")
        raise typer.Exit(1)

    # Apply slice
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        tasks = tasks[slice(*values)]

    # Skip existing
    if not redo_existing and (output_path / "results.json").exists():
        existing_tasks = list(json.loads((output_path / "results.json").read_text()).keys())
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

    # Print summary
    if (output_path / "results.json").exists():
        results = json.loads((output_path / "results.json").read_text())
        total_tasks = len(results)
        completed = sum(1 for r in results.values() if r.get("done"))
        total_score = sum(r.get("score", 0) for r in results.values())
        avg_score = total_score / total_tasks if total_tasks else 0
        total_cost = sum(r.get("cost", 0) for r in results.values())

        typer.echo(f"\n{'='*50}")
        typer.echo(f"Summary:")
        typer.echo(f"  Total tasks: {total_tasks}")
        typer.echo(f"  Completed: {completed} ({100*completed/total_tasks:.1f}%)")
        typer.echo(f"  Average score: {avg_score:.3f}")
        typer.echo(f"  Total cost: ${total_cost:.2f}")
        typer.echo(f"{'='*50}")


if __name__ == "__main__":
    app()
