"""Run a single HLE (Humanity's Last Exam) task."""

from __future__ import annotations

import traceback
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.panel import Panel

from miniagenticrouter import global_config_dir
from miniagenticrouter.agents.multitool import MultiToolAgent
from miniagenticrouter.config import builtin_config_dir, get_config_path
from miniagenticrouter.data.hle_tasks import (
    enumerate_hle_tasks,
    get_hle_stats,
    get_hle_task_by_id,
    list_hle_subjects,
)
from miniagenticrouter.eval.hle_judge import LLMJudge
from miniagenticrouter.run.utils.router import create_model_or_router
from miniagenticrouter.run.utils.save import save_traj
from miniagenticrouter.run.utils.verbose import VerboseConfig, VerboseRunner
from miniagenticrouter.utils.log import logger

app = typer.Typer(add_completion=False)
console = Console()

DEFAULT_OUTPUT = global_config_dir / "last_hle_single_run.traj.json"


# fmt: off
@app.command()
def main(
    task_id: str = typer.Argument(None, help="HLE task ID"),
    index: int = typer.Option(None, "-i", "--index", help="Task index (0-based) as alternative to task_id"),
    model_name: str | None = typer.Option(None, "-m", "--model", help="Model(s) to use. For routers, use comma-separated names"),
    router: str = typer.Option("none", "-r", "--router", help="Router type: none, roulette, or interleaving"),
    config_path: Path = typer.Option(builtin_config_dir / "extra" / "hle.yaml", "-c", "--config", help="Path to config file"),
    output: Path = typer.Option(DEFAULT_OUTPUT, "-o", "--output", help="Output trajectory file"),
    list_tasks: bool = typer.Option(False, "--list-tasks", help="List first N tasks and exit"),
    list_subjects: bool = typer.Option(False, "--list-subjects", help="List available subjects and exit"),
    stats: bool = typer.Option(False, "--stats", help="Show dataset statistics and exit"),
    show_n: int = typer.Option(10, "-n", help="Number of tasks to show with --list-tasks"),
    verbose: bool = typer.Option(False, "--verbose", "-V", help="Print full interaction trace with colors"),
) -> None:
    # fmt: on
    """Run a single HLE (Humanity's Last Exam) task.

    Example:
        mtr-extra hle-single --stats
        mtr-extra hle-single --list-tasks -n 20
        mtr-extra hle-single --index 0 -V  # verbose mode with first task
        mtr-extra hle-single <task_id> -m anthropic/claude-sonnet-4-5-20250929

    Router examples:
        mtr-extra hle-single -i 0 -V -r roulette -m "openai/gpt-4o,anthropic/claude-sonnet-4-5-20250929"
    """
    # Handle info commands
    if stats:
        try:
            hle_stats = get_hle_stats()
            console.print("\n[bold cyan]HLE Dataset Statistics[/bold cyan]\n")
            console.print(f"Total questions: {hle_stats['total']}")
            console.print(f"Text-only: {hle_stats['text_only']}")
            console.print(f"Multimodal: {hle_stats['multimodal']}")
            console.print(f"\nQuestion types:")
            for qtype, count in hle_stats["question_types"].items():
                console.print(f"  {qtype}: {count}")
            console.print(f"\nSubjects ({len(hle_stats['subjects'])}):")
            for subject, count in sorted(hle_stats["subjects"].items(), key=lambda x: -x[1])[:10]:
                console.print(f"  {subject}: {count}")
            if len(hle_stats["subjects"]) > 10:
                console.print(f"  ... and {len(hle_stats['subjects']) - 10} more")
        except Exception as e:
            console.print(f"[red]Error loading dataset: {e}[/red]")
            console.print("[yellow]Make sure you have accepted the dataset terms on Hugging Face[/yellow]")
        raise typer.Exit()

    if list_subjects:
        try:
            subjects = list_hle_subjects()
            console.print("\n[bold cyan]Available HLE Subjects[/bold cyan]\n")
            for subject in subjects:
                console.print(f"  - {subject}")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit()

    if list_tasks:
        try:
            tasks = enumerate_hle_tasks(slice_spec=f"0:{show_n}")
            console.print(f"\n[bold cyan]First {len(tasks)} HLE Tasks[/bold cyan]\n")
            for i, task in enumerate(tasks):
                q_preview = task["question"][:80] + "..." if len(task["question"]) > 80 else task["question"]
                console.print(f"[{i}] {task['task_id']}: {q_preview}")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit()

    # Get task
    if task_id is None and index is None:
        typer.echo("Error: Either TASK_ID or --index is required. Use --list-tasks to see available tasks.")
        raise typer.Exit(1)

    try:
        if index is not None:
            tasks = enumerate_hle_tasks(slice_spec=f"{index}:{index+1}")
            if not tasks:
                typer.echo(f"Error: No task found at index {index}")
                raise typer.Exit(1)
            task_data = tasks[0]
        else:
            task_data = get_hle_task_by_id(task_id)
            if task_data is None:
                typer.echo(f"Error: Task '{task_id}' not found")
                raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error loading task: {e}[/red]")
        console.print("[yellow]Make sure you have accepted the dataset terms on Hugging Face[/yellow]")
        raise typer.Exit(1)

    task_id = task_data["task_id"]
    question = task_data["question"]
    ground_truth = task_data["answer"]

    logger.info(f"Loaded task: {task_id}")
    logger.info(f"Question: {question[:200]}...")
    logger.info(f"Ground truth: {ground_truth[:100]}...")

    # Load config
    config_path = get_config_path(config_path)
    logger.info(f"Loading config from '{config_path}'")
    config = yaml.safe_load(config_path.read_text())

    # Create model or router based on CLI options
    model = create_model_or_router(model_name, router, config.get("model", {}))

    if router != "none":
        logger.info(f"Using {router} router with models: {model_name}")

    # Create agent
    agent_config = config.get("agent", {})
    agent_config.pop("agent_class", None)  # Remove class spec from kwargs

    agent = MultiToolAgent(
        model=model,
        env=None,  # HLE doesn't need an environment
        **agent_config,
    )

    # Add task data to template vars
    agent.extra_template_vars["question"] = question
    agent.extra_template_vars["task_id"] = task_id
    agent.extra_template_vars["subject"] = task_data.get("subject", "")

    exit_status, result, extra_info = None, None, None
    try:
        if verbose:
            # Run with verbose output using VerboseRunner
            runner = VerboseRunner(VerboseConfig(
                benchmark_name="HLE",
                observation_role="tool",
            ))
            exit_status, result = runner.run(agent, question)
        else:
            exit_status, result = agent.run(question)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        exit_status, result = "Interrupted", "User interrupted execution"
    except Exception as e:
        logger.error(f"Error processing task {task_id}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        # Build benchmark_data for unified trajectory format
        benchmark_data = {
            "question": question,
            "ground_truth": ground_truth,
            "subject": task_data.get("subject", ""),
            "question_type": task_data.get("question_type", "short_answer"),
            "prediction": result if exit_status == "Submitted" else None,
        }

        # Run LLM Judge if enabled and we have a result
        judge_result = None
        judge_config = config.get("judge", {})
        if judge_config.get("enabled", False) and result and exit_status == "Submitted":
            try:
                console.print("\n[bold cyan]═══ Running LLM Judge ═══[/bold cyan]")
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
                benchmark_data["judge_result"] = judge_result.to_dict()
            except Exception as e:
                logger.error(f"LLM Judge failed: {e}")
                console.print(f"[red]LLM Judge error: {e}[/red]")

        extra_info = (extra_info or {}) | {
            "task_id": task_id,
            "benchmark": "hle",
            "benchmark_data": benchmark_data,
        }
        save_traj(agent, output, exit_status=exit_status, result=result, extra_info=extra_info)

    # Show summary
    console.print(f"\n[bold]Task completed[/bold]")
    console.print(f"Exit status: {exit_status}")
    console.print(f"Ground truth: {ground_truth}")
    if result:
        console.print(f"Prediction: {result[:200]}...")

    # Show Judge results if available
    if judge_result:
        color = "green" if judge_result.correct else "red"
        status = "✓ Correct" if judge_result.correct else "✗ Incorrect"
        console.print(Panel(
            f"[bold]{status}[/bold]\n\n"
            f"[bold]Extracted Answer:[/bold] {judge_result.prediction}\n"
            f"[bold]Confidence:[/bold] {judge_result.confidence:.0%}\n"
            f"[bold]Reasoning:[/bold] {judge_result.reasoning[:300]}{'...' if len(judge_result.reasoning) > 300 else ''}",
            title="[bold cyan]Judge Result[/bold cyan]",
            border_style=color,
        ))

    console.print(f"\nTrajectory saved to: {output}")


if __name__ == "__main__":
    app()
