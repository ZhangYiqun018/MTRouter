"""Run a single ScienceWorld task."""

from __future__ import annotations

import traceback
from pathlib import Path

import typer
import yaml

from miniagenticrouter import global_config_dir
from miniagenticrouter.agents.flexible import FlexibleAgent
from miniagenticrouter.config import builtin_config_dir, get_config_path
from miniagenticrouter.data.scienceworld_tasks import get_scienceworld_task_names
from miniagenticrouter.environments.extra.scienceworld import ScienceWorldEnvironment
from miniagenticrouter.run.utils.router import RouterType, create_model_or_router
from miniagenticrouter.run.utils.save import save_traj
from miniagenticrouter.run.utils.verbose import VerboseConfig, VerboseRunner
from miniagenticrouter.utils.log import logger

app = typer.Typer(add_completion=False)

DEFAULT_OUTPUT = global_config_dir / "last_scienceworld_single_run.traj.json"


# fmt: off
@app.command()
def main(
    task_name: str = typer.Argument(None, help="ScienceWorld task name (e.g., 'boil', 'melt-ice')"),
    variation: int = typer.Option(0, "-v", "--variation", help="Task variation index"),
    simplification: str = typer.Option("", "--simplification", help="Simplification string for the task"),
    model_name: str | None = typer.Option(None, "-m", "--model", help="Model(s) to use. For routers, use comma-separated names (e.g., 'model1,model2')"),
    router: str = typer.Option("none", "-r", "--router", help="Router type: none, roulette, or interleaving"),
    config_path: Path = typer.Option(builtin_config_dir / "extra" / "scienceworld.yaml", "-c", "--config", help="Path to config file"),
    output: Path = typer.Option(DEFAULT_OUTPUT, "-o", "--output", help="Output trajectory file"),
    list_tasks: bool = typer.Option(False, "--list-tasks", help="List available task names and exit"),
    verbose: bool = typer.Option(False, "--verbose", "-V", help="Print full interaction trace with colors"),
) -> None:
    # fmt: on
    """Run a single ScienceWorld task.

    Example:
        mar-extra scienceworld-single boil --variation 0
        mar-extra scienceworld-single --list-tasks
        mar-extra scienceworld-single boil -V  # verbose mode

    Router examples:
        mar-extra scienceworld-single boil -V -r roulette -m "openai/gpt-4o,claude/claude-sonnet-4-5-20250929"
        mar-extra scienceworld-single boil -r interleaving -m "model1,model2"
    """
    if list_tasks:
        task_names = get_scienceworld_task_names()
        typer.echo("Available ScienceWorld tasks:")
        for name in sorted(task_names):
            typer.echo(f"  - {name}")
        raise typer.Exit()

    if task_name is None:
        typer.echo("Error: TASK_NAME is required. Use --list-tasks to see available tasks.")
        raise typer.Exit(1)

    # Load config
    config_path = get_config_path(config_path)
    logger.info(f"Loading config from '{config_path}'")
    config = yaml.safe_load(config_path.read_text())

    # Create environment
    env_config = config.get("environment", {})
    env = ScienceWorldEnvironment(
        task_name=task_name,
        variation_idx=variation,
        simplification_str=simplification or env_config.get("simplification_str", ""),
    )

    # Load the task and get initial observation
    task_info = env.load_task(task_name, variation)
    initial_observation = task_info["observation"]
    task_description = task_info["task_description"]

    logger.info(f"Loaded task: {task_name} (variation {variation})")
    logger.info(f"Task description: {task_description[:200]}...")

    # Create agent
    agent_config = config.get("agent", {})
    agent_config.pop("agent_class", None)  # Remove class spec from kwargs

    # Create model or router based on CLI options
    model = create_model_or_router(model_name, router, config.get("model", {}))

    if router != "none":
        logger.info(f"Using {router} router with models: {model_name}")

    agent = FlexibleAgent(
        model=model,
        env=env,
        **agent_config,
    )

    # Add initial observation to template vars
    # Note: task_description is already provided by env.get_template_vars()
    agent.extra_template_vars["initial_observation"] = initial_observation

    exit_status, result, extra_info = None, None, None
    try:
        if verbose:
            # Run with verbose output using VerboseRunner
            def on_finish(env_arg, e):
                if env_arg:
                    return f"[bold cyan]Final score: {env_arg.get_score():.2f}[/bold cyan]"
                return None

            runner = VerboseRunner(VerboseConfig(
                benchmark_name="ScienceWorld",
                on_finish=on_finish,
            ))
            exit_status, result = runner.run(agent, task_description, env=env)
        else:
            exit_status, result = agent.run(task_description)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        exit_status, result = "Interrupted", "User interrupted execution"
    except Exception as e:
        logger.error(f"Error processing task {task_name}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        final_score = env.get_score()
        extra_info = (extra_info or {}) | {
            "task_name": task_name,
            "variation_idx": variation,
            "final_score": final_score,
        }
        save_traj(agent, output, exit_status=exit_status, result=result, extra_info=extra_info)
        env.close()

    logger.info(f"Task completed. Exit status: {exit_status}, Final score: {final_score}")
    typer.echo(f"\nFinal score: {final_score}")
    typer.echo(f"Trajectory saved to: {output}")


if __name__ == "__main__":
    app()
