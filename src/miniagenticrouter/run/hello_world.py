import os
from pathlib import Path

import typer
import yaml

from miniagenticrouter import package_dir
from miniagenticrouter.agents.default import DefaultAgent
from miniagenticrouter.environments.local import LocalEnvironment
from miniagenticrouter.models.litellm_model import LitellmModel

app = typer.Typer()


@app.command()
def main(
    task: str = typer.Option(..., "-t", "--task", help="Task/problem statement", show_default=False, prompt=True),
    model_name: str = typer.Option(
        os.getenv("MAR_MODEL_NAME"),
        "-m",
        "--model",
        help="Model name (defaults to MAR_MODEL_NAME env var)",
        prompt="What model do you want to use?",
    ),
) -> DefaultAgent:
    agent = DefaultAgent(
        LitellmModel(model_name=model_name),
        LocalEnvironment(),
        **yaml.safe_load(Path(package_dir / "config" / "default.yaml").read_text())["agent"],
    )
    agent.run(task)
    return agent


if __name__ == "__main__":
    app()
