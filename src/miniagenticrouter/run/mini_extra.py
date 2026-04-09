#!/usr/bin/env python3

import sys
from importlib import import_module

from rich.console import Console

subcommands = [
    ("miniagenticrouter.run.extra.config", ["config"], "Manage the global config file"),
    ("miniagenticrouter.run.inspector", ["inspect", "i", "inspector"], "Run inspector (browse trajectories)"),
    ("miniagenticrouter.run.github_issue", ["github-issue", "gh"], "Run on a GitHub issue"),
    ("miniagenticrouter.run.extra.swebench", ["swebench"], "Evaluate on SWE-bench (batch mode)"),
    ("miniagenticrouter.run.extra.swebench_single", ["swebench-single"], "Evaluate on SWE-bench (single instance)"),
    ("miniagenticrouter.run.extra.scienceworld", ["scienceworld", "sw"], "Evaluate on ScienceWorld (batch mode)"),
    ("miniagenticrouter.run.extra.scienceworld_single", ["scienceworld-single", "sw-single"], "Evaluate on ScienceWorld (single task)"),
    ("miniagenticrouter.run.extra.hle", ["hle"], "Evaluate on HLE (batch mode)"),
    ("miniagenticrouter.run.extra.hle_single", ["hle-single"], "Evaluate on HLE (single question)"),
]


def get_docstring() -> str:
    lines = [
        "This is the [yellow]central entry point for all extra commands[/yellow] from MTRouter.",
        "",
        "Available sub-commands:",
        "",
    ]
    for _, aliases, description in subcommands:
        alias_text = " or ".join(f"[bold green]{alias}[/bold green]" for alias in aliases)
        lines.append(f"  {alias_text}: {description}")
    return "\n".join(lines)


def main():
    args = sys.argv[1:]

    if len(args) == 0 or len(args) == 1 and args[0] in ["-h", "--help"]:
        return Console().print(get_docstring())

    for module_path, aliases, _ in subcommands:
        if args[0] in aliases:
            return import_module(module_path).app(args[1:], prog_name=f"mtr-extra {aliases[0]}")

    return Console().print(get_docstring())


if __name__ == "__main__":
    main()
