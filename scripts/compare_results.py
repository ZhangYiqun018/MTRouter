#!/usr/bin/env python3
"""
Result comparison script for methods under trajectories/test.

Analyzes summary_test_id.json files, including:
1. Overall performance (score, cost, efficiency)
2. Per-task comparisons
3. Model-selection distribution for routing methods
4. Cost-performance frontier
5. Visualization charts
"""

import json
import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


def compute_ci(values: List[float], confidence: float = 0.95) -> Optional[Tuple[float, float]]:
    """Compute confidence interval for a list of values.

    Uses t-distribution for small samples.

    Args:
        values: List of numeric values.
        confidence: Confidence level (default 0.95 for 95% CI).

    Returns:
        Tuple of (lower, upper) bounds, or None if not enough data.
    """
    n = len(values)
    if n < 2:
        return None

    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std_err = math.sqrt(variance / n)

    # t-value for 95% CI (two-tailed)
    t_values = {
        2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
        6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262,
        15: 2.145, 20: 2.093, 30: 2.045, 50: 2.009, 100: 1.984,
    }
    if n >= 100:
        t = 1.96
    else:
        keys = sorted(t_values.keys())
        t = t_values.get(n)
        if t is None:
            for k in keys:
                if k >= n:
                    t = t_values[k]
                    break
            else:
                t = 1.96

    margin = t * std_err
    return (round(mean - margin, 2), round(mean + margin, 2))

# Try importing visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class MethodResult:
    """Result for one method."""
    name: str
    mode: str
    avg_score: float
    total_cost: float
    avg_steps: float
    avg_steps_success: float  # Average steps for successful tasks
    total_tasks: int
    submitted: int  # Number of normally submitted tasks
    limits_exceeded: int  # Number of limit-exceeded tasks
    by_task: Dict[str, Dict[str, Any]]
    by_run: Dict[str, Dict[str, Any]]  # Statistics grouped by run
    model_usage: Dict[str, Dict[str, Any]]
    config: Dict[str, Any]
    timestamp: str
    score_ci: Optional[Tuple[float, float]] = None  # 95% CI for avg_score

    @property
    def n_runs(self) -> int:
        """Number of runs"""
        return len(self.by_run)

    @property
    def cost_per_run(self) -> float:
        """Cost per run."""
        n_runs = len(self.by_run) if self.by_run else 1
        return self.total_cost / n_runs

    @property
    def efficiency(self) -> float:
        """Cost efficiency: score per dollar based on per-run cost."""
        if self.cost_per_run == 0:
            return float('inf')
        return self.avg_score / self.cost_per_run

    @property
    def completion_rate(self) -> float:
        """Normal completion rate."""
        return self.submitted / self.total_tasks if self.total_tasks > 0 else 0

    def format_score_with_ci(self) -> str:
        """Format score with CI if available"""
        if self.score_ci:
            return f"{self.avg_score:.2f} [{self.score_ci[0]:.1f}, {self.score_ci[1]:.1f}]"
        return f"{self.avg_score:.2f}"


def load_summary(summary_path: Path) -> Optional[MethodResult]:
    """Load one summary file."""
    if not summary_path.exists():
        return None

    with open(summary_path) as f:
        data = json.load(f)

    metadata = data.get('metadata', {})
    overall = data.get('overall', {})
    by_exit = data.get('by_exit_status', {})
    by_run = data.get('by_run', {})

    # Extract method name from directory name
    name = summary_path.parent.name

    # Compute CI from per-run scores if multiple runs available
    # Weight by total tasks per run (avg_score now includes all tasks, not just success)
    score_ci = None
    valid_runs = [
        (run_data.get('total', 0), run_data.get('avg_score', 0))
        for run_data in by_run.values()
        if run_data.get('total', 0) > 0
    ]
    if len(valid_runs) >= 2:
        # Use weighted average for CI calculation
        # Each run's avg_score is weighted by its total task count
        total_tasks = sum(n for n, _ in valid_runs)
        weighted_mean = sum(n * s for n, s in valid_runs) / total_tasks

        # Compute weighted variance
        weighted_var = sum(n * (s - weighted_mean) ** 2 for n, s in valid_runs) / total_tasks
        # Effective sample size for weighted data
        n_runs = len(valid_runs)
        std_err = math.sqrt(weighted_var / n_runs)

        # t-value lookup
        t_values = {
            2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
            6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262,
        }
        t = t_values.get(n_runs, 1.96 if n_runs >= 30 else 2.0)
        margin = t * std_err
        score_ci = (round(weighted_mean - margin, 2), round(weighted_mean + margin, 2))

    return MethodResult(
        name=name,
        mode=metadata.get('mode', name),
        avg_score=overall.get('avg_score', 0),
        total_cost=overall.get('total_cost', 0),
        avg_steps=overall.get('avg_steps', 0),
        avg_steps_success=overall.get('avg_steps_success', 0),
        total_tasks=overall.get('total', 0),
        submitted=by_exit.get('Submitted', 0),
        limits_exceeded=by_exit.get('LimitsExceeded', 0),
        by_task=data.get('by_task', {}),
        by_run=by_run,
        model_usage=data.get('model_usage', {}),
        config=metadata.get('config', {}),
        timestamp=metadata.get('timestamp', ''),
        score_ci=score_ci,
    )


def load_all_results(test_dir: Path) -> List[MethodResult]:
    """Load results for all methods."""
    results = []

    for method_dir in sorted(test_dir.iterdir()):
        if not method_dir.is_dir():
            continue

        summary_path = method_dir / 'summary_test_id.json'
        result = load_summary(summary_path)

        if result:
            results.append(result)
        else:
            print(f"Warning: No summary found for {method_dir.name}")

    return results


def format_table(headers: List[str], rows: List[List[str]],
                 alignments: Optional[List[str]] = None) -> str:
    """Format table output."""
    if not rows:
        return ""

    # Compute column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    # Default alignment
    if alignments is None:
        alignments = ['l'] + ['r'] * (len(headers) - 1)

    def format_cell(cell, width, align):
        s = str(cell)
        if align == 'l':
            return s.ljust(width)
        elif align == 'r':
            return s.rjust(width)
        else:
            return s.center(width)

    # Build table
    lines = []

    # Header
    header_line = ' | '.join(format_cell(h, widths[i], alignments[i])
                              for i, h in enumerate(headers))
    lines.append(header_line)

    # Separator
    sep_line = '-+-'.join('-' * w for w in widths)
    lines.append(sep_line)

    # Rows
    for row in rows:
        row_line = ' | '.join(format_cell(c, widths[i], alignments[i])
                               for i, c in enumerate(row))
        lines.append(row_line)

    return '\n'.join(lines)


def print_overall_comparison(results: List[MethodResult]):
    """Print overall comparison."""
    print("\n" + "=" * 80)
    print("Overall Performance Comparison")
    print("=" * 80)

    # Check if any result has CI (multiple runs)
    has_ci = any(r.score_ci is not None for r in results)
    score_header = 'Average Score [95% CI]' if has_ci else 'Average Score'

    headers = ['Method', score_header, 'Cost/Run($)', 'Efficiency(score/$)', 'Avg Steps',
               'Tasks', 'Runs', 'Completion Rate', 'Date']

    rows = []
    for r in sorted(results, key=lambda x: x.avg_score, reverse=True):
        rows.append([
            r.name[:35],  # Truncate long names
            r.format_score_with_ci(),
            f"{r.cost_per_run:.4f}",
            f"{r.efficiency:.2f}",
            f"{r.avg_steps:.1f}",
            str(r.total_tasks),
            str(r.n_runs) if r.n_runs > 1 else "1",
            f"{r.completion_rate:.1%}",
            r.timestamp.split('T')[0] if r.timestamp else 'N/A'
        ])

    print(format_table(headers, rows))

    # Efficiency ranking
    print("\nCost-efficiency ranking (score per dollar):")
    for i, r in enumerate(sorted(results, key=lambda x: x.efficiency, reverse=True), 1):
        print(f"  {i}. {r.name}: {r.efficiency:.2f} score/$")


def print_steps_analysis(results: List[MethodResult]):
    """Print step analysis."""
    print("\n" + "=" * 80)
    print("Step Analysis")
    print("=" * 80)

    headers = ['Method', 'Overall Avg Steps', 'Successful Task Steps', 'Failed Task Steps', 'Completion Rate']

    rows = []
    for r in sorted(results, key=lambda x: x.avg_steps):
        # Compute average steps for failed tasks
        if r.limits_exceeded > 0 and r.submitted > 0:
            # total_steps = avg_steps * total_tasks
            # success_steps = avg_steps_success * submitted
            # failed_steps = total_steps - success_steps
            total_steps = r.avg_steps * r.total_tasks
            success_steps = r.avg_steps_success * r.submitted
            failed_steps = total_steps - success_steps
            avg_steps_failed = failed_steps / r.limits_exceeded
        elif r.limits_exceeded > 0:
            avg_steps_failed = r.avg_steps  # All tasks failed
        else:
            avg_steps_failed = 0.0

        rows.append([
            r.name[:35],
            f"{r.avg_steps:.1f}",
            f"{r.avg_steps_success:.1f}" if r.submitted > 0 else "-",
            f"{avg_steps_failed:.1f}" if r.limits_exceeded > 0 else "-",
            f"{r.completion_rate:.1%}",
        ])

    print(format_table(headers, rows))

    # Step-efficiency analysis
    print("\nStep-efficiency ranking (fewer steps for successful tasks is better):")
    valid_results = [r for r in results if r.submitted > 0]
    for i, r in enumerate(sorted(valid_results, key=lambda x: x.avg_steps_success), 1):
        print(f"  {i}. {r.name}: successful tasks avg {r.avg_steps_success:.1f} steps")


def print_task_comparison(results: List[MethodResult]):
    """Print per-task comparison."""
    print("\n" + "=" * 80)
    print("Per-Task Comparison (Average Score)")
    print("=" * 80)

    # Collect all task names
    all_tasks = set()
    for r in results:
        all_tasks.update(r.by_task.keys())
    all_tasks = sorted(all_tasks)

    # Build table
    headers = ['Task'] + [r.name[:20] for r in results]
    rows = []

    for task in all_tasks:
        row = [task]
        for r in results:
            task_data = r.by_task.get(task, {})
            score = task_data.get('avg_score', '-')
            if isinstance(score, (int, float)):
                row.append(f"{score:.1f}")
            else:
                row.append('-')
        rows.append(row)

    print(format_table(headers, rows))

    # Cost comparison
    print("\nPer-Task Comparison (Cost)")
    print("-" * 60)

    headers = ['Task'] + [r.name[:20] for r in results]
    rows = []

    for task in all_tasks:
        row = [task]
        for r in results:
            task_data = r.by_task.get(task, {})
            cost = task_data.get('total_cost', '-')
            if isinstance(cost, (int, float)):
                row.append(f"${cost:.3f}")
            else:
                row.append('-')
        rows.append(row)

    print(format_table(headers, rows))


def print_model_usage(results: List[MethodResult]):
    """Print model-usage analysis."""
    print("\n" + "=" * 80)
    print("Model Usage Distribution Analysis")
    print("=" * 80)

    for r in results:
        if len(r.model_usage) <= 1:
            continue  # Skip single-model baselines

        print(f"\n{r.name}:")
        print("-" * 40)

        total_calls = sum(m['calls'] for m in r.model_usage.values())
        total_cost = sum(m['cost'] for m in r.model_usage.values())

        headers = ['Model', 'Calls', 'Call Share', 'Cost($)', 'Cost Share', 'Cost/Call']
        rows = []

        for model, usage in sorted(r.model_usage.items(),
                                    key=lambda x: x[1]['calls'], reverse=True):
            calls = usage['calls']
            cost = usage['cost']
            rows.append([
                model,
                str(calls),
                f"{calls/total_calls:.1%}" if total_calls > 0 else '-',
                f"{cost:.4f}",
                f"{cost/total_cost:.1%}" if total_cost > 0 else '-',
                f"{cost/calls:.5f}" if calls > 0 else '-'
            ])

        print(format_table(headers, rows))


def print_best_model_by_task(results: List[MethodResult]):
    """Analyze the best method for each task."""
    print("\n" + "=" * 80)
    print("Best Method by Task")
    print("=" * 80)

    # Collect all tasks
    all_tasks = set()
    for r in results:
        all_tasks.update(r.by_task.keys())

    headers = ['Task', 'Best Score Method', 'Score', 'Lowest Cost Method', 'Cost']
    rows = []

    for task in sorted(all_tasks):
        # Find highest score
        best_score_method = None
        best_score = -float('inf')

        # Find lowest cost
        best_cost_method = None
        best_cost = float('inf')

        for r in results:
            task_data = r.by_task.get(task, {})
            score = task_data.get('avg_score')
            cost = task_data.get('total_cost')

            if score is not None and score > best_score:
                best_score = score
                best_score_method = r.name[:20]

            if cost is not None and cost < best_cost:
                best_cost = cost
                best_cost_method = r.name[:20]

        rows.append([
            task,
            best_score_method or '-',
            f"{best_score:.1f}" if best_score > -float('inf') else '-',
            best_cost_method or '-',
            f"${best_cost:.3f}" if best_cost < float('inf') else '-'
        ])

    print(format_table(headers, rows))


def analyze_router_improvement(results: List[MethodResult]):
    """Analyze learned-router improvement over other methods."""
    print("\n" + "=" * 80)
    print("Learned Router Relative Improvement Analysis")
    print("=" * 80)

    # Find learned router and baselines
    learned_router = None
    baselines = []
    roulette = None

    for r in results:
        if 'learned' in r.name.lower():
            learned_router = r
        elif 'roulette' in r.name.lower():
            roulette = r
        elif 'baseline' in r.name.lower():
            baselines.append(r)

    if not learned_router:
        print("No learned-router result found")
        return

    print(f"\nLearned router: {learned_router.name}")
    print(f"  - Average score: {learned_router.avg_score:.2f}")
    print(f"  - Total cost: ${learned_router.total_cost:.4f}")
    print(f"  - Efficiency: {learned_router.efficiency:.2f} score/$")

    # Compare with baselines
    print("\nCompared with baselines:")
    for bl in baselines:
        score_diff = learned_router.avg_score - bl.avg_score
        cost_diff = learned_router.total_cost - bl.total_cost
        eff_ratio = learned_router.efficiency / bl.efficiency if bl.efficiency > 0 else float('inf')

        print(f"\n  vs {bl.name}:")
        print(f"    Score difference: {score_diff:+.2f} ({score_diff/abs(bl.avg_score)*100 if bl.avg_score != 0 else 0:+.1f}%)")
        print(f"    Cost difference: ${cost_diff:+.4f} ({cost_diff/bl.total_cost*100 if bl.total_cost > 0 else 0:+.1f}%)")
        print(f"    Efficiency ratio: {eff_ratio:.2f}x")

    # Compare with roulette
    if roulette:
        print(f"\nCompared with roulette:")
        score_diff = learned_router.avg_score - roulette.avg_score
        cost_diff = learned_router.total_cost - roulette.total_cost
        eff_ratio = learned_router.efficiency / roulette.efficiency if roulette.efficiency > 0 else float('inf')

        print(f"  Score difference: {score_diff:+.2f}")
        print(f"  Cost difference: ${cost_diff:+.4f}")
        print(f"  Efficiency ratio: {eff_ratio:.2f}x")


def generate_plots(results: List[MethodResult], output_dir: Path):
    """Generate visualization charts."""
    if not HAS_MATPLOTLIB:
        print("\nWarning: matplotlib is not installed; skipping chart generation")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Score-cost scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))

    for r in results:
        # Use different colors and markers
        if 'learned' in r.name.lower():
            color, marker = 'red', '*'
            size = 200
        elif 'roulette' in r.name.lower():
            color, marker = 'orange', 's'
            size = 150
        else:
            color, marker = 'blue', 'o'
            size = 100

        ax.scatter(r.total_cost, r.avg_score, c=color, marker=marker,
                   s=size, label=r.name[:25], alpha=0.7)

    ax.set_xlabel('Total Cost ($)', fontsize=12)
    ax.set_ylabel('Average Score', fontsize=12)
    ax.set_title('Score vs Cost Trade-off', fontsize=14)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'score_vs_cost.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'score_vs_cost.png'}")

    # 2. Per-task score comparison bar chart
    fig, ax = plt.subplots(figsize=(14, 8))

    all_tasks = sorted(set(t for r in results for t in r.by_task.keys()))
    x = range(len(all_tasks))
    width = 0.8 / len(results)

    for i, r in enumerate(results):
        scores = [r.by_task.get(t, {}).get('avg_score', 0) for t in all_tasks]
        offset = (i - len(results)/2 + 0.5) * width
        bars = ax.bar([xi + offset for xi in x], scores, width,
                      label=r.name[:20], alpha=0.8)

    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Average Score', fontsize=12)
    ax.set_title('Score Comparison by Task', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(all_tasks, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'score_by_task.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'score_by_task.png'}")

    # 3. Model-usage pie charts for routing methods only
    for r in results:
        if len(r.model_usage) <= 1:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        models = list(r.model_usage.keys())
        calls = [r.model_usage[m]['calls'] for m in models]
        costs = [r.model_usage[m]['cost'] for m in models]

        # Call-count distribution
        ax1.pie(calls, labels=models, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Model Usage by Calls')

        # Cost distribution
        ax2.pie(costs, labels=models, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Model Usage by Cost')

        fig.suptitle(f'Model Distribution: {r.name[:30]}', fontsize=12)
        plt.tight_layout()

        safe_name = r.name.replace('/', '_')
        plt.savefig(output_dir / f'model_dist_{safe_name}.png', dpi=150)
        plt.close()
        print(f"Saved: {output_dir / f'model_dist_{safe_name}.png'}")


def export_to_csv(results: List[MethodResult], output_path: Path):
    """Export results to CSV."""
    import csv

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Overall results
        writer.writerow(['Overall Results'])
        writer.writerow(['Method', 'Avg Score', 'Total Cost', 'Efficiency',
                        'Avg Steps', 'Tasks', 'Completion Rate'])
        for r in results:
            writer.writerow([r.name, r.avg_score, r.total_cost, r.efficiency,
                           r.avg_steps, r.total_tasks, r.completion_rate])

        writer.writerow([])

        # Per-task results
        writer.writerow(['By Task Results'])
        all_tasks = sorted(set(t for r in results for t in r.by_task.keys()))
        writer.writerow(['Task'] + [r.name for r in results])

        for task in all_tasks:
            row = [task]
            for r in results:
                score = r.by_task.get(task, {}).get('avg_score', '')
                row.append(score)
            writer.writerow(row)

    print(f"\nExported CSV: {output_path}")


def check_runs_recommendation(results: List[MethodResult]):
    """Print recommendations for adding more runs."""
    print("\n" + "=" * 80)
    print("Multi-Run Recommendations")
    print("=" * 80)

    for r in results:
        # Use actual runs from by_run data, not config
        runs = r.n_runs
        print(f"\n{r.name}:")
        print(f"  Current runs: {runs}")

        if runs < 3:
            print(f"  [Suggestion] Increase to at least 3 runs for more reliable statistics")
            print(f"         Estimated extra cost: ${r.cost_per_run * (3 - runs):.2f}")
        else:
            print(f"  [OK] Enough runs are available")

        # Check whether the method has routing randomness
        if len(r.model_usage) > 1:
            print(f"  [Note] This method uses multi-model routing and can be more stochastic; consider adding runs")


def main():
    parser = argparse.ArgumentParser(description='Compare experiment results under trajectories/test')
    parser.add_argument('--test-dir', type=Path,
                       default=Path('trajectories/test'),
                       help='Path to the test result directory')
    parser.add_argument('--output-dir', type=Path,
                       default=Path('results/comparison'),
                       help='Chart output directory')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip chart generation')
    parser.add_argument('--export-csv', type=Path,
                       help='Export results to a CSV file')
    args = parser.parse_args()

    # Load all results
    print(f"Loading results from {args.test_dir}...")
    results = load_all_results(args.test_dir)

    if not results:
        print("No result files found!")
        return 1

    print(f"Found {len(results)} method results")

    # Print comparisons
    print_overall_comparison(results)
    print_steps_analysis(results)
    print_task_comparison(results)
    print_model_usage(results)
    print_best_model_by_task(results)
    analyze_router_improvement(results)
    check_runs_recommendation(results)

    # Generate charts
    if not args.no_plots:
        generate_plots(results, args.output_dir)

    # Export CSV
    if args.export_csv:
        export_to_csv(results, args.export_csv)

    return 0


if __name__ == '__main__':
    exit(main())
