#!/usr/bin/env python3
"""
结果对比脚本 - 对比 trajectories/test 目录下的不同方案结果

基于 summary_test_id.json 文件进行分析，包括：
1. 整体性能对比（分数、成本、效率）
2. 按任务分组对比
3. 模型选择分布分析（针对路由器方案）
4. 成本-性能前沿分析
5. 可视化图表
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

# 尝试导入可视化库
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
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
    """单个方案的结果"""
    name: str
    mode: str
    avg_score: float
    total_cost: float
    avg_steps: float
    avg_steps_success: float  # 成功任务的平均步数
    total_tasks: int
    submitted: int  # 正常提交完成的任务数
    limits_exceeded: int  # 超限的任务数
    by_task: Dict[str, Dict[str, Any]]
    by_run: Dict[str, Dict[str, Any]]  # 按 run 分组的统计
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
        """每个 run 的成本"""
        n_runs = len(self.by_run) if self.by_run else 1
        return self.total_cost / n_runs

    @property
    def efficiency(self) -> float:
        """成本效率: score per dollar (基于 per-run 成本)"""
        if self.cost_per_run == 0:
            return float('inf')
        return self.avg_score / self.cost_per_run

    @property
    def completion_rate(self) -> float:
        """正常完成率"""
        return self.submitted / self.total_tasks if self.total_tasks > 0 else 0

    def format_score_with_ci(self) -> str:
        """Format score with CI if available"""
        if self.score_ci:
            return f"{self.avg_score:.2f} [{self.score_ci[0]:.1f}, {self.score_ci[1]:.1f}]"
        return f"{self.avg_score:.2f}"


def load_summary(summary_path: Path) -> Optional[MethodResult]:
    """加载单个 summary 文件"""
    if not summary_path.exists():
        return None

    with open(summary_path) as f:
        data = json.load(f)

    metadata = data.get('metadata', {})
    overall = data.get('overall', {})
    by_exit = data.get('by_exit_status', {})
    by_run = data.get('by_run', {})

    # 提取方案名称（从目录名）
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
    """加载所有方案的结果"""
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
    """格式化表格输出"""
    if not rows:
        return ""

    # 计算每列宽度
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    # 默认对齐方式
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

    # 构建表格
    lines = []

    # 表头
    header_line = ' | '.join(format_cell(h, widths[i], alignments[i])
                              for i, h in enumerate(headers))
    lines.append(header_line)

    # 分隔线
    sep_line = '-+-'.join('-' * w for w in widths)
    lines.append(sep_line)

    # 数据行
    for row in rows:
        row_line = ' | '.join(format_cell(c, widths[i], alignments[i])
                               for i, c in enumerate(row))
        lines.append(row_line)

    return '\n'.join(lines)


def print_overall_comparison(results: List[MethodResult]):
    """打印整体对比"""
    print("\n" + "=" * 80)
    print("整体性能对比")
    print("=" * 80)

    # Check if any result has CI (multiple runs)
    has_ci = any(r.score_ci is not None for r in results)
    score_header = '平均分数 [95% CI]' if has_ci else '平均分数'

    headers = ['方案', score_header, '成本/Run($)', '效率(score/$)', '平均步数',
               '任务数', 'Runs', '完成率', '时间']

    rows = []
    for r in sorted(results, key=lambda x: x.avg_score, reverse=True):
        rows.append([
            r.name[:35],  # 截断长名称
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

    # 效率排名
    print("\n按成本效率排名（score per dollar）:")
    for i, r in enumerate(sorted(results, key=lambda x: x.efficiency, reverse=True), 1):
        print(f"  {i}. {r.name}: {r.efficiency:.2f} score/$")


def print_steps_analysis(results: List[MethodResult]):
    """打印步数分析"""
    print("\n" + "=" * 80)
    print("步数分析")
    print("=" * 80)

    headers = ['方案', '总体平均步数', '成功任务步数', '失败任务步数', '完成率']

    rows = []
    for r in sorted(results, key=lambda x: x.avg_steps):
        # 计算失败任务的平均步数
        if r.limits_exceeded > 0 and r.submitted > 0:
            # total_steps = avg_steps * total_tasks
            # success_steps = avg_steps_success * submitted
            # failed_steps = total_steps - success_steps
            total_steps = r.avg_steps * r.total_tasks
            success_steps = r.avg_steps_success * r.submitted
            failed_steps = total_steps - success_steps
            avg_steps_failed = failed_steps / r.limits_exceeded
        elif r.limits_exceeded > 0:
            avg_steps_failed = r.avg_steps  # 全是失败的
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

    # 步数效率分析
    print("\n步数效率分析（成功任务完成越快越好）:")
    valid_results = [r for r in results if r.submitted > 0]
    for i, r in enumerate(sorted(valid_results, key=lambda x: x.avg_steps_success), 1):
        print(f"  {i}. {r.name}: 成功任务平均 {r.avg_steps_success:.1f} 步")


def print_task_comparison(results: List[MethodResult]):
    """打印按任务分组的对比"""
    print("\n" + "=" * 80)
    print("按任务类型对比 (平均分数)")
    print("=" * 80)

    # 收集所有任务名称
    all_tasks = set()
    for r in results:
        all_tasks.update(r.by_task.keys())
    all_tasks = sorted(all_tasks)

    # 构建表格
    headers = ['任务'] + [r.name[:20] for r in results]
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

    # 成本对比
    print("\n按任务类型对比 (成本)")
    print("-" * 60)

    headers = ['任务'] + [r.name[:20] for r in results]
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
    """打印模型使用情况分析"""
    print("\n" + "=" * 80)
    print("模型使用分布分析")
    print("=" * 80)

    for r in results:
        if len(r.model_usage) <= 1:
            continue  # 跳过单模型 baseline

        print(f"\n{r.name}:")
        print("-" * 40)

        total_calls = sum(m['calls'] for m in r.model_usage.values())
        total_cost = sum(m['cost'] for m in r.model_usage.values())

        headers = ['模型', '调用次数', '调用占比', '成本($)', '成本占比', '单次成本']
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
    """分析每个任务的最佳模型"""
    print("\n" + "=" * 80)
    print("每个任务的最佳方案")
    print("=" * 80)

    # 收集所有任务
    all_tasks = set()
    for r in results:
        all_tasks.update(r.by_task.keys())

    headers = ['任务', '最高分方案', '分数', '最低成本方案', '成本']
    rows = []

    for task in sorted(all_tasks):
        # 找最高分
        best_score_method = None
        best_score = -float('inf')

        # 找最低成本
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
    """分析学习路由器相对于其他方案的改进"""
    print("\n" + "=" * 80)
    print("学习路由器相对改进分析")
    print("=" * 80)

    # 找到学习路由器和各 baseline
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
        print("未找到学习路由器结果")
        return

    print(f"\n学习路由器: {learned_router.name}")
    print(f"  - 平均分数: {learned_router.avg_score:.2f}")
    print(f"  - 总成本: ${learned_router.total_cost:.4f}")
    print(f"  - 效率: {learned_router.efficiency:.2f} score/$")

    # 对比各 baseline
    print("\n与 Baseline 对比:")
    for bl in baselines:
        score_diff = learned_router.avg_score - bl.avg_score
        cost_diff = learned_router.total_cost - bl.total_cost
        eff_ratio = learned_router.efficiency / bl.efficiency if bl.efficiency > 0 else float('inf')

        print(f"\n  vs {bl.name}:")
        print(f"    分数差异: {score_diff:+.2f} ({score_diff/abs(bl.avg_score)*100 if bl.avg_score != 0 else 0:+.1f}%)")
        print(f"    成本差异: ${cost_diff:+.4f} ({cost_diff/bl.total_cost*100 if bl.total_cost > 0 else 0:+.1f}%)")
        print(f"    效率比: {eff_ratio:.2f}x")

    # 对比 roulette
    if roulette:
        print(f"\n与 Roulette 对比:")
        score_diff = learned_router.avg_score - roulette.avg_score
        cost_diff = learned_router.total_cost - roulette.total_cost
        eff_ratio = learned_router.efficiency / roulette.efficiency if roulette.efficiency > 0 else float('inf')

        print(f"  分数差异: {score_diff:+.2f}")
        print(f"  成本差异: ${cost_diff:+.4f}")
        print(f"  效率比: {eff_ratio:.2f}x")


def generate_plots(results: List[MethodResult], output_dir: Path):
    """生成可视化图表"""
    if not HAS_MATPLOTLIB:
        print("\nWarning: matplotlib 未安装，跳过图表生成")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 分数-成本散点图
    fig, ax = plt.subplots(figsize=(10, 6))

    for r in results:
        # 使用不同颜色和标记
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
    print(f"已保存: {output_dir / 'score_vs_cost.png'}")

    # 2. 按任务分数对比柱状图
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
    print(f"已保存: {output_dir / 'score_by_task.png'}")

    # 3. 模型使用分布饼图（仅路由器方案）
    for r in results:
        if len(r.model_usage) <= 1:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        models = list(r.model_usage.keys())
        calls = [r.model_usage[m]['calls'] for m in models]
        costs = [r.model_usage[m]['cost'] for m in models]

        # 调用次数分布
        ax1.pie(calls, labels=models, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Model Usage by Calls')

        # 成本分布
        ax2.pie(costs, labels=models, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Model Usage by Cost')

        fig.suptitle(f'Model Distribution: {r.name[:30]}', fontsize=12)
        plt.tight_layout()

        safe_name = r.name.replace('/', '_')
        plt.savefig(output_dir / f'model_dist_{safe_name}.png', dpi=150)
        plt.close()
        print(f"已保存: {output_dir / f'model_dist_{safe_name}.png'}")


def export_to_csv(results: List[MethodResult], output_path: Path):
    """导出结果到 CSV"""
    import csv

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # 整体结果
        writer.writerow(['Overall Results'])
        writer.writerow(['Method', 'Avg Score', 'Total Cost', 'Efficiency',
                        'Avg Steps', 'Tasks', 'Completion Rate'])
        for r in results:
            writer.writerow([r.name, r.avg_score, r.total_cost, r.efficiency,
                           r.avg_steps, r.total_tasks, r.completion_rate])

        writer.writerow([])

        # 按任务结果
        writer.writerow(['By Task Results'])
        all_tasks = sorted(set(t for r in results for t in r.by_task.keys()))
        writer.writerow(['Task'] + [r.name for r in results])

        for task in all_tasks:
            row = [task]
            for r in results:
                score = r.by_task.get(task, {}).get('avg_score', '')
                row.append(score)
            writer.writerow(row)

    print(f"\n已导出 CSV: {output_path}")


def check_runs_recommendation(results: List[MethodResult]):
    """检查是否需要更多 runs 的建议"""
    print("\n" + "=" * 80)
    print("多次 Run 建议")
    print("=" * 80)

    for r in results:
        # Use actual runs from by_run data, not config
        runs = r.n_runs
        print(f"\n{r.name}:")
        print(f"  当前 runs: {runs}")

        if runs < 3:
            print(f"  [建议] 增加到至少 3 runs 以获得更可靠的统计")
            print(f"         预计额外成本: ${r.cost_per_run * (3 - runs):.2f}")
        else:
            print(f"  [OK] 已有足够的 runs 数量")

        # 检查模型是否有随机性
        if len(r.model_usage) > 1:
            print(f"  [注意] 该方案使用多模型路由，随机性较大，建议增加 runs")


def main():
    parser = argparse.ArgumentParser(description='对比 trajectories/test 下的实验结果')
    parser.add_argument('--test-dir', type=Path,
                       default=Path('trajectories/test'),
                       help='测试结果目录路径')
    parser.add_argument('--output-dir', type=Path,
                       default=Path('results/comparison'),
                       help='图表输出目录')
    parser.add_argument('--no-plots', action='store_true',
                       help='跳过图表生成')
    parser.add_argument('--export-csv', type=Path,
                       help='导出结果到 CSV 文件')
    args = parser.parse_args()

    # 加载所有结果
    print(f"从 {args.test_dir} 加载结果...")
    results = load_all_results(args.test_dir)

    if not results:
        print("未找到任何结果文件！")
        return 1

    print(f"找到 {len(results)} 个方案的结果")

    # 打印各种对比
    print_overall_comparison(results)
    print_steps_analysis(results)
    print_task_comparison(results)
    print_model_usage(results)
    print_best_model_by_task(results)
    analyze_router_improvement(results)
    check_runs_recommendation(results)

    # 生成图表
    if not args.no_plots:
        generate_plots(results, args.output_dir)

    # 导出 CSV
    if args.export_csv:
        export_to_csv(results, args.export_csv)

    return 0


if __name__ == '__main__':
    exit(main())
