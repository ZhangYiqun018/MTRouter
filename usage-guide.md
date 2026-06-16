# MTRouter Usage Guide

MTRouter is a cost-aware multi-turn LLM routing framework for choosing the best model during an agent trajectory.
The package still imports as `miniagenticrouter` for compatibility; the preferred CLI aliases are `mtr` and `mtr-extra`.
Legacy aliases `mar` and `mar-extra` remain available.

This guide covers installation, model configuration, benchmark usage, and the research workflow.
For the research implementation details, see `src/miniagenticrouter/research/README.md`.

## Installation

```bash
# Base package
pip install -e .

# Research dependencies for training and evaluation utilities
pip install -e '.[research]'

# Benchmark-specific extras
pip install -e '.[scienceworld]'
pip install -e '.[hle]'

# Everything
pip install -e '.[full]'
```

ScienceWorld requires a Java runtime. On Ubuntu/Debian:

```bash
sudo apt-get install default-jre
```

## API Keys and Model Configuration

Set provider keys through the config helper or environment variables:

```bash
mtr-extra config setup
```

Common environment variables:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export DEEPSEEK_API_KEY=...
export SERPER_API_KEY=...   # HLE search tool
export JINA_API_KEY=...     # Optional HLE browse summarization
```

Model metadata and pricing are defined in:

```text
src/miniagenticrouter/config/models/custom_models.yaml
```

The research model pool is defined in:

```text
src/miniagenticrouter/config/research/training.yaml
```

The paper-aligned model pool is:

```text
openai/gpt-5
deepseek/deepseek-v3.2
minimax/minimax-m2
moonshotai/kimi-k2-0905
google/gemini-2.5-flash-lite
openai/gpt-oss-120b
```

Keep model names and order consistent between data collection, training, and checkpoint loading.

## ScienceWorld CLI

Run one task:

```bash
mtr-extra scienceworld-single boil \
  --variation 0 \
  -m openai/gpt-5
```

Run a batch:

```bash
mtr-extra scienceworld \
  --task 'boil*' \
  --workers 4 \
  --model openai/gpt-5 \
  -o ./scienceworld_results
```

The paper research workflow uses `scripts/collect_data.py`, which handles fixed splits and research summaries:

```bash
PYTHONPATH=src python scripts/collect_data.py --first-try --dry-run
```

Collect roulette trajectories:

```bash
PYTHONPATH=src python scripts/collect_data.py \
  --mode roulette \
  --split train \
  --first-try \
  --runs 1 \
  --workers 4 \
  --output-dir trajectories/scienceworld
```

Collect a single-model baseline:

```bash
PYTHONPATH=src python scripts/collect_data.py \
  --mode baseline \
  --model openai/gpt-5 \
  --split train \
  --first-try \
  --runs 1 \
  --workers 4 \
  --output-dir trajectories/scienceworld
```

ScienceWorld research defaults are configured in `data_split.yaml`; paper-style evaluation uses a 50-step limit and a USD 2.0 per-episode budget.

## HLE CLI

HLE uses a multi-tool agent with `search`, `python`, `browse`, and `answer` tools.
Research collection uses `scripts/collect_hle_data.py`:

```bash
PYTHONPATH=src python scripts/collect_hle_data.py --dry-run
```

Collect roulette trajectories:

```bash
PYTHONPATH=src python scripts/collect_hle_data.py \
  --mode roulette \
  --split train \
  --runs 1 \
  --workers 4 \
  --output-dir trajectories/hle
```

Collect a single-model baseline:

```bash
PYTHONPATH=src python scripts/collect_hle_data.py \
  --mode baseline \
  --model openai/gpt-5 \
  --split train \
  --runs 1 \
  --workers 4 \
  --output-dir trajectories/hle
```

HLE research defaults are configured in `hle_data_split.yaml`; paper-style evaluation uses a 30-step limit and a USD 2.0 per-episode budget.

## Training MTRouter

### 1. Precompute Embeddings

```bash
PYTHONPATH=src python scripts/precompute_embeddings.py \
  --data-dir trajectories/scienceworld/roulette_propensity \
  --output embeddings/scienceworld_roulette_propensity.pt
```

The default research config uses `Qwen/Qwen3-Embedding-0.6B` with up to 8192 tokens.
If `history_encoder.backend` is `vllm`, make sure the vLLM embedding server is running and its served model ID matches `vllm_model_id`.

### 2. Train the Neural Router

```bash
PYTHONPATH=src python scripts/train_q_function.py \
  --data-dir trajectories/scienceworld/roulette_propensity \
  --precomputed-path embeddings/scienceworld_roulette_propensity.pt \
  --output-dir outputs/q_function \
  --epochs 100 \
  --batch-size 64 \
  --lambda_ 0.0
```

`lambda_` defaults to `0.0` for the paper's single outcome/value head.
Non-zero values are retained for legacy ablations, not for the default MTRouter objective.

Enable AEC targets when desired:

```bash
PYTHONPATH=src python scripts/train_q_function.py \
  --data-dir trajectories/scienceworld/roulette_propensity \
  --precomputed-path embeddings/scienceworld_roulette_propensity.pt \
  --output-dir outputs/q_function_aec \
  --aec-enabled \
  --lambda_ 0.0
```

### 3. Evaluate the Learned Router

```bash
PYTHONPATH=src python scripts/collect_data.py \
  --mode learned \
  --checkpoint outputs/q_function/q_function_best.pt \
  --split test_id \
  --first-try \
  --runs 1 \
  --workers 4 \
  --use-batching \
  --output-dir trajectories/eval
```

Use `--split ood_test` for OOD ScienceWorld tasks.
For HLE, use `scripts/collect_hle_data.py` with the same `--mode learned` and `--checkpoint` pattern.

## Baseline Routers

Baseline routers fit separate score and cost estimates and combine them as `score - lambda * cost` during inference.
They are comparison baselines, not the default neural MTRouter objective.

Per-model Ridge:

```bash
PYTHONPATH=src python scripts/train_per_model_ridge.py \
  --precomputed-path embeddings/scienceworld_roulette_propensity.pt \
  --data-dir trajectories/scienceworld/roulette_propensity \
  --output-dir outputs/per_model_ridge
```

Per-model XGBoost:

```bash
PYTHONPATH=src python scripts/train_per_model_xgboost.py \
  --precomputed-path embeddings/scienceworld_roulette_propensity.pt \
  --data-dir trajectories/scienceworld/roulette_propensity \
  --output-dir outputs/per_model_xgboost
```

Cluster lookup baseline:

```bash
PYTHONPATH=src python scripts/train_cluster_baseline.py \
  --precomputed-path embeddings/scienceworld_roulette_propensity.pt \
  --data-dir trajectories/scienceworld/roulette_propensity \
  --output-dir outputs/cluster_baseline \
  --n-clusters 32
```

Evaluate a baseline:

```bash
PYTHONPATH=src python scripts/collect_data.py \
  --mode per_model_ridge \
  --model-dir outputs/per_model_ridge \
  --lambda 1.0 \
  --split test_id \
  --first-try \
  --output-dir trajectories/eval
```

## Comparing Results

Evaluation scripts write `summary_<split>.json` files under each method directory.
Compare them with:

```bash
PYTHONPATH=src python scripts/compare_results.py \
  --test-dir trajectories/eval \
  --output-dir results/comparison \
  --export-csv results/comparison/results.csv
```

The comparison script reports score, cost, score-per-dollar efficiency, step counts, completion rate, per-task breakdowns, and model usage for routers.

## SWE-bench Compatibility

This repository keeps mini-swe-agent compatibility utilities.
They are not part of the MTRouter paper experiments, but the CLI remains available.

Batch mode:

```bash
mtr-extra swebench \
  --model openai/gpt-5 \
  --subset verified \
  --split test \
  --workers 4
```

Single instance:

```bash
mtr-extra swebench-single \
  --subset verified \
  --split test \
  --model openai/gpt-5 \
  -i sympy__sympy-15599
```

Useful options include `--slice`, `--filter`, `--shuffle`, `--redo-existing`, `--output`, and `--workers`.

## Troubleshooting

- If checkpoint loading reports a model-name mismatch, verify that `training.yaml` has the same model pool and order used during training.
- If online inference uses different embeddings from training, align `history_encoder.backend`, `model_name`, `vllm_model_id`, `max_tokens`, and pooling settings.
- If XGBoost training fails with an import error, run `pip install -e '.[research]'` or install `xgboost` directly.
- If collection skips trajectories, pass `--force` to overwrite existing results.
- If HLE tools fail, check `SERPER_API_KEY`, `JINA_API_KEY`, and the network/proxy settings.
