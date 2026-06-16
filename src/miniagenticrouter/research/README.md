# Research Module

This package contains the research workflow for MTRouter: cost-aware multi-turn LLM routing with history-model joint embeddings.
It is intended for reproducing and extending the experiments described in the MTRouter paper (https://arxiv.org/abs/2604.23530).

## What This Module Provides

- Task splits for ScienceWorld and HLE with train, validation, in-distribution test, and OOD test partitions.
- Trajectory collection for single-model baselines, uniform random routing, learned routing, beta-mixed exploration, and baseline routers.
- Trajectory parsing into per-turn training samples with benchmark-specific score and cost normalization.
- Neural MTRouter training with a single outcome/value head.
- Baseline trainers for per-model Ridge, per-model XGBoost, and cluster lookup routers.
- Batched inference wrappers for faster learned-router evaluation.
- Result aggregation and plotting utilities.

## Paper-Aligned Modeling Choices

MTRouter treats model selection at turn `t` as an action `a_t` conditioned on the observable history `h_t`.
The learned scorer estimates a scalar terminal-outcome value for each candidate model:

```text
s_hat_theta(h_t, a)
```

The paper trains this single value head on terminal outcome targets rather than adding a separate cost penalty to the target.
Cost awareness comes from the fixed per-episode budget, the turn limit, and the terminal score observed in completed trajectories.
The repository still keeps a `lambda_` field for checkpoint compatibility and for baseline/ablation routers that explicitly combine predicted score and cost, but the paper/default neural setting uses `lambda_: 0.0`.

Other paper-aligned defaults:

- History encoder: frozen `Qwen/Qwen3-Embedding-0.6B`, 1024-dimensional embeddings, up to 8192 tokens.
- History truncation: keep the task block and retain recent interaction context within the token budget.
- Model encoder: 8 metadata features, 32-dimensional attribute embedding, 16-dimensional learned residual, 64-dimensional output.
- Residual regularization: L2 penalty with weight `0.001`.
- Training: AdamW, learning rate `1e-3`, weight decay `0.01`, cosine schedule, batch size 64, 100 epochs, early stopping patience 3.
- Annealed Error Cost (AEC): optional time-dependent penalty for format/tool/action errors.
- Evaluation budgets: ScienceWorld uses at most 50 steps and HLE uses at most 30 steps; both use a USD 2.0 per-episode budget.

## Repository Map

```text
src/miniagenticrouter/research/
  data/                    Task split utilities for ScienceWorld and HLE
  trajectory/              Trajectory parsing and normalized sample extraction
  collection/              Collection modes and collectors
  routers/                 Learned router interface and baseline Q-functions
  training/                Encoders, datasets, neural Q-network, batching
  utils/                   Config loading helpers
```

Key scripts:

```text
scripts/collect_data.py              ScienceWorld collection/evaluation
scripts/collect_hle_data.py          HLE collection/evaluation
scripts/precompute_embeddings.py     Offline history-embedding precomputation
scripts/train_q_function.py          Neural MTRouter training
scripts/train_per_model_ridge.py     Per-model Ridge baseline training
scripts/train_per_model_xgboost.py   Per-model XGBoost baseline training
scripts/train_cluster_baseline.py    Cluster lookup baseline training
scripts/compare_results.py           Summary aggregation and plots
```

## Configuration Files

```text
src/miniagenticrouter/config/research/data_split.yaml       ScienceWorld splits and collection defaults
src/miniagenticrouter/config/research/hle_data_split.yaml   HLE splits and collection defaults
src/miniagenticrouter/config/research/training.yaml         Model pool, encoder, optimizer, and precompute defaults
src/miniagenticrouter/config/models/custom_models.yaml      Model metadata and pricing
src/miniagenticrouter/config/extra/scienceworld.yaml        ScienceWorld agent defaults
src/miniagenticrouter/config/extra/hle.yaml                 HLE multi-tool agent defaults
```

The model pool in `training.yaml` is the training/inference source of truth for learned and baseline routers.
The roulette and single-model collection pools are defined in the benchmark split YAML files.
Model names and ordering must match between data collection, training, and checkpoint loading.

## Installation

```bash
pip install -e '.[research]'
```

Install benchmark extras as needed:

```bash
pip install -e '.[scienceworld]'
pip install -e '.[hle]'
```

Useful optional services:

- A vLLM embedding server for `Qwen/Qwen3-Embedding-0.6B` if `history_encoder.backend: vllm`.
- API keys for routed LLM providers.
- `SERPER_API_KEY` and `JINA_API_KEY` for HLE web tools if those tools are enabled.

Do not commit trajectory directories or `.env` files. Trajectories may contain provider metadata, endpoints, prompts, responses, and cost records.

## End-to-End ScienceWorld Workflow

### 1. Inspect the split

```bash
PYTHONPATH=src python scripts/collect_data.py --first-try --dry-run
```

### 2. Collect exploration data

Uniform random routing records propensities and is the main offline training source.

```bash
PYTHONPATH=src python scripts/collect_data.py \
  --mode roulette \
  --split train \
  --first-try \
  --runs 1 \
  --workers 4 \
  --output-dir trajectories/scienceworld
```

### 3. Collect single-model baselines

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

Repeat for each model in the paper model pool if you want the same coverage as the full experiment.

### 4. Precompute history embeddings

```bash
PYTHONPATH=src python scripts/precompute_embeddings.py \
  --data-dir trajectories/scienceworld/roulette_propensity \
  --output embeddings/scienceworld_roulette_propensity.pt
```

Use the same encoder backend and model ID for precomputation and online inference.
The checkpoint stores encoder config metadata so mismatches can be detected later.

### 5. Train the neural router

```bash
PYTHONPATH=src python scripts/train_q_function.py \
  --data-dir trajectories/scienceworld/roulette_propensity \
  --precomputed-path embeddings/scienceworld_roulette_propensity.pt \
  --output-dir outputs/q_function \
  --epochs 100 \
  --batch-size 64 \
  --lambda_ 0.0
```

Enable AEC if you want the paper-style error-adjusted target:

```bash
PYTHONPATH=src python scripts/train_q_function.py \
  --data-dir trajectories/scienceworld/roulette_propensity \
  --precomputed-path embeddings/scienceworld_roulette_propensity.pt \
  --output-dir outputs/q_function_aec \
  --aec-enabled \
  --lambda_ 0.0
```

### 6. Evaluate the learned router

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

For OOD ScienceWorld evaluation, use `--split ood_test`.

## HLE Workflow

HLE uses the same router/training code but a different collector and split file.
The default research budget is 30 steps and USD 2.0 per episode.

```bash
PYTHONPATH=src python scripts/collect_hle_data.py --dry-run
```

Collect roulette data:

```bash
PYTHONPATH=src python scripts/collect_hle_data.py \
  --mode roulette \
  --split train \
  --runs 1 \
  --workers 4 \
  --output-dir trajectories/hle
```

Evaluate a learned checkpoint:

```bash
PYTHONPATH=src python scripts/collect_hle_data.py \
  --mode learned \
  --checkpoint outputs/q_function/q_function_best.pt \
  --split test_id \
  --runs 1 \
  --workers 4 \
  --use-batching \
  --output-dir trajectories/hle_eval
```

## Baseline Routers

The baseline routers train separate score/cost estimates and combine them during inference with `score - lambda * cost`.
This is intentionally different from the paper's neural MTRouter target and is mainly for controlled comparisons.

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

Evaluate a baseline router:

```bash
PYTHONPATH=src python scripts/collect_data.py \
  --mode per_model_ridge \
  --model-dir outputs/per_model_ridge \
  --lambda 1.0 \
  --split test_id \
  --first-try \
  --output-dir trajectories/eval
```

Replace `per_model_ridge` with `per_model_xgboost` or `cluster` as needed.

## Compare Results

Evaluation collectors write `summary_<split>.json` files. Compare methods with:

```bash
PYTHONPATH=src python scripts/compare_results.py \
  --test-dir trajectories/eval \
  --output-dir results/comparison \
  --export-csv results/comparison/results.csv
```

The comparison script reports average score, cost per run, efficiency, step counts, completion rate, per-task scores, and model-use distribution for routers.

## Notes on Data and Compatibility

- `TrajectoryDataset` exposes `score_targets` and `cost_targets`; neural training builds a single scalar target from these fields.
- `target_value` is still emitted by the parser for older consumers, but it is not the primary neural training interface.
- Historical dual-head checkpoints are rejected by `NeuralQFunction.load()`; retrain with the restored single-head-compatible scripts.
- `xgboost` is an optional research dependency. `train_per_model_xgboost.py --help` works without it, but actual XGBoost training requires installing the research extras.
