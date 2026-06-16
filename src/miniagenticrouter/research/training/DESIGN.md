# Q-Function Training Design

This document describes the current training implementation for MTRouter's learned router.
It supersedes the early prototype notes that used one-hot model attributes or a dual-head score/cost network.

## Objective

At each turn, the router observes a conversation history `h_t` and chooses one model `a_t` from the configured model pool.
The neural Q-function predicts one scalar value per candidate model:

```text
s_hat_theta(h_t, a)
```

The target is the normalized terminal outcome of the trajectory, optionally adjusted by AEC (Annealed Error Cost).
The paper/default target does not add a separate explicit cost penalty; the default `lambda_` is therefore `0.0`.
The code still accepts non-zero `lambda_` values for legacy ablations and for baseline routers that combine score and cost estimates.

## Components

### HistoryEncoder

Implementation: `src/miniagenticrouter/research/training/encoders.py`

The history encoder converts task context plus recent interaction history into a single fixed-dimensional embedding.
The current paper-aligned configuration uses:

- Frozen embedding model: `Qwen/Qwen3-Embedding-0.6B`
- Output dimension: 1024
- Maximum token budget: 8192 tokens
- Pooling: last-token pooling
- Backends: local HuggingFace (`hf`) or OpenAI-compatible vLLM (`vllm`)

The segmentation policy keeps the task block first, then guarantees a minimum number of recent turns, then fills remaining budget with additional recent context.

### ModelEncoder

Implementation: `src/miniagenticrouter/research/training/model_encoder.py`

The model encoder maps each routed LLM to a learned representation using metadata plus a small residual embedding.
The paper-aligned architecture is:

```text
8 metadata features -> MLP -> 32-d attribute embedding
per-model residual embedding -> 16-d residual
concat(attribute, residual) -> projection + LayerNorm -> 64-d model embedding
```

Metadata comes from `src/miniagenticrouter/config/models/custom_models.yaml` and includes context limits, output limits, pricing, and knowledge cutoff date features.
The residual embedding is initialized to zero and regularized with L2 weight `0.001`.

### QNetwork

Implementation: `src/miniagenticrouter/research/training/q_network.py`

The Q-network is a single-head MLP:

```text
concat(history_embedding, model_embedding) -> hidden MLP -> scalar value
```

The current `NeuralQFunction.forward()` and `forward_precomputed()` methods return one tensor of Q/value predictions.
Historical dual-head checkpoints with `score_head` and `cost_head` are intentionally rejected by the loader.

## Training Data

`TrajectoryParser` reads `*.traj.json` files and emits per-turn samples.
Each sample includes:

- `messages`: the visible history at the turn
- `model_name` and `model_idx`: the chosen action
- `final_score`: normalized benchmark score, optionally AEC-adjusted
- `remaining_cost`: raw remaining cost from the turn to episode end
- `normalized_cost`: filled by `TrajectoryDataset` after per-benchmark normalization
- `score_targets` and `cost_targets`: tensors used by current training scripts
- `target_value`: legacy scalar target retained for backward compatibility
- `propensity`: behavior-policy probability when roulette or mixed collection records it

Benchmark normalization is handled in `src/miniagenticrouter/research/trajectory/parser.py` and `src/miniagenticrouter/research/training/dataset.py`.
ScienceWorld scores are normalized from `[-100, 100]` to `[0, 1]`; HLE judge scores are already binary `[0, 1]`.

## AEC: Annealed Error Cost

AEC optionally reduces the terminal outcome target when later trajectory steps contain parser, action, tool, or formatting errors.
The penalty is time-dependent: later errors receive larger weight than early exploratory errors.

The implementation stores:

- `error_detected`
- `error_types`
- `step_aec_penalty`
- `aec_penalty` for the cumulative penalty from the current step to episode end
- `original_final_score`

Training scripts enable this path with `--aec-enabled` and use rules from the error-rule configuration loaded by `ErrorDetector`.

## Training Scripts

### Neural MTRouter

Use `scripts/train_q_function.py` for the main neural training run:

```bash
PYTHONPATH=src python scripts/train_q_function.py \
  --data-dir trajectories/scienceworld/roulette_propensity \
  --precomputed-path embeddings/scienceworld_roulette_propensity.pt \
  --output-dir outputs/q_function \
  --epochs 100 \
  --batch-size 64 \
  --lambda_ 0.0
```

The script supports:

- multiple `--data-dir` values
- online or precomputed embeddings
- AEC targets
- optional AWR and margin-loss ablations
- checkpoint rotation and early stopping
- TensorBoard and JSONL logs

### Baseline Trainers

The baseline trainers are useful for comparisons but do not implement the paper's neural single-head objective.
They fit separate score and cost estimates and combine them at inference time with `score - lambda * cost`.

- `scripts/train_per_model_ridge.py`
- `scripts/train_per_model_xgboost.py`
- `scripts/train_cluster_baseline.py`

## Embedding Precomputation

Precompute embeddings before training when possible:

```bash
PYTHONPATH=src python scripts/precompute_embeddings.py \
  --data-dir trajectories/scienceworld/roulette_propensity \
  --output embeddings/scienceworld_roulette_propensity.pt
```

The precomputed file stores embeddings and encoder configuration metadata.
Training scripts validate model ordering and encoder configuration where possible to avoid silent train/inference mismatches.

## Checkpoint Format

A neural checkpoint stores:

- trainable state dict for `model_encoder` and `q_network`
- `model_names` in training order
- nested encoder/network configs
- `lambda_` metadata for compatibility

The frozen history-encoder weights are not saved to keep checkpoints small.
At load time, model names must match the runtime model pool exactly, including order.

## Current Defaults

Defined mainly in `src/miniagenticrouter/config/research/training.yaml`:

```text
model_pool: GPT-5, DeepSeek-V3.2, MiniMax-M2, Kimi-K2, Gemini-2.5-Flash-Lite, GPT-OSS-120B
history_encoder.model_name: Qwen/Qwen3-Embedding-0.6B
history_encoder.max_tokens: 8192
training.epochs: 100
training.batch_size: 64
training.learning_rate: 1e-3
training.lambda_: 0.0
training.optimizer: adamw
training.weight_decay: 0.01
training.scheduler: cosine
training.early_stop: 3
model_encoder.residual_reg_weight: 0.001
q_network.hidden_dims: [256, 256]
q_network.dropout: 0.1
```

## Known Compatibility Notes

- The restored training scripts are updated for the current single-head API.
- Older dual-head checkpoints must be retrained.
- `target_value` remains in parsed samples for older utilities, but new training code uses `score_targets` and `cost_targets`.
- XGBoost training requires the optional `xgboost` dependency from `pip install -e '.[research]'`.
