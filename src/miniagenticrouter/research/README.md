# Research 模块：多轮 Agent 模型路由研究工具

本模块面向“多轮 Agent 任务中动态选择调用模型（LLM routing）”的研究/实验，提供可复现的端到端流程：

- 数据划分：任务级（train vs OOD）+ variation 级（train/val/test_id）
- 轨迹采集：支持 baseline / roulette / learned / β-mixed / ridge / cluster 等策略
- 轨迹解析：从 `*.traj.json` 提取 per-step 特征与训练样本
- 训练与推理：HistoryEncoder + 多种 QFunction（神经网络/线性/聚类/启发式）
- 评估与对比：统一生成 `summary_{split}.json`，并用脚本聚合/可视化

> 默认实验环境耦合 ScienceWorld（采集侧）。路由器/训练侧大多与环境无关，可迁移。

---

## 快速索引（常用入口）

- 采集：`scripts/collect_data.py`（推荐看 `--help`）
- 预计算 embedding：`scripts/precompute_embeddings.py`
- 训练（神经 Q）：`scripts/train_q_function.py` / `scripts/train_stage_b.py`
- 训练（线性/聚类基线）：`scripts/train_per_model_ridge.py` / `scripts/train_cluster_baseline.py`
- 评估对比：`scripts/compare_results.py`

配置文件：
- 数据划分：`src/miniagenticrouter/config/research/data_split.yaml`
- 训练配置：`src/miniagenticrouter/config/research/training.yaml`
- 模型成本/上下文等：`src/miniagenticrouter/config/models/custom_models.yaml`

---

## 模块结构

```
src/miniagenticrouter/research/
├── data/                      # 数据划分（DataSplit）
├── trajectory/                # 轨迹解析（TrajectoryParser）
├── collection/                # 采集（Collector + CollectionMode）
├── routers/                   # LearnedRouter + QFunction + Policy
├── training/                  # NeuralQFunction (HistoryEncoder/ModelEncoder/QNetwork)
└── utils/                     # config loader
```

---

## 安装与安全提示

安装研究依赖：

```bash
pip install -e ".[research]"
```

安全提示（强烈建议阅读）：

> 轨迹文件（`*.traj.json`）会保存模型配置与响应元数据。如果 `model_kwargs` 中包含 API key/endpoint 等敏感字段，可能被写入轨迹。建议：
> - 用环境变量注入 key（不要写进 YAML）
> - 不要提交 `trajectories/` 到公开仓库
> - 共享前做脱敏检查

另外，`scripts/collect_data.py` 会优先加载仓库根目录的 `.env`（若存在）。

---

## 研究建模（核心目标）

将多轮 Agent 的“选择哪个模型”抽象成 action：

- 历史（可观测）`h_t`：messages / 对话上下文
- 动作 `a_t ∈ {1..m}`：选择哪个 LLM
- 代价 `c_t`：该步 token/费用成本
- 终止回报 `S_final`：任务最终得分（如 ScienceWorld score）

常用优化形式（拉格朗日）：

```
J_λ(π) = E[ S_final - λ * Σ_t c_t ]
```

在实现中通常学习两个量并组合：

- `score_hat(x, a)`：预测最终得分
- `cost_hat(x, a)`：预测剩余成本（或归一化成本）
- `Q_λ(x, a) = score_hat - λ * cost_hat`

---

## 路由算法一览（实现方式 + 参数）

### 1) 采集/评估模式（CollectionMode）

这些模式主要用于“跑环境 + 记录轨迹”，实现集中在 `src/miniagenticrouter/research/collection/modes.py`：

| 模式 | 目的 | 实现 | 关键参数 |
|---|---|---|---|
| Baseline | 单模型上/下界 | `BaselineMode` | `model_name` |
| Roulette | 随机探索数据（可记录 propensity） | `RouletteMode` + `PropensityRouletteRouter` | `record_propensity` |
| Learned | 用训练好的 QFunction 贪心选模型 | `LearnedMode` + `LearnedRouter` | `checkpoint`、`use_batching`、`batch_size`、`timeout` |
| Mixed (β-mixed) | Stage-B 迭代采集（learned+roulette 混合，记录 propensity） | `MixedMode` + `PropensityMixedRouter` | `checkpoint`、`beta`、`use_batching` |
| Per-model Ridge | 线性基线：每模型独立回归器 | `PerModelRidgeMode` + `PerModelRidgeQFunction` | `model_dir`、`lambda_`、`use_batching` |
| Cluster | 聚类基线：cluster→统计表 lookup | `ClusterMode` + `ClusterQFunction` | `model_dir`、`lambda_`、`use_batching` |

propensity 记录：
- uniform roulette：`propensity = 1 / n`（`src/miniagenticrouter/research/collection/propensity_router.py`）
- β-mixed：`propensity(a)=β*I[a=greedy]+(1-β)/n`（`src/miniagenticrouter/research/collection/propensity_mixed_router.py`）

### 2) LearnedRouter（统一路由器壳）

`src/miniagenticrouter/research/routers/learned.py` 提供：

- `QFunction` 抽象接口：`predict(history, available_models) -> np.ndarray`
- `LearnedRouter`：调用 QFunction 产出 Q 值，再用 policy 选模型

`LearnedRouter` 关键参数（`LearnedRouterConfig`）：
- `model_kwargs`: 模型池（list[dict]，内部会 `get_model(config)`）
- `q_function_path`: 可选（目前默认不自动加载，通常直接传 `q_function` 实例）
- `policy`: `"greedy" | "epsilon_greedy" | "softmax" | "ucb" | "thompson"`
- `policy_kwargs`: 各策略参数

### 3) SelectionPolicy（选择策略）

实现：`src/miniagenticrouter/research/routers/policies.py`

| policy | 作用 | 参数（policy_kwargs） |
|---|---|---|
| greedy | 直接选 argmax(Q) | 无 |
| epsilon_greedy | 探索/利用 | `epsilon`、`decay`、`min_epsilon` |
| softmax | 按 softmax(Q/τ) 采样 | `temperature` |
| ucb | bandit 上置信界 | `c` |
| thompson | beta posterior 采样（适合二元/归一化回报） | `alpha_prior`、`beta_prior` |

> 评估/部署建议用 `greedy`；采集/在线学习可用带探索的策略。

---

## QFunction 实现（路由算法核心）

QFunction 的选择决定了“如何从 history 估计每个模型的价值”。主要实现都在 `src/miniagenticrouter/research/routers/` 与 `src/miniagenticrouter/research/training/`。

### A) NeuralQFunction（神经网络 Q）

实现：`src/miniagenticrouter/research/training/q_network.py`

结构：
- `HistoryEncoder`：messages → embedding（可用 HF 或 vLLM backend）
- `ModelEncoder`：model_idx → embedding（基于模型属性/残差 embedding）
- `QNetwork`：concat(state, model) → 标量 Q

推理加速（并发多线程）：
- `BatchedQFunction`：`src/miniagenticrouter/research/training/batched_inference.py`

训练入口：
- `scripts/train_q_function.py`：从 `trajectories/*` 训练出 `outputs/q_function/...pt`
- `scripts/train_stage_b.py`：支持 warm-start + 多数据源联合训练（Stage-B）

### B) PerModelRidgeQFunction（每模型独立 Ridge）

实现：`src/miniagenticrouter/research/routers/per_model_ridge_q.py`

思想：每个模型都有自己的线性回归器（只吃 history embedding）：

```
score_hat_a(z) = w_a · z + b_a
cost_hat_a(z)  = v_a · z + c_a
Q_λ = score_hat - λ * cost_hat
```

优点：模型排序可以随上下文变化（比“共享权重 + 模型特征”的线性形式更灵活）。

训练入口：
- `scripts/train_per_model_ridge.py` → `outputs/per_model_ridge/per_model_ridge.pkl`

推理加速：
- `BatchedPerModelRidgeQFunction`（同文件内）

### C) ClusterQFunction（KMeans 聚类 lookup）

实现：`src/miniagenticrouter/research/routers/cluster_q.py`

思想：在 history embedding 空间做 KMeans，把状态划分成 K 个 region；每个 region 维护 per-model 的 `(score_mean, cost_mean)` 统计表：

```
cluster_id = kmeans(z)
Q_λ = score_mean(cluster_id, a) - λ * cost_mean(cluster_id, a)
```

训练入口：
- `scripts/train_cluster_baseline.py` → `outputs/cluster_baseline/{cluster_kmeans.pkl, cluster_stats.json}`

推理加速：
- `BatchedClusterQFunction`（同文件内）

### D) RidgeQFunction（共享 Ridge：history + model features）

实现：`src/miniagenticrouter/research/routers/ridge_q.py`

说明：
- 这是一个“共享线性”基线：拼接 `[history_emb, model_feat]`，训练两个 Ridge（score/cost），推理时组合 `Q_λ`。
- 当前实现里 `MODEL_NAMES` 与 `RAW_MODEL_FEATURES` 是硬编码常量，默认只适配脚本里那 3 个模型。
- 该基线目前没有在 `scripts/collect_data.py` 中作为 mode 暴露（你可以手动组装成 `LearnedRouter(q_function=RidgeQFunction(...))` 使用）。

权重格式：
- 期望目录包含：`ridge_score.pkl` 与 `ridge_cost.pkl`（可选：`metrics.json` 用于一致性校验提示）

备注：
- 本仓库当前未提供 `RidgeQFunction` 的训练脚本（可参考 `PerModelRidgeQFunction` 的训练脚本自行实现，或直接使用已有权重目录）。

推理加速：
- `BatchedRidgeQFunction`（同文件内）

### E) 启发式基线（无需训练）

实现：`src/miniagenticrouter/research/routers/learned.py`

- `RandomQFunction`：随机（等价随机选）
- `ConstantQFunction`：永远偏好某个模型
- `CostAwareQFunction`：按成本惩罚（仅用于 sanity check/基线）

---

## 配置文件（参数如何控制）

### 1) 数据划分：`data_split.yaml`

文件：`src/miniagenticrouter/config/research/data_split.yaml`

关键字段：
- `split_seed`：variation 划分随机种子（固定后不要改）
- `train_tasks` / `ood_test_tasks`：任务级划分
- `first_try`：快速验证子集（9 个任务）
- `variation_split.{train_ratio,val_ratio,test_id_ratio}`：同分布切分
- `models.roulette_model_pool`：roulette/baseline模式使用的模型池
- `collection.*`：采集默认 runs/是否记录 propensity/step&cost 限制
- `metrics`：评估关注的指标（summary 中会体现部分）

> 注意：roulette/baseline模式使用 `data_split.yaml` 的 `roulette_model_pool`；learned/mixed/ridge/cluster等模式使用 `training.yaml` 的 `model_pool`。

### 2) 训练配置：`training.yaml`

文件：`src/miniagenticrouter/config/research/training.yaml`

重点在 `history_encoder` 与 `precompute`：
- `history_encoder.backend`: `"hf"` 或 `"vllm"`
- `history_encoder.model_name`：本地路径或 HF 名称（用于 tokenizer/或本地加载）
- `history_encoder.vllm_base_url` / `vllm_model_id`：当 backend 为 vllm 时的服务端参数
- `history_encoder.max_tokens` / `min_recent_turns` / `pooling_mode`：分词截断策略与 pooling
- `precompute.*`：预计算 embedding 的后端与缓存策略（建议与在线推理一致）

> 训练/推理一致性：如果你希望“训练时的 embedding”与“在线路由时的 embedding”一致，应保证 backend/model_id/max_tokens/pooling_mode 等一致（必要时可把训练配置快照随 checkpoint 保存）。

### 3) 模型属性与成本：`custom_models.yaml`

文件：`src/miniagenticrouter/config/models/custom_models.yaml`

主要用于：
- 成本估算（input/output cost）
- 上下文/输出长度上限
- provider 与连接参数（可通过环境变量覆盖）

---

## 端到端用法（采集→训练→评估→对比）

下面给出“最小可跑”流程。实际跑实验建议参考 `scripts/normal/*.sh`，它们会设置 NO_PROXY、激活 `.venv` 等。

### Step 0：确认模型池与顺序

多个训练/推理脚本内部写死了 `MODEL_NAMES`（默认 3 个）：

- `openai/gpt-5`
- `deepseek/deepseek-v3.2`
- `minimax/minimax-m2`

建议采集和训练都严格使用这套模型池/顺序（否则会出现“样本被跳过/模型名对不上”的问题）。如果你要换模型池，需要同时修改：
- 相关脚本里的 `MODEL_NAMES`
- Ridge 基线里的 `RAW_MODEL_FEATURES`（如果继续使用）
- 以及 `data_split.yaml` 中用于采集的 `models.*` 列表

### Step 1：采集数据（baseline + roulette）

1) roulette（推荐记录 propensity，用于重要性加权/离线评估）：

```bash
python scripts/collect_data.py --mode roulette --first-try --split train --output-dir trajectories --workers 16
```

2) baseline（每个模型跑一套，用于上下界与补充数据）：

```bash
python scripts/collect_data.py --mode baseline --model deepseek/deepseek-v3.2 --first-try --split train --output-dir trajectories --workers 16
python scripts/collect_data.py --mode baseline --model minimax/minimax-m2 --first-try --split train --output-dir trajectories --workers 16
python scripts/collect_data.py --mode baseline --model openai/gpt-5 --first-try --split train --output-dir trajectories --workers 16
```

采集输出：
- 轨迹：`trajectories/<mode>/<task_id>_runK/<task_id>_runK.traj.json`
- 汇总：`trajectories/<mode>/results.json` + `trajectories/<mode>/summary_<split>.json`

### Step 2：预计算 history embedding（可选但强烈推荐）

```bash
python scripts/precompute_embeddings.py \
  --data-dir trajectories/roulette_propensity \
  --output embeddings/roulette_propensity.pt
```

默认会读取 `training.yaml` 的 `history_encoder`/`precompute` 配置作为后端参数（HF 或 vLLM；vLLM 支持 SQLite cache）。

### Step 3：训练路由器（不同算法）

神经网络 Q-function：

```bash
python scripts/train_q_function.py \
  --data-dir trajectories/roulette_propensity \
  --output-dir outputs/q_function \
  --epochs 50
```

Stage-B（用 mixed 数据迭代）：

```bash
python scripts/train_stage_b.py \
  --data-dir trajectories/roulette_propensity \
  --data-dir trajectories/mixed_b1 \
  --init-checkpoint outputs/q_function/q_function_best.pt \
  --output-dir outputs/q_function_v1
```

Per-model Ridge：

```bash
python scripts/train_per_model_ridge.py \
  --precomputed-path embeddings/roulette_propensity.pt \
  --data-dir trajectories/roulette_propensity \
  --output-dir outputs/per_model_ridge
```

Cluster baseline：

```bash
python scripts/train_cluster_baseline.py \
  --precomputed-path embeddings/roulette_propensity.pt \
  --data-dir trajectories/roulette_propensity \
  --output-dir outputs/cluster_baseline
```

> 重要性加权：`train_per_model_ridge.py` 与 `train_cluster_baseline.py` 支持 `--use-propensity`，会用 inverse propensity 做 sample weight（需轨迹中记录 propensity）。

### Step 4：评估（生成 test_id / ood_test summary）

所有评估统一通过 `scripts/collect_data.py --split {val|test_id|ood_test} --output-dir trajectories/test --runs 1` 生成汇总文件，目录名即 method name。

Learned（神经 Q）：

```bash
python scripts/collect_data.py \
  --mode learned \
  --checkpoint outputs/q_function/q_function_best.pt \
  --first-try \
  --split test_id \
  --output-dir trajectories/test \
  --runs 1 \
  --workers 16 \
  --use-batching --batch-size 16
```

β-mixed（用于采集 Stage-B 数据/也可评估）：

```bash
python scripts/collect_data.py \
  --mode mixed \
  --checkpoint outputs/q_function/q_function_best.pt \
  --beta 0.5 \
  --first-try \
  --split test_id \
  --output-dir trajectories/test \
  --runs 1 \
  --workers 16 \
  --use-batching --batch-size 16
```

Per-model Ridge：

```bash
python scripts/collect_data.py \
  --mode per_model_ridge \
  --model-dir outputs/per_model_ridge \
  --lambda 0.85 \
  --first-try \
  --split test_id \
  --output-dir trajectories/test \
  --runs 1 \
  --workers 32 \
  --use-batching --batch-size 32
```

Cluster：

```bash
python scripts/collect_data.py \
  --mode cluster \
  --model-dir outputs/cluster_baseline \
  --lambda 1.0 \
  --first-try \
  --split test_id \
  --output-dir trajectories/test \
  --runs 1 \
  --workers 16 \
  --use-batching --batch-size 32
```

Baseline（对照组）：

```bash
python scripts/collect_data.py --mode baseline --model deepseek/deepseek-v3.2 --first-try --split test_id --output-dir trajectories/test --runs 1 --workers 16
```

### Step 5：对比与可视化

```bash
python scripts/compare_results.py --test-dir trajectories/test --output-dir results/comparison
```

该脚本默认读取每个 method 目录下的 `summary_test_id.json`，并输出：
- overall 对比（score/cost/steps/完成率等）
- 按任务对比
- 路由器的模型使用分布（如果是多模型方案）
- 成本-性能前沿（可选 matplotlib）
- 简单 CI（基于 per-run 统计）

---

## 评估方案建议（如何做一套可复现实验）

1) 固定划分：确认 `data_split.yaml` 后不要再修改（尤其 `split_seed`、任务列表）。

2) 超参选择：
- `λ`（成本惩罚）建议在 `val` 上选，然后报告 `test_id` 与 `ood_test`。
- `β`（mixed 比例）用于 Stage-B 采样/稳健性权衡：β 越大越偏向 learned，越小越偏向探索。

3) 多 runs：
- `Collector` 支持对每个 variation 跑多次（`runs_per_variation`）。
- `compare_results.py` 会提示 runs 数不足时的建议与粗略 CI。

4) 指标：
- score：`summary_*.json` 的 `overall.avg_score`
- cost：`overall.total_cost` 与 `model_usage` 的近似分摊
- steps：`overall.avg_steps` / `overall.avg_steps_success`
- completion：按 `exit_status` 统计（Submitted、LimitsExceeded 等）

---

## 输出文件与格式

### 1) 结果目录结构

以 `--output-dir trajectories/test --mode learned` 为例（目录名由 mode 自动生成；启用 `--use-batching` 时通常会附加 `_batched` 后缀）：

```
trajectories/test/
└── learned_q_function_best/
    ├── results.json
    ├── summary_test_id.json
    └── <task_id>_run0/
        └── <task_id>_run0.traj.json
```

### 2) 轨迹里与路由相关的字段

采集时，路由器会把信息写进每一步的 message `extra.response`（不同 provider 的字段略有差异）。research 侧主要依赖：
- 每步选中的 `model_name`
- 每步 cost（从 usage/cost 字段累计）
- （可选）`propensity` / `router_info`（roulette/mixed）

解析入口：`miniagenticrouter.research.trajectory.parser.TrajectoryParser`

---

## 常见坑点（强相关于本仓库的默认实现）

1) BaselineMode 参数更新  
代码里 `BaselineMode` 使用 `model_name`（不是 `model_index`）。如果你在旧文档/旧脚本里看到 `model_index`，以 `src/miniagenticrouter/research/collection/modes.py` 为准。

2) 模型池一致性（最重要）  
神经 Q / ridge / cluster 等实现通常假设“训练时的 model_names 列表”与“运行时模型池及顺序”一致：
- `scripts/train_q_function.py` / `scripts/train_stage_b.py` / `scripts/test_inference.py` 写死 `MODEL_NAMES`
- `MixedMode/LearnedMode` 会校验 checkpoint 里的 model_names 是否匹配
- per-model ridge/cluster/ridge baseline 也默认围绕 3 模型设计

3) 训练/推理一致性（embedding 后端）  
`HistoryEncoder` 支持 `hf` 与 `vllm` 两种 backend。要做严谨实验，建议：
- 预计算 embedding 的 backend 与在线路由时一致（都用 hf 或都用 vllm）
- 固定 `max_tokens/pooling_mode` 等配置（训练/推理同一套）

4) 动态 batching  
`--use-batching` 会启用后台 worker 收集 request 批处理（提高 GPU/vLLM 利用率）。脚本进程结束会退出；如果你在长驻进程里使用 mode/router，建议显式调用对应 `shutdown()`。

---

## API 参考（最小）

```python
from miniagenticrouter.research import DataSplit, Collector, CollectionConfig
from miniagenticrouter.research import RouletteMode, BaselineMode, LearnedMode

split = DataSplit.from_yaml(use_first_try=True)
config = CollectionConfig(output_dir="trajectories", runs_per_variation=1, workers=8)

collector = Collector(data_split=split, mode=RouletteMode(record_propensity=True), config=config)
collector.collect(split="train")
```

路由器（LearnedRouter）可直接用于 `FlexibleAgent` 的 `model=` 参数（其基类是 `BaseRouter`）。

---

## 扩展到其他 Benchmark（提示）

采集侧目前强耦合 ScienceWorld（主要在 `Collector.collect_single()` 里创建环境与 agent）。迁移到其他 benchmark 时一般需要：
- `data/split.py`：提供任务与 variation 枚举方式
- `collection/collector.py`：替换 environment + agent 构造逻辑
- `src/miniagenticrouter/config/research/data_split.yaml`：替换任务列表与模型池

路由器/训练侧（`routers/*`、`training/*`）大多是通用的，只依赖“messages → embedding”与“最终得分/成本标签”的定义。
