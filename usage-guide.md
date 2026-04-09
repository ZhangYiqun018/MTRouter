# Mini Agentic Router 使用指南

本文档介绍如何使用 mini-agentic-router 来解决 SWE-bench 中的题目，以及如何配置模型和路由模式。

> **Note:** 本项目基于 [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) 二次开发，提供了增强的模型路由功能。

---

## 目录

1. [安装](#安装)
2. [运行 SWE-bench 题目](#运行-swe-bench-题目)
3. [模型配置](#模型配置)
4. [路由模式（Roulette/Interleaving）](#路由模式)
5. [评估结果](#评估结果)
6. [常见问题](#常见问题)

---

## 安装

```bash
# 从源码安装（开发模式）
git clone https://github.com/anonymous/mini-agentic-router.git
cd mini-agentic-router && pip install -e .
```

---

## 运行 SWE-bench 题目

### 方式一：批量运行（Batch Mode）

```bash
# 运行所有题目
mar-extra swebench \
    --model anthropic/claude-sonnet-4-5-20250929 \
    --subset verified \
    --split test \
    --workers 4

# 运行前5道题（使用 --slice）
mar-extra swebench \
    --model anthropic/claude-sonnet-4-5-20250929 \
    --subset verified \
    --split test \
    --slice 0:5 \
    --workers 4

# 通过正则表达式过滤特定题目（使用 --filter）
mar-extra swebench \
    --model anthropic/claude-sonnet-4-5-20250929 \
    --subset verified \
    --split test \
    --filter "sympy__sympy.*" \
    --workers 4
```

### 方式二：单题调试模式（Single Instance）

适合调试单个题目：

```bash
# 通过 instance ID 指定题目
mar-extra swebench-single \
    --subset verified \
    --split test \
    --model anthropic/claude-sonnet-4-5-20250929 \
    -i sympy__sympy-15599

# 或通过索引指定（第一道题）
mar-extra swebench-single \
    --subset verified \
    --split test \
    --model anthropic/claude-sonnet-4-5-20250929 \
    -i 0

# 运行后立即退出（不提示确认）
mar-extra swebench-single \
    --subset verified \
    --split test \
    --model anthropic/claude-sonnet-4-5-20250929 \
    -i 0 \
    --exit-immediately
```

### 主要参数说明

| 参数 | 说明 |
|------|------|
| `-m`, `--model` | 模型名称，如 `anthropic/claude-sonnet-4-5-20250929` |
| `-o`, `--output` | 输出目录 |
| `-c`, `--config` | 配置文件路径（默认使用 `swebench.yaml`）|
| `-w`, `--workers` | 并行工作线程数 |
| `--subset` | 数据集子集：`lite`、`verified` 或自定义路径 |
| `--split` | 数据集分割：`dev` 或 `test` |
| `--slice` | 切片，如 `0:5` 表示前5道题 |
| `--filter` | 正则表达式过滤 instance ID |
| `--shuffle` | 打乱顺序 |
| `--redo-existing` | 重新运行已存在的实例 |
| `--environment-class` | 环境类型：`docker` 或 `singularity` |

---

## 模型配置

### 1. 设置 API Key

有多种方式设置 API Key：

```bash
# 方式一（推荐）：交互式设置
mar-extra config setup

# 方式二：命令行设置
mar-extra config set ANTHROPIC_API_KEY "your-api-key"
mar-extra config set OPENAI_API_KEY "your-api-key"

# 方式三：环境变量
export ANTHROPIC_API_KEY="your-api-key"
export OPENAI_API_KEY="your-api-key"

# 如果只用单一模型，可以设置通用 key
mar-extra config set MAR_MODEL_API_KEY "your-api-key"
```

### 2. 设置默认模型

```bash
# 命令行设置
mar-extra config set MAR_MODEL_NAME "anthropic/claude-sonnet-4-5-20250929"

# 或使用环境变量
export MAR_MODEL_NAME="anthropic/claude-sonnet-4-5-20250929"
```

### 3. 在 YAML 配置文件中配置模型

创建或编辑配置文件（如 `my_config.yaml`）：

```yaml
# 基本配置
model:
  model_name: "anthropic/claude-sonnet-4-5-20250929"
  model_kwargs:
    temperature: 0.0

# OpenAI 模型
model:
  model_name: "openai/gpt-5"
  model_kwargs:
    drop_params: true
    reasoning_effort: "high"

# Deepseek 模型
model:
  model_name: "deepseek/deepseek-chat"
  model_kwargs:
    temperature: 0.0
```

### 4. 配置本地模型（自定义 API Base）

```yaml
model:
  model_name: "my-local-model"
  model_kwargs:
    custom_llm_provider: "openai"
    api_base: "http://localhost:8000/v1"
    # 其他参数...
  cost_tracking: "ignore_errors"  # 本地模型可忽略成本追踪
```

### 5. 使用 vLLM 本地模型示例

```bash
# 1. 启动 vLLM 服务器
vllm serve ricdomolm/mini-coder-1.7b &

# 2. 创建模型注册文件 registry.json
cat > registry.json <<'EOF'
{
  "ricdomolm/mini-coder-1.7b": {
    "max_tokens": 40960,
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
    "litellm_provider": "hosted_vllm",
    "mode": "chat"
  }
}
EOF

# 3. 配置 YAML 文件
# model:
#   model_name: "hosted_vllm/ricdomolm/mini-coder-1.7b"
#   model_kwargs:
#     api_base: "http://localhost:8000/v1"

# 4. 运行
LITELLM_MODEL_REGISTRY_PATH=registry.json mar-extra swebench \
    --output test/ --subset verified --split test --filter '^(django__django-11099)$'
```

### 6. 使用 OpenRouter

```yaml
model:
  model_name: "moonshotai/kimi-k2-0905"
  model_class: "openrouter"
  model_kwargs:
    temperature: 0.0
    provider:
      allow_fallbacks: false
      only: ["Moonshot AI"]
```

需要设置 `OPENROUTER_API_KEY`。

### 7. 使用 Portkey

```yaml
model:
  model_name: "@openai/gpt-5-mini"
  model_class: "portkey"
  model_kwargs:
    reasoning_effort: "medium"
```

需要设置 `PORTKEY_API_KEY`。

---

## 路由模式

Mini-agentic-router 支持两种路由器来组合多个模型：

### 1. RouletteRouter（随机选择模式）

每次调用时**随机选择**一个模型。

```yaml
model:
  model_name: "roulette"
  model_class: "miniagenticrouter.routers.roulette.RouletteRouter"
  model_kwargs:
    - model_name: "anthropic/claude-sonnet-4-5-20250929"
      model_kwargs:
        temperature: 0.0
    - model_name: "openai/gpt-5"
      model_kwargs:
        temperature: 0.0
```

### 2. InterleavingRouter（交替选择模式）

按照指定顺序**交替选择**模型。

```yaml
model:
  model_name: "interleaving"
  model_class: "miniagenticrouter.routers.interleaving.InterleavingRouter"
  model_kwargs:
    - model_name: "anthropic/claude-sonnet-4-5-20250929"
      model_kwargs:
        temperature: 0.0
    - model_name: "openai/gpt-5"
      model_kwargs:
        temperature: 0.0
  # 可选：自定义交替序列
  # sequence: [0, 0, 1]  # 前两次用第一个模型，第三次用第二个模型，然后循环
```

### 使用预置的 Roulette 配置运行 SWE-bench

```bash
mar-extra swebench \
    --config swebench_roulette \
    --subset verified \
    --split test \
    --workers 4
```

### 路由模式工作原理

- **RouletteRouter**: 使用 `random.choice()` 随机选择模型
- **InterleavingRouter**:
  - 默认按顺序轮流选择：`n_calls % len(models)`
  - 可通过 `sequence` 参数自定义顺序，如 `[0, 0, 1]` 表示"模型0, 模型0, 模型1"循环

---

## 评估结果

运行完成后会生成 `preds.json` 文件，可以用以下方式评估：

### 云端评估（推荐）

使用 [sb-cli](https://www.swebench.com/sb-cli/)，免费且快速（通常20分钟内出结果）：

```bash
# 安装 sb-cli 并获取 token 后
sb-cli submit swe-bench_verified test \
    --predictions_path preds.json \
    --run_id my-run-id
```

### 本地评估

```bash
python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --predictions_path preds.json \
    --max_workers 4 \
    --run_id my-run-id
```

---

## 常见问题

### Q: 如何设置全局成本限制？

```bash
# 设置调用次数限制
mar-extra config set MAR_GLOBAL_CALL_LIMIT 100

# 设置美元成本限制
mar-extra config set MAR_GLOBAL_COST_LIMIT 10.00

# 或使用环境变量
export MAR_GLOBAL_CALL_LIMIT=100
export MAR_GLOBAL_COST_LIMIT=10.00
```

### Q: Docker 不可用怎么办？

使用 Singularity/Apptainer：

```bash
mar-extra swebench \
    --environment-class singularity \
    ...
```

### Q: 如何忽略成本追踪错误？

```yaml
# 在配置文件中
model:
  cost_tracking: "ignore_errors"
```

或：

```bash
export MAR_COST_TRACKING="ignore_errors"
```

### Q: 如何使用自定义数据集？

只要数据集遵循 SWE-bench 格式：

```bash
mar-extra swebench \
    --subset /path/to/your/dataset \
    ...
```

### Q: 某些任务卡在 "initializing task"？

可能在拉取 Docker 镜像，可以增加超时时间：

```yaml
environment:
  pull_timeout: 300  # 默认是 120 秒
```

---

## 支持的模型列表

常用模型：
- `anthropic/claude-sonnet-4-5-20250929`
- `anthropic/claude-haiku-4-5-20251001`
- `openai/gpt-5`
- `openai/gpt-5-mini`
- `gemini/gemini-2.5-pro`
- `deepseek/deepseek-chat`
- `deepseek/deepseek-reasoner`

更多模型请参考 [litellm 支持的模型列表](https://docs.litellm.ai/docs/providers)。

---

## 参考链接

- [mini-swe-agent 官方文档](https://mini-swe-agent.com/latest/) - 原项目文档
- [mini-swe-agent GitHub](https://github.com/SWE-agent/mini-swe-agent) - 原项目仓库
- [SWE-bench 官网](https://www.swebench.com/)
- [litellm 文档](https://docs.litellm.ai/)
