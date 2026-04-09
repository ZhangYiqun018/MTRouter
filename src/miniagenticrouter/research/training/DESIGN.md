# Q-Function 训练设计文档

本文档记录 Q 函数训练模块的设计讨论和实现方案。

## 1. 整体架构

Q 函数估计 Q(x, a) 的值，其中：
- **x (state)**: 当前对话历史/状态
- **a (action)**: 模型选择

```
                    ┌─────────────────┐
                    │   Q(x, a)       │
                    │   Q-Network     │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
    ┌─────────▼─────────┐         ┌─────────▼─────────┐
    │   State Encoder   │         │   Model Encoder   │
    │   (history → h)   │         │   (model → m)     │
    └─────────┬─────────┘         └─────────┬─────────┘
              │                             │
    ┌─────────▼─────────┐         ┌─────────▼─────────┐
    │  Conversation     │         │  Model Attributes │
    │  History          │         │  (one-hot)        │
    └───────────────────┘         └───────────────────┘
```

## 2. State Encoder (状态编码)

### 2.1 输入

对话历史，可以是：
- 完整的 messages 列表 `[{role, content}, ...]`
- 拼接后的文本字符串

### 2.2 编码方案

**方案 A: Sentence Transformers**

```python
from sentence_transformers import SentenceTransformer

class StateEncoder(nn.Module):
    def __init__(self, model_name="all-MiniLM-L6-v2", output_dim=64):
        super().__init__()
        self.encoder = SentenceTransformer(model_name)
        self.projector = nn.Linear(384, output_dim)  # 384 是 MiniLM 输出维度

    def forward(self, history_text: str) -> Tensor:
        with torch.no_grad():
            embedding = self.encoder.encode(history_text, convert_to_tensor=True)
        return self.projector(embedding)
```

**方案 B: 手工特征**

提取统计特征（用于快速实验）：
- 对话轮数
- 总 token 数
- 最近一轮的长度
- 任务类型 one-hot（如果已知）

### 2.3 建议

初期使用 Sentence Transformers，简单且效果好。后续可以尝试更复杂的编码器。

---

## 3. Model Encoder (模型编码)

### 3.1 设计思路

不使用可学习的 Embedding Table，而是：
1. 定义模型的**静态属性**（人工标注）
2. 将属性离散化为 **one-hot 编码**
3. 通过 **MLP** 投影到嵌入空间

**优点**：
- 无需学习 embedding，属性直接编码
- 新模型只需填写属性配置即可使用
- 属性有明确语义，可解释性强

### 3.2 模型属性定义

```yaml
# config/research/model_attributes.yaml

models:
  claude-3-5-haiku-latest:
    price_tier: low        # low / medium / high
    capability: weak       # weak / medium / strong
    speed: fast            # fast / medium / slow
    context_length: medium # short / medium / long

  claude-sonnet-4-5-20250514:
    price_tier: high
    capability: strong
    speed: slow
    context_length: long

  gpt-4o-mini:
    price_tier: low
    capability: medium
    speed: fast
    context_length: medium

  gpt-4o:
    price_tier: medium
    capability: strong
    speed: medium
    context_length: long

# 属性值定义
attribute_values:
  price_tier: [low, medium, high]
  capability: [weak, medium, strong]
  speed: [fast, medium, slow]
  context_length: [short, medium, long]
```

### 3.3 One-Hot 编码

```python
# 属性 → one-hot 维度
# price_tier:     3 维  [low, medium, high]
# capability:     3 维  [weak, medium, strong]
# speed:          3 维  [fast, medium, slow]
# context_length: 3 维  [short, medium, long]
# ─────────────────────
# 总计:          12 维

# 示例编码
"claude-3-5-haiku" → [1,0,0, 1,0,0, 1,0,0, 0,1,0]
#                     price  capab  speed  context
#                     low    weak   fast   medium

"claude-sonnet-4-5" → [0,0,1, 0,0,1, 0,0,1, 0,0,1]
#                      high   strong slow   long
```

### 3.4 MLP 投影

```python
class ModelEncoder(nn.Module):
    """将模型 one-hot 属性编码为稠密向量"""

    def __init__(
        self,
        onehot_dim: int = 12,      # 属性 one-hot 总维度
        embed_dim: int = 32,       # 输出嵌入维度
        hidden_dim: int = 32,      # 隐藏层维度
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(onehot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, onehot: Tensor) -> Tensor:
        """
        Args:
            onehot: shape (batch, onehot_dim) 或 (onehot_dim,)
        Returns:
            embedding: shape (batch, embed_dim) 或 (embed_dim,)
        """
        return self.mlp(onehot)
```

### 3.5 属性加载工具

```python
class ModelAttributeLoader:
    """从 YAML 加载模型属性并转换为 one-hot"""

    def __init__(self, config_path: Path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.attribute_values = self.config["attribute_values"]
        self._build_encoder()

    def _build_encoder(self):
        """构建 one-hot 编码器"""
        self.attr_names = list(self.attribute_values.keys())
        self.onehot_dim = sum(len(v) for v in self.attribute_values.values())

    def encode(self, model_name: str) -> np.ndarray:
        """将模型名转换为 one-hot 向量"""
        attrs = self.config["models"][model_name]
        onehot = []

        for attr_name in self.attr_names:
            values = self.attribute_values[attr_name]
            value = attrs[attr_name]
            # one-hot for this attribute
            oh = [1.0 if v == value else 0.0 for v in values]
            onehot.extend(oh)

        return np.array(onehot, dtype=np.float32)

    def encode_batch(self, model_names: list[str]) -> np.ndarray:
        """批量编码"""
        return np.stack([self.encode(m) for m in model_names])
```

---

## 4. Q-Network 架构

### 4.1 基础架构

```python
class QNetwork(nn.Module):
    """Q(x, a) 网络"""

    def __init__(
        self,
        state_dim: int = 64,    # StateEncoder 输出维度
        model_dim: int = 32,    # ModelEncoder 输出维度
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.q_head = nn.Sequential(
            nn.Linear(state_dim + model_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出 Q 值
        )

    def forward(self, state_embed: Tensor, model_embed: Tensor) -> Tensor:
        """
        Args:
            state_embed: (batch, state_dim)
            model_embed: (batch, model_dim)
        Returns:
            q_value: (batch, 1)
        """
        combined = torch.cat([state_embed, model_embed], dim=-1)
        return self.q_head(combined)
```

### 4.2 完整 Q 函数实现

```python
class NeuralQFunction(QFunction):
    """基于神经网络的 Q 函数实现"""

    def __init__(
        self,
        state_encoder: StateEncoder,
        model_encoder: ModelEncoder,
        q_network: QNetwork,
        model_attr_loader: ModelAttributeLoader,
    ):
        self.state_encoder = state_encoder
        self.model_encoder = model_encoder
        self.q_network = q_network
        self.model_attr_loader = model_attr_loader

    def predict(
        self,
        history: str | list[dict],
        available_models: list[str]
    ) -> np.ndarray:
        """预测每个模型的 Q 值"""

        # 1. 编码状态
        if isinstance(history, list):
            history = self._format_history(history)
        state_embed = self.state_encoder(history)  # (state_dim,)

        # 2. 编码所有模型
        model_onehots = self.model_attr_loader.encode_batch(available_models)
        model_embeds = self.model_encoder(torch.tensor(model_onehots))  # (n_models, model_dim)

        # 3. 计算每个模型的 Q 值
        state_embed_expanded = state_embed.unsqueeze(0).expand(len(available_models), -1)
        q_values = self.q_network(state_embed_expanded, model_embeds)  # (n_models, 1)

        return q_values.squeeze(-1).detach().numpy()
```

---

## 5. 训练目标

### 5.1 Monte-Carlo Q-Learning

使用轨迹的实际回报作为目标：

```
y_t = S_final - λ · C_remaining
```

其中：
- `S_final`: 任务最终得分 (0-1 或 0-100)
- `C_remaining`: 从 step t 到结束的累计 cost
- `λ`: cost 权重系数

### 5.2 损失函数

```python
def compute_loss(
    q_network: QNetwork,
    batch: dict,  # 包含 state_embed, model_embed, target_q
) -> Tensor:
    pred_q = q_network(batch["state_embed"], batch["model_embed"])
    target_q = batch["target_q"]

    return F.mse_loss(pred_q, target_q)
```

### 5.3 Propensity Weighting (可选)

对于 off-policy 数据（如 roulette 采集），可以使用重要性采样：

```python
# weight = π(a|s) / μ(a|s)
# 其中 μ 是行为策略的 propensity
weight = 1.0 / propensity  # 如果目标策略是均匀分布
weighted_loss = weight * (pred_q - target_q) ** 2
```

---

## 6. 文件结构规划

```
research/training/
├── __init__.py
├── DESIGN.md              # 本文档
├── encoders.py            # StateEncoder, ModelEncoder, ModelAttributeLoader
├── q_network.py           # QNetwork, NeuralQFunction
├── dataset.py             # TrainingDataset (PyTorch Dataset)
├── trainer.py             # 训练循环
└── configs/
    └── default.yaml       # 默认训练配置
```

---

## 7. 待讨论问题

1. **State 编码粒度**: 编码完整历史 vs 只编码最近 N 轮？
2. **属性扩展**: 是否需要更多模型属性（如 multimodal 能力）？
3. **在线更新**: 是否支持 online learning / continual learning？

---

## 更新日志

- 2024-12: 初始设计，确定 one-hot 属性编码方案
