# STAEformer JEPA-style Backbone 改造报告

## 模型名

STAEformer

## 原始文件位置

`src/models/staeformer.py`

## 新文件建议位置

`src/models/staeformer_jepa.py`

## 是否能合理拆分为 encode / predict / decode

可以。

STAEformer 的结构非常适合拆分为 JEPA 三阶段：

1. `input_proj + temporal embedding + spatial embedding + adaptive embedding + temporal attention + spatial attention` 可以作为 `encode`；
2. 原始模型没有显式 hidden-space predictor，因此新增轻量 JEPA predictor；
3. 原始 `output_proj` / `temporal_proj + output_proj` 可以作为 `decode`。

## 是否已有 encoder

有。

原始 `forward` 中从 `input_proj` 到 `attn_layers_t`、`attn_layers_s` 结束之前的部分就是比较清晰的 encoder。其输出为：

```text
(B, his_len, N, model_dim)
```

其中：

- `B` 是 batch size；
- `his_len` 是输入历史长度；
- `N` 是节点数量；
- `model_dim = input_embedding_dim + temporal_embedding_dim + spatial_embedding_dim + adaptive_embedding_dim`。

## 是否已有 predictor

没有。

原始 STAEformer 是直接将 attention backbone 的 hidden representation 输入到输出投影层中得到预测值，没有显式 `Ex -> Ey_pred` 的 hidden-space predictor。

因此新增：

```python
self.jepa_time_predictor = nn.Linear(self.his_len, self.his_len)
self.jepa_dim_predictor = nn.Linear(self.model_dim, self.model_dim)
```

两者都初始化为 identity，以尽量降低对原始 finetune 行为的扰动。

## 是否已有 decoder

有。

原始模型有两种输出路径：

### use_mixed_proj = True

```python
self.output_proj = nn.Linear(self.his_len * self.model_dim, self.pred_len * self.output_dim)
```

这一分支将 `(B, his_len, N, model_dim)` reshape 后直接映射到 `(B, pred_len, N, output_dim)`。

### use_mixed_proj = False

```python
self.temporal_proj = nn.Linear(self.his_len, self.pred_len)
self.output_proj = nn.Linear(self.model_dim, self.output_dim)
```

这一分支先做时间维映射，再做输出维映射。

这两种情况都可以自然封装为 `decode`。

## 无法合理拆分的部分

无。

但是需要注意：当 `adaptive_embedding_dim > 0` 时，原始 `adaptive_embedding` 的 shape 是：

```text
(his_len, node_num, adaptive_embedding_dim)
```

所以 encoder 对输入序列长度有约束：输入给 `encode` 的序列长度必须等于 `self.his_len`。

## 无法拆分的原因

不适用。

## 采取的修改

### 1. 新增 `_build_feature_embeddings(...)`

负责构建输入 token embedding，包括：

- value embedding；
- temporal feature embedding；
- node spatial embedding；
- adaptive embedding。

输入：

```text
seq: (B, T, N, 1)
features: (B, T, F) 或 (B, T, N, F)
```

输出：

```text
(B, T, N, model_dim)
```

### 2. 新增 `encode(...)`

封装原始 STAEformer 的 embedding + temporal attention + spatial attention。

输入：

```text
seq: (B, T, N, 1)
features: (B, T, F) 或 (B, T, N, F)
```

输出：

```text
Ex 或 Ey: (B, T, N, model_dim)
```

当前实验中要求 `T == his_len`。

### 3. 新增 `predict(...)`

输入：

```text
Ex: (B, his_len, N, model_dim)
```

输出：

```text
Ey_pred: (B, his_len, N, model_dim)
```

predictor 先沿时间维做映射，再沿 hidden 维做映射。

### 4. 新增 `decode(...)`

输入：

```text
Ey_pred: (B, his_len, N, model_dim)
```

输出：

```text
y_pred: (B, pred_len, N, output_dim)
```

该函数复用原始 STAEformer 的输出投影逻辑。

### 5. 重构 `forward(...)`

新的 `forward` 支持：

```python
mode="pretrain"
mode="finetune"
```

默认：

```python
mode="finetune"
```

以保持下游训练兼容性。

## pretrain forward 返回

```python
return Ey, Ey_pred
```

其中：

```text
Ey:      (B, his_len, N, model_dim)
Ey_pred: (B, his_len, N, model_dim)
```

满足：

```python
Ey_pred.shape == Ey.shape
```

## finetune forward 返回

```python
return y_pred
```

其中：

```text
y_pred: (B, pred_len, N, output_dim)
```

与原始 STAEformer 的输出 shape 保持一致。

## 是否需要人工确认

需要轻微确认。

## 原因

主要确认点是：

### 1. his_len 与 pred_len 是否总是相等

当前项目的 `short` 和 `long` 配置中：

```text
short: his_len = 12, pred_len = 12
long:  his_len = 96, pred_len = 96
```

因此当前设定下 `input_seq` 和 `label` 都可以被同一个 STAEformer encoder 编码。

但如果未来改为 `his_len != pred_len`，由于 `adaptive_embedding` 和 `temporal attention` 当前绑定 `his_len`，target sequence 不能直接用同一个 encoder 编码，需要进一步设计：

- 单独的 target adaptive embedding；
- 或关闭 adaptive embedding；
- 或对 target length 做额外投影 / padding；
- 或构造 target-specific encoder。

### 2. finetune 是否应该经过 JEPA predictor

当前实现中 finetune 路径为：

```text
input_seq -> encode -> predict -> decode -> y_pred
```

这样和 JEPA pretraining 的结构保持一致。

如果希望完全保留原始 STAEformer 的 finetune 行为，也可以提供一个选项，让 finetune 直接：

```text
input_seq -> encode -> decode -> y_pred
```

但为了和 STGCN_JEPA / GWNET_JEPA / STTN_JEPA 的接口统一，当前版本默认经过 `predict`。

## 推荐新增配置文件

建议新增：

```yaml
# configs/model/staeformer_jepa.yaml
_target_: 'src.models.staeformer_jepa.STAEformer'
input_embedding_dim: 32
temporal_embedding_dim: 32
spatial_embedding_dim: 32
adaptive_embedding_dim: 32
feed_forward_dim: 128
num_heads: 4
num_layers: 2
use_mixed_proj: true
dropout: 0.3
```

这样不会覆盖原始 `configs/model/staeformer.yaml`，可以通过：

```bash
model=staeformer_jepa
```

单独调用 JEPA 改造版本。

## 总结

STAEformer 是目前这批模型中最适合 JEPA-style 改造的模型之一，因为它本身就具有非常清楚的：

```text
embedding -> attention backbone -> output projection
```

结构。

因此本次改造是合理的：

```text
encode: input_proj + embeddings + temporal/spatial attention
predict: JEPA hidden-space predictor
decode: 原始 output projection
```

当前实现满足：

- `forward(mode="pretrain")` 返回 `(Ey, Ey_pred)`；
- `forward(mode="finetune")` 返回 `y_pred`；
- 默认 `mode="finetune"`；
- `Ey_pred.shape == Ey.shape`；
- `y_pred` shape 与原始模型输出一致；
- 新增 predictor 参数可被 `model.parameters()` 收集。
