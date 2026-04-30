# GWNET 的 JEPA-style Backbone 改造报告

## 模型名

GWNET / Graph WaveNet

## 文件位置

原始模型位置：

```text
src/models/gwnet.py
```

建议新增模型文件位置：

```text
src/models/gwnet_jepa.py
```

如果不想修改 `ModelFactory`，建议新配置文件使用：

```yaml
_target_: 'src.models.gwnet_jepa.GWNET'
```

这样 `model_name = model_kwargs["_target_"].split(".")[-1]` 仍然是 `GWNET`，可以复用现有 `ModelFactory` 中的 `GWNET` 构造逻辑。

---

## 是否能合理拆分为 encode / predict / decode

可以合理拆分。

Graph WaveNet 的原始 forward 结构大致是：

```text
input_seq + input_features
    -> start_conv
    -> dilated gated temporal convolutions
    -> graph convolution / adaptive adjacency
    -> skip hidden representation
    -> optional mlp_projection
    -> end_conv_1
    -> end_conv_2
    -> y_pred
```

因此可以把：

```text
start_conv + WaveNet blocks + GCN + skip aggregation
```

视为 encoder，把：

```text
optional mlp_projection + end_conv_1 + end_conv_2
```

视为 decoder。

---

## 是否已有 encoder

有。

原模型中 `start_conv`、`filter_convs`、`gate_convs`、`gconv`、`skip_convs`、`bn` 共同构成了能够提取时空 hidden representation 的 backbone。改造后的 `encode(...)` 复用了这些模块，并返回最终预测头之前的 skip hidden representation。

### encode 输入输出 shape

输入：

```text
seq:      (B, T_or_L, N, C_seq)
features: (B, T_or_L, F) 或 (B, T_or_L, N, F)
```

内部 Graph WaveNet 格式：

```text
(B, C_in, N, T_or_L)
```

encoder 输出：

```text
Ex / Ey: (B, t_enc, N, skip_channels)
```

其中：

```text
t_enc = max(seq_len, receptive_field) - receptive_field + 1
```

当前仓库的任务配置是：

```text
short: his_len = 12, pred_len = 12
long:  his_len = 96, pred_len = 96
```

所以当前实验 setting 下 input 和 label 的 encoded time length 是一致的。

---

## 是否已有 predictor

原模型没有显式 predictor。

因此新增了两个轻量 predictor：

```python
self.jepa_time_predictor = nn.Linear(source_encoded_time_dim, target_encoded_time_dim)
self.jepa_dim_predictor = nn.Linear(skip_channels, skip_channels)
```

其中：

```text
source_encoded_time_dim = encoded time length of input_seq
target_encoded_time_dim = encoded time length of target_seq / label
```

`predict(...)` 的输入输出为：

```text
Ex:      (B, t_src, N, skip_channels)
Ey_pred: (B, t_tgt, N, skip_channels)
```

如果 `t_src == t_tgt`，`jepa_time_predictor` 会用 identity 初始化。如果 hidden dim 相同，`jepa_dim_predictor` 也会用 identity 初始化。

新增 predictor 是 `self.xxx` 成员，因此参数会被 `model.parameters()` 收集到，可以被 optimizer 更新。

---

## 是否已有 decoder

有。

原模型最后的输出头由以下部分组成：

```text
optional mlp_projection
end_conv_1
end_conv_2
```

改造后的 `decode(...)` 复用了这些模块。

### decode 输入输出 shape

输入：

```text
Ey_pred: (B, t_tgt, N, skip_channels)
```

内部转换为：

```text
skip: (B, skip_channels, N, t_tgt)
```

输出：

```text
y_pred: (B, pred_len, N, output_dim)
```

在当前数据设定中，`output_dim` 通常为 1。

---

## 无法合理拆分的部分

无。

但有一个需要注意的约束：当 `mlp_projection` 存在时，decoder 期望输入 hidden 的时间维与 `self.mlp_input_dim` 一致。当前 `short` 和 `long` 配置下是兼容的，但如果以后修改 `his_len / pred_len / blocks / layers / kernel_size`，需要重新确认 encoded time length 与 `mlp_projection` 的输入维度是否一致。

---

## 无法拆分的原因

不适用。

该模型可以拆分。原模型的 skip representation 是比较自然的 embedding-space representation，因此适合作为 JEPA pretraining 的 hidden target。

---

## 采取的修改

新增文件：

```text
src/models/gwnet_jepa.py
```

主要修改如下：

1. 新增 `encode(seq, features=None)`：
   - 复用原始 Graph WaveNet backbone；
   - 返回最终预测头之前的 skip hidden representation；
   - 输出统一为 `(B, t_enc, N, skip_channels)`。

2. 新增 `predict(Ex)`：
   - 使用 `jepa_time_predictor` 进行 encoded time 维度映射；
   - 使用 `jepa_dim_predictor` 进行 hidden/channel 维度映射；
   - 输出 `Ey_pred`，保证和 `Ey` shape 对齐。

3. 新增 `decode(Ey_pred)`：
   - 将 `(B, t, N, D)` 转回 `(B, D, N, t)`；
   - 复用原模型的 `mlp_projection / end_conv_1 / end_conv_2`；
   - 输出最终 forecasting 结果。

4. 重构 `forward(...)`：
   - `mode="pretrain"`：返回 `(Ey, Ey_pred)`；
   - `mode="finetune"`：返回 `y_pred`；
   - 默认 `mode="finetune"`，保持和原训练 engine 兼容。

5. 保留原有图结构处理：
   - `supports`；
   - `adp_adj`；
   - adaptive adjacency；
   - `GCN`；
   - `nconv`；
   - `linear`。

---

## pretrain forward 返回

```python
Ey, Ey_pred = model(
    input_seq=batch.input_seq,
    input_features=batch.input_features,
    target_seq=batch.target_seq,
    target_features=batch.target_features,
    mode="pretrain",
)
```

返回：

```text
Ey:      (B, t_tgt, N, skip_channels)
Ey_pred: (B, t_tgt, N, skip_channels)
```

满足：

```python
Ey_pred.shape == Ey.shape
```

可直接用于 JEPA hidden-space loss，例如：

```text
MSE(Ey_pred, Ey)
SmoothL1(Ey_pred, Ey)
SIGReg-style hidden regularization
```

---

## finetune forward 返回

```python
y_pred = model(
    input_seq=batch.input_seq,
    input_features=batch.input_features,
    mode="finetune",
)
```

返回：

```text
y_pred: (B, pred_len, N, output_dim)
```

与原 Graph WaveNet 输出预测 shape 保持一致。

---

## 是否需要人工确认

需要轻度人工确认。

原因不是模型不能拆，而是 Graph WaveNet 的 decoder 中存在一个和 temporal length 相关的 `mlp_projection`：

```python
self.mlp_input_dim = (
    self.his_len - (self.kernel_size - 1) * (1 + self.layers) * blocks
)
```

当前仓库默认配置下：

```text
kernel_size = 2
blocks = 4
layers = 2
```

因此：

```text
short: his_len = 12 -> mlp_input_dim = 0，不使用 mlp_projection
long:  his_len = 96 -> mlp_input_dim = 84，使用 mlp_projection
```

Graph WaveNet 的实际 encoded time length 为：

```text
t_enc = max(seq_len, receptive_field) - receptive_field + 1
```

在当前 `short / long` 配置下，该值与 decoder 逻辑兼容。若后续修改 `his_len`、`pred_len`、`kernel_size`、`blocks` 或 `layers`，需要重新确认：

```text
Ey_pred 的时间维是否等于 mlp_projection 期望的输入维度
```

否则 decoder 可能会报 temporal dimension mismatch。

---

## 建议配套配置

建议新增：

```text
configs/model/gwnet_jepa.yaml
```

内容可以写成：

```yaml
_target_: 'src.models.gwnet_jepa.GWNET'
init_dim: 32
skip_channels: 256
end_channels: 512
dropout: 0.1
kernel_size: 2
blocks: 4
layers: 2
adp_adj: true
```

这样可以不修改 `ModelFactory`，因为目标类名仍然是 `GWNET`，现有 `GWNetBuilder` 仍然可以根据 `model_name == "GWNET"` 处理 `supports`、`residual_channels`、`dilation_channels` 等参数。

---

## 推荐 smoke test

放入文件后，可以先执行：

```bash
PYTHONPATH=. python -m src.experiments.pretrain \
  dataset=pems03 \
  model=gwnet_jepa \
  task=short \
  exp=jepa_pretrain \
  exp.max_epochs=1 \
  exp.save_freq=1 \
  exp.num_workers=0 \
  hydra.run.dir=debug_smoke/gwnet_pretrain
```

再测试 finetune：

```bash
PYTHONPATH=. python -m src.experiments.finetune \
  dataset=pems03 \
  model=gwnet_jepa \
  task=short \
  exp=jepa_finetune \
  exp.max_epochs=1 \
  exp.num_workers=0 \
  'exp.pretrain_ckpt=""' \
  hydra.run.dir=debug_smoke/gwnet_finetune
```

如果以上两个命令能跑通，则说明：

```text
forward(mode="pretrain") 能返回 Ey/Ey_pred
forward(mode="finetune") 能返回 y_pred
新增 predictor 参数可以被 optimizer 更新
```
