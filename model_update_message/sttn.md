# STTN JEPA-style Backbone 重构报告

## 处理结果

模型名：STTN

文件位置：

- 原始模型：`src/models/sttn.py`
- 新增模型：`src/models/sttn_jepa.py`

是否能合理拆分为 `encode / predict / decode`：可以，但有条件。

是否已有 encoder：有。原始 STTN 中 `start_conv + SpatialTransformer + TemporalTransformer + BatchNorm/residual` 可以视为 backbone encoder，用于提取时空图 hidden representation。

是否已有 predictor：没有。原始 STTN 是标准 forecasting 模型，没有从 input hidden representation 预测 target hidden representation 的显式 predictor。因此新增了轻量 JEPA predictor。

是否已有 decoder：有。原始模型的 `end_conv_1 + end_conv_2` 是明确的输出 projection / forecasting head，可以封装为 `decode(...)`。

无法合理拆分的部分：无完全无法拆分部分。

无法拆分的原因：无。但存在一个重要限制：STTN 的 `TemporalTransformer` 内部使用固定长度的 temporal positional embedding，并且 `window_size` 初始化为 `self.his_len`。因此，当前重构版本要求被 `encode(...)` 的序列长度等于 `self.his_len`。在当前项目的 `short` 和 `long` setting 中，`his_len == pred_len`，所以 input 和 label 可以共享同一个 encoder；如果未来改成 `his_len != pred_len`，则需要人工设计 target encoder 或动态时间位置编码机制。

采取的修改：

1. 新增 `encode(...)`：
   - 输入 `seq` 和可选 `features`；
   - 将 `input_seq / target_seq` 与时间特征拼接；
   - 复用原始 `start_conv`、`s_modules`、`t_modules`、`bn`；
   - 输出 hidden representation，统一整理为 `(B, T, N, hidden_channels)`。

2. 新增 `predict(...)`：
   - 输入 `Ex`，shape 为 `(B, T, N, hidden_channels)`；
   - 先通过 `self.jepa_time_predictor = nn.Linear(self.his_len, self.his_len)` 在时间维上预测；
   - 再通过 `self.jepa_dim_predictor = nn.Linear(hidden_channels, hidden_channels)` 在 hidden/channel 维上预测；
   - 两个 predictor 都是 `self.xxx` 成员变量，能被 `model.parameters()` 收集并参与优化；
   - predictor 初始化为 identity，以尽量降低对 finetune 初始行为的破坏。

3. 新增 `decode(...)`：
   - 输入 `Ey_pred`，shape 为 `(B, T, N, hidden_channels)`；
   - 转换回原始 STTN head 所需的 `(B, hidden_channels, N, T)`；
   - 保留原始模型只取最后一个 hidden time step 的逻辑：`x = x[..., -1:]`；
   - 复用 `end_conv_1 + end_conv_2` 输出最终预测值。

4. 重构 `forward(...)`：
   - 默认 `mode="finetune"`；
   - `mode="pretrain"` 时返回 `(Ey, Ey_pred)`；
   - `mode="finetune"` 时返回 `y_pred`；
   - 保持现有 dataloader / engine 的参数兼容性，支持 `input_seq`、`input_features`、`target_seq`、`target_features`、`label` 等输入。

pretrain forward 返回：

```python
Ey, Ey_pred
```

其中：

```text
Ey.shape      = (B, L, N, hidden_channels)
Ey_pred.shape = (B, T, N, hidden_channels)
```

当前要求：

```text
T == L == self.his_len
```

因此实际可保证：

```text
Ey_pred.shape == Ey.shape
```

finetune forward 返回：

```python
y_pred
```

其中：

```text
y_pred.shape = (B, pred_len, N, output_dim)
```

当 `output_dim == 1` 时，该 shape 与原始 STTN 的输出形式兼容。

是否需要人工确认：部分需要。

原因：

1. 当前 `short` 和 `long` 配置中 `his_len == pred_len`，因此可以直接共享同一个 STTN encoder 编码 input 和 label。
2. 如果未来任务改成 `his_len != pred_len`，当前模型会主动报错，而不是强行对齐不可靠的 hidden representation。
3. 如果需要支持 `his_len != pred_len`，建议人工确认以下方案之一：
   - 为 target sequence 单独建立 target encoder；
   - 将 `TemporalTransformer` 的 positional embedding 和 window mask 改为动态长度版本；
   - 或在 predictor 中显式完成 encoded time length 的映射，并确保 decode 输入语义合理。

## 建议新增配置

建议不要覆盖原始 `configs/model/sttn.yaml`，而是新增：

```yaml
# configs/model/sttn_jepa.yaml
_target_: 'src.models.sttn_jepa.STTN'
blocks: 2
mlp_expand: 2
hidden_channels: 32
end_channels: 512
dropout: 0.1
```

这样可以保留原始 STTN，同时让 JEPA 实验通过：

```bash
model=sttn_jepa
```

来调用新模型。

## 推荐 smoke test

在 repo 根目录下执行：

```bash
PYTHONPATH=. python -m src.experiments.pretrain \
  dataset=pems03 \
  model=sttn_jepa \
  task=short \
  exp=jepa_pretrain \
  exp.max_epochs=1 \
  exp.save_freq=1 \
  exp.num_workers=0 \
  hydra.run.dir=debug_smoke/sttn_pretrain
```

然后测试 finetune：

```bash
PYTHONPATH=. python -m src.experiments.finetune \
  dataset=pems03 \
  model=sttn_jepa \
  task=short \
  exp=jepa_finetune \
  exp.max_epochs=1 \
  exp.num_workers=0 \
  'exp.pretrain_ckpt=""' \
  hydra.run.dir=debug_smoke/sttn_finetune
```

如果这两个命令通过，则说明：

- `forward(mode="pretrain")` 可以返回 `(Ey, Ey_pred)`；
- `Ey_pred.shape == Ey.shape`；
- `forward(mode="finetune")` 可以返回 forecasting 结果；
- 新增 predictor 参数可以被 optimizer 收集。

## 总结

STTN 是可以进行 JEPA-style 改造的模型。它本身具有清晰的时空 Transformer backbone 和输出 projection head，因此可以自然拆分为：

```text
encode  = start_conv + SpatialTransformer + TemporalTransformer + BN/residual
predict = JEPA hidden-space time predictor + dim predictor
decode  = last hidden step + end_conv_1 + end_conv_2
```

当前版本最重要的限制是：由于 temporal positional embedding 和 window size 固定为 `self.his_len`，所以只建议在 `his_len == pred_len` 的实验设置下使用。该限制与当前 short/long setting 一致。
