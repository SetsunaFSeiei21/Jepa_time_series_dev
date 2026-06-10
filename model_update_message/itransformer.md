# iTransformer 的 JEPA-style Backbone 重构说明

## 1. 基本信息

- 模型名称：iTransformer
- 代码位置：`src/models/time-series/itransformer.py`
- 建议说明文件位置：`model_update_message/itransformer.md`
- 本次处理目标：只对 iTransformer 的 backbone / forward 接口进行 JEPA-style 重构，不修改训练脚本、不修改 dataloader、不修改 ModelFactory。

本次重构的核心目标是让 iTransformer 同时支持两种训练模式：

1. `mode="pretrain"`：用于 JEPA-style pretraining，返回 `(Ey, Ey_pred)`，用于 embedding-space representation 对齐。
2. `mode="finetune"`：用于 downstream forecasting finetuning，返回最终预测结果 `y_pred`。

默认 `mode="finetune"`，因此原有 downstream 训练代码在不显式传入 `mode` 时，仍然走 finetune 路径。

---

## 2. 原始 iTransformer 的输入输出约定

当前项目中的 iTransformer 不是直接使用 `(B, N, T)` 格式，而是使用现有 dataloader 的格式：

```text
input_seq:       (B, T, N, 1)
input_features:  (B, T, F)
target_features: (B, L, F)
output:          (B, L, N, 1)
```

其中：

- `B` 是 batch size；
- `T` 是历史输入步长，即 `his_len`；
- `L` 是预测步长，即 `pred_len`；
- `N` 是节点数 / 变量数；
- `F` 是时间特征维度；
- 最后一维 `1` 是 value channel。

原始 forward 的主要逻辑如下：

1. 将 `input_seq` 从 `(B, T, N, 1)` squeeze 成 `(B, T, N)`；
2. 使用 `DataEmbedding_inverted` 将时间维投影到 hidden 维度；
3. 使用 Transformer encoder 在变量/token 维度上建模；
4. 使用 `projection = nn.Linear(d_model, pred_len)` 将 hidden representation 投影到预测时间步；
5. 对 projection 输出做 permute 和 node slicing，得到 `(B, L, N, 1)`。

---

## 3. 是否能够合理拆分为 encode / predict / decode

结论：iTransformer 可以合理拆分为 `encode / predict / decode` 三个阶段，但本次建议作为 v0 baseline 使用。

原因如下：

1. iTransformer 有明确的 encoder 路径：`enc_embedding + encoder`；
2. iTransformer 有明确的 value-space decoder / output head：`projection + permute + node slicing`；
3. encoder 输出的 hidden representation 形状稳定，可以作为 JEPA 的 embedding representation；
4. iTransformer 原本没有显式 JEPA predictor，但 `Ex` 和 `Ey` 的 shape 在当前 `his_len == pred_len` setting 下可以可靠确定，因此可以添加一个轻量 hidden-space predictor。

---

## 4. iTransformer 与 Autoformer / FEDformer 的关键区别

Autoformer 和 FEDformer 的 encoder 输出通常是：

```text
Ex: (B, T, d_model)
Ey: (B, L, d_model)
```

因此它们的 v0 predictor 可以先做时间维映射：

```text
T -> L
```

但 iTransformer 不同。它使用 `DataEmbedding_inverted`，会先把输入从：

```text
(B, T, N)
```

变换为：

```text
(B, N, T)
```

再通过 `nn.Linear(his_len, d_model)` 将每个变量/节点的历史时间序列投影成一个 hidden token。

因此 iTransformer 的 encoder 输出不是 `(B, T, d_model)`，而是：

```text
Ex: (B, N, d_model)
```

如果传入了 `input_features`，`DataEmbedding_inverted` 会把时间特征也当作额外 token 拼接进去，因此输出可能是：

```text
Ex: (B, N + F, d_model)
```

对应 label encoder 输出为：

```text
Ey: (B, N, d_model)
```

或：

```text
Ey: (B, N + F, d_model)
```

因此 iTransformer 的 JEPA predictor 不应该使用 `nn.Linear(his_len, pred_len)`，而应该使用 hidden-space predictor：

```text
(B, N, d_model) -> (B, N, d_model)
```

或：

```text
(B, N + F, d_model) -> (B, N + F, d_model)
```

---

## 5. encode 阶段设计

### 5.1 功能

`encode(...)` 用于将 input 或 label 编码到 embedding space。

对于输入序列：

```text
input_seq -> Ex
```

对于 label / target 序列：

```text
label -> Ey
```

这里的 `Ex` 和 `Ey` 不是最终值域预测结果，而是 iTransformer encoder 输出的 hidden representation。

### 5.2 支持的输入格式

为了兼容现有 dataloader 和通用 JEPA 设定，`encode` 前增加 `_to_btn(...)` 作为 shape 转换辅助函数。

支持以下三种输入格式：

```text
(B, T, N, 1)  # 当前项目 dataloader 格式
(B, T, N)     # squeeze 后的 iTransformer 内部格式
(B, N, T)     # 通用 JEPA-style time series 格式
```

内部统一转换为：

```text
(B, T, N)
```

对于 label，如果输入为 `(B, L, N, 1)`、`(B, L, N)` 或 `(B, N, L)`，同样会转换为：

```text
(B, L, N)
```

### 5.3 encode 输出 shape

如果不使用时间特征：

```text
Ex: (B, N, d_model)
Ey: (B, N, d_model)
```

如果使用时间特征，并且时间特征维度为 `F`：

```text
Ex: (B, N + F, d_model)
Ey: (B, N + F, d_model)
```

### 5.4 使用的原始模块

`encode` 直接复用 iTransformer 原始 encoder 路径：

```python
enc_out = self.enc_embedding(seq, seq_features)
enc_out, _ = self.encoder(enc_out, attn_mask=None)
return enc_out
```

因此没有引入任意中间张量，也没有为了 JEPA 强行构造 representation。

### 5.5 需要注意的长度限制

`DataEmbedding_inverted` 内部使用 `nn.Linear(c_in, d_model)`，其中 `c_in` 在当前模型初始化时为 `self.his_len`。

因此 `encode(label)` 要求 label 的时间长度能够匹配 `his_len`。

当前实验设置为：

```text
his_len == pred_len
```

因此 `label` 的长度 `L` 等于 `his_len`，可以复用同一个 `enc_embedding`。

如果后续设置为：

```text
his_len != pred_len
```

则当前方案不再可靠，需要单独设计 target encoder 或改造 embedding 层。

---

## 6. predict 阶段设计

### 6.1 功能

`predict(...)` 用于从 input embedding `Ex` 预测 label embedding `Ey_pred`：

```text
Ex -> Ey_pred
```

其中 `Ey_pred` 必须和 `Ey` shape 完全一致，方便 JEPA pretraining 阶段直接计算 representation-level loss：

```text
loss(Ey_pred, Ey)
```

例如 MSE loss、SmoothL1 loss 或 SIGReg-enhanced JEPA loss。

### 6.2 为什么不能使用时间维 Linear(his_len -> pred_len)

iTransformer 的时间维已经在 `DataEmbedding_inverted` 中被压入 hidden dimension：

```text
(B, T, N) -> (B, N, d_model)
```

此时 encoder 输出中已经没有显式的时间维 `T`，因此不应该像 Autoformer / FEDformer 那样写：

```python
nn.Linear(self.his_len, self.pred_len)
```

否则 Linear 会作用在错误的维度上。

### 6.3 本次采用的 predictor

由于 `Ex` 和 `Ey` 在当前 setting 下 shape 相同，采用一个轻量 hidden-space MLP predictor：

```python
self.jepa_predictor = nn.Sequential(
    nn.Linear(d_model, d_model),
    nn.GELU(),
    nn.Linear(d_model, d_model),
)
```

这个 predictor 是 `self.xxx`，因此会被 `model.parameters()` 正确收集，optimizer 可以更新它的参数。

### 6.4 predictor 的 shape 流程

不使用时间特征时：

```text
Ex:      (B, N, d_model)
Ey_pred: (B, N, d_model)
```

使用时间特征时：

```text
Ex:      (B, N + F, d_model)
Ey_pred: (B, N + F, d_model)
```

对应代码逻辑：

```python
Ey_pred = self.jepa_predictor(Ex)
return Ey_pred
```

### 6.5 v0 baseline 定位

这个 predictor 是第一版轻量 baseline，目标是优先打通 JEPA-style pretraining 和 downstream finetuning 的接口与 shape 路径。

它不是最终最强 predictor。后续可以替换为：

1. 更深的 MLP predictor；
2. token-wise Transformer predictor；
3. 使用 future query 的 cross-attention predictor；
4. 针对 iTransformer inverted token 结构设计的变量维 predictor。

---

## 7. decode 阶段设计

### 7.1 功能

`decode(...)` 用于将 predicted label embedding `Ey_pred` 解码回 forecasting 的值域空间：

```text
Ey_pred -> y_pred
```

其中不使用时间特征时：

```text
Ey_pred: (B, N, d_model)
```

使用时间特征时：

```text
Ey_pred: (B, N + F, d_model)
```

最终输出：

```text
y_pred: (B, L, N, 1)
```

`y_pred` 用于 downstream finetuning 阶段计算 MAE / MSE 等预测误差。

### 7.2 使用的原始模块

`decode` 复用 iTransformer 原始 output projection：

```python
dec_out = self.projection(Ey_pred).permute(0, 2, 1)[:, :, :self.node_num]
return dec_out.unsqueeze(-1)
```

其中：

1. `self.projection` 将 `d_model` 映射到 `pred_len`；
2. `permute(0, 2, 1)` 得到 `(B, pred_len, token_num)`；
3. `[:, :, :self.node_num]` 只保留真实节点 token，去掉可能由时间特征产生的额外 token；
4. `unsqueeze(-1)` 恢复 `(B, L, N, 1)` 格式。

---

## 8. forward 重构设计

重构后的 forward 建议使用如下接口：

```python
def forward(
    self,
    input_seq,
    input_features=None,
    target_features=None,
    target_seq=None,
    label=None,
    mode="finetune",
    *args,
    **kwargs,
):
    ...
```

这里显式加入 `target_seq=None` 是为了兼容当前项目中的 `BatchData.to_dict()`，因为 engine 传入模型时通常使用的是 `target_seq`，不是 `label`。

### 8.1 pretrain 模式

当 `mode == "pretrain"` 时：

```python
Ex = self.encode(input_seq, input_features)
Ey = self.encode(label, target_features)
Ey_pred = self.predict(Ex)
return Ey, Ey_pred
```

返回不使用时间特征时：

```text
Ey:      (B, N, d_model)
Ey_pred: (B, N, d_model)
```

返回使用时间特征时：

```text
Ey:      (B, N + F, d_model)
Ey_pred: (B, N + F, d_model)
```

需要显式检查：

```python
Ey_pred.shape == Ey.shape
```

如果 shape 不一致，应直接报错，而不是静默 broadcast 或截断。

### 8.2 finetune 模式

当 `mode == "finetune"` 时：

```python
Ex = self.encode(input_seq, input_features)
Ey_pred = self.predict(Ex)
y_pred = self.decode(Ey_pred)
return y_pred
```

返回：

```text
y_pred: (B, L, N, 1)
```

这个输出 shape 应与原始 iTransformer forward 的输出 shape 保持一致。

### 8.3 默认行为兼容

`mode` 默认设置为：

```python
mode="finetune"
```

因此原有训练代码如果仍然调用：

```python
model(input_seq, input_features, target_features)
```

会默认走 finetune 路径，并返回：

```text
(B, L, N, 1)
```

---

## 9. 与现有训练框架的兼容性

本次只处理 iTransformer backbone，不需要修改以下文件：

```text
ModelFactory
configs/model/itransformer.yaml
src/experiments/main.py
src/engines/jepa_pretrain.py
```

原因：

1. `ModelFactory` 已经能实例化 iTransformer，不需要改变构造方式；
2. `configs/model/itransformer.yaml` 已经提供 iTransformer 所需参数；
3. `src/experiments/main.py` 已经根据 `train_mode` 分流到 pretrain 或 finetune engine；
4. `JEPAPretrainEngine` 设计上就是调用模型的 `mode="pretrain"`，并优化 JEPA representation loss。

为了兼容现有 engine 可能传入 `target_seq` 而不是 `label` 的情况，forward 中需要加入：

```python
if label is None:
    label = target_seq

if label is None:
    label = kwargs.get("target_seq", None)
```

这样 pretrain 阶段可以通过 `target_seq` 获取 label。

---

## 10. 实际需要修改的文件

本轮 iTransformer 重构只需要修改两个文件：

```text
src/models/time-series/itransformer.py
model_update_message/itransformer.md
```

其中：

- `src/models/time-series/itransformer.py` 是真正的代码修改；
- `model_update_message/itransformer.md` 是本说明文档，用于记录修改原因、shape 约定和人工确认事项。

---

## 11. 是否需要人工确认

需要。

原因是：

iTransformer 原始 downstream 路径中直接使用 encoder 输出 `enc_out` 进入 `projection`。JEPA-style 重构后，finetune 路径中 projection 使用的是 predictor 生成的 `Ey_pred`。

从 shape 和模块边界看，这个设计是合理的：

```text
Ey_pred: (B, N, d_model)
```

或：

```text
Ey_pred: (B, N + F, d_model)
```

可以作为 projection 的输入。

但从训练行为上看，是否稳定、是否优于直接原始 forward，需要通过实验确认。

此外，由于 `DataEmbedding_inverted` 的输入时间长度固定为 `his_len`，当前版本优先用于 `his_len == pred_len` 的 setting。如果后续 `his_len != pred_len`，需要人工重新检查 label encoder 是否仍然适配。

因此建议至少做两级检查：

1. shape-level smoke test；
2. 小规模训练测试。

---

## 12. 基本检查命令

### 12.1 语法检查

```bash
python -m py_compile src/models/time-series/itransformer.py
```

### 12.2 查看修改内容

```bash
git diff src/models/time-series/itransformer.py
git diff model_update_message/itransformer.md
```

### 12.3 提交修改

```bash
git add src/models/time-series/itransformer.py model_update_message/itransformer.md
git commit -m "Refactor iTransformer backbone for JEPA-style pretraining"
git push
```

---

## 13. 最小 shape check 示例

建议写一个最小测试，确认 pretrain 和 finetune 两条路径都能跑通。

伪代码如下：

```python
Ey, Ey_pred = model(
    input_seq=input_seq,
    input_features=input_features,
    target_seq=target_seq,
    target_features=target_features,
    mode="pretrain",
)

assert Ey.shape == Ey_pred.shape

y_pred = model(
    input_seq=input_seq,
    input_features=input_features,
    target_features=target_features,
    mode="finetune",
)

assert y_pred.shape == target_seq.shape
```

期望 shape：

```text
Ey:      (B, N, d_model) 或 (B, N + F, d_model)
Ey_pred: (B, N, d_model) 或 (B, N + F, d_model)
y_pred:  (B, L, N, 1)
```

---

## 14. 处理结果总结

模型名：iTransformer

文件位置：`src/models/time-series/itransformer.py`

是否能合理拆分为 encode / predict / decode：能，但建议作为 v0 baseline 使用

是否已有 encoder：有，`enc_embedding + encoder`

是否已有 predictor：原始模型没有，新增 `jepa_predictor`

是否已有 decoder：有，`projection + permute + node slicing`

无法合理拆分的部分：无硬阻断

无法拆分的原因：无硬阻断

采取的修改：

1. 新增 `_to_btn(...)`，统一处理 `(B,T,N,1)`、`(B,T,N)`、`(B,N,T)`；
2. 新增 `encode(...)`，复用 `enc_embedding + encoder`；
3. 新增 `predict(...)`，使用 hidden-space MLP 预测 `Ey_pred`；
4. 新增 `decode(...)`，复用 `projection + permute + node slicing` 输出值域预测；
5. 重构 `forward(...)`，支持 `mode="pretrain"` 和 `mode="finetune"`。

pretrain forward 返回：

```text
Ey, Ey_pred
```

其中：

```text
Ey:      (B, N, d_model) 或 (B, N + F, d_model)
Ey_pred: (B, N, d_model) 或 (B, N + F, d_model)
```

finetune forward 返回：

```text
y_pred: (B, L, N, 1)
```

是否需要人工确认：需要

原因：

1. 虽然 shape 和模块边界合理，但 projection 使用 `Ey_pred` 而不是原始 `enc_out` 后的训练稳定性和最终预测性能需要实验验证；
2. `DataEmbedding_inverted` 的输入时间长度固定为 `his_len`，当前版本优先适用于 `his_len == pred_len` 的实验设置。
