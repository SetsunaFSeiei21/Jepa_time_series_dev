# Autoformer 的 JEPA-style Backbone 重构说明

## 1. 基本信息

- 模型名称：Autoformer
- 代码位置：`src/models/time-series/autoformer.py`
- 建议说明文件位置：`model_update_message/autoformer.md`
- 本次处理目标：只对 Autoformer 的 backbone / forward 接口进行 JEPA-style 重构，不修改训练脚本、不修改 dataloader、不修改 ModelFactory。

本次重构的核心目标是让 Autoformer 同时支持两种训练模式：

1. `mode="pretrain"`：用于 JEPA-style pretraining，返回 `(Ey, Ey_pred)`，用于 embedding-space representation 对齐。
2. `mode="finetune"`：用于 downstream forecasting finetuning，返回最终预测结果 `y_pred`。

默认 `mode="finetune"`，因此原有 downstream 训练代码在不显式传入 `mode` 时，仍然走 finetune 路径。

---

## 2. 原始 Autoformer 的输入输出约定

当前项目中的 Autoformer 不是直接使用 `(B, N, T)` 格式，而是使用现有 dataloader 的格式：

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
2. 对输入序列做 Autoformer decomposition，得到 seasonal / trend 分量；
3. 使用 `enc_embedding + encoder` 得到历史输入的 encoder hidden representation；
4. 使用 `dec_embedding + decoder` 生成预测输出；
5. 对 decoder 输出取最后 `pred_len` 个时间步，并 reshape 成 `(B, L, N, 1)`。

---

## 3. 是否能够合理拆分为 encode / predict / decode

结论：Autoformer 可以合理拆分为 `encode / predict / decode` 三个阶段。

原因如下：

1. Autoformer 有明确的 encoder 路径：`enc_embedding + encoder`；
2. Autoformer 有明确的 decoder 路径：`dec_embedding + decoder + final projection/slicing`；
3. encoder 输出的 hidden representation 形状稳定，可以作为 JEPA 的 embedding representation；
4. Autoformer 原本没有显式 JEPA predictor，但 `Ex` 和 `Ey` 的 shape 可以可靠确定，因此可以添加一个轻量 predictor。

---

## 4. encode 阶段设计

### 4.1 功能

`encode(...)` 用于将 input 或 label 编码到 embedding space。

对于输入序列：

```text
input_seq -> Ex
```

对于 label / target 序列：

```text
label -> Ey
```

这里的 `Ex` 和 `Ey` 不是最终值域预测结果，而是 Autoformer encoder 输出的 hidden representation。

### 4.2 支持的输入格式

为了兼容现有 dataloader 和通用 JEPA 设定，`encode` 前增加 `_to_btn(...)` 作为 shape 转换辅助函数。

支持以下三种输入格式：

```text
(B, T, N, 1)  # 当前项目 dataloader 格式
(B, T, N)     # squeeze 后的 Autoformer 内部格式
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

### 4.3 encode 输出 shape

对于 input：

```text
Ex: (B, T, d_model)
```

对于 label：

```text
Ey: (B, L, d_model)
```

### 4.4 使用的原始模块

`encode` 直接复用 Autoformer 原始 encoder 路径：

```python
enc_out = self.enc_embedding(seq, seq_features)
enc_out, _ = self.encoder(enc_out, attn_mask=None)
return enc_out
```

因此没有引入任意中间张量，也没有为了 JEPA 强行构造 representation。

---

## 5. predict 阶段设计

### 5.1 功能

`predict(...)` 用于从 input embedding `Ex` 预测 label embedding `Ey_pred`：

```text
Ex -> Ey_pred
```

其中 `Ey_pred` 必须和 `Ey` shape 完全一致，方便 JEPA pretraining 阶段直接计算 representation-level loss：

```text
loss(Ey_pred, Ey)
```

例如 MSE loss、SmoothL1 loss 或 SIGReg-enhanced JEPA loss。

### 5.2 为什么可以添加 predictor

Autoformer 原本没有显式 predictor，但其 encoder 输出 shape 是可靠的：

```text
Ex: (B, T, d_model)
Ey: (B, L, d_model)
```

这里 hidden/channel 维度都是 `d_model`，因此不需要做 hidden dimension 映射，只需要做时间维映射：

```text
T -> L
```

因此可以在 `__init__` 中新增：

```python
self.d_model = d_model
self.jepa_time_predictor = nn.Linear(self.his_len, self.pred_len)
```

这个 predictor 是 `self.xxx`，因此会被 `model.parameters()` 正确收集，optimizer 可以更新它的参数。

### 5.3 predictor 的 shape 流程

`nn.Linear` 默认作用在最后一维，因此需要先 transpose：

```text
Ex:      (B, T, d_model)
x:       (B, d_model, T)
mapped:  (B, d_model, L)
Ey_pred: (B, L, d_model)
```

对应代码逻辑：

```python
x = Ex.transpose(1, 2)        # (B, d_model, T)
x = self.jepa_time_predictor(x)  # (B, d_model, L)
Ey_pred = x.transpose(1, 2)   # (B, L, d_model)
return Ey_pred
```

---

## 6. decode 阶段设计

### 6.1 功能

`decode(...)` 用于将 predicted label embedding `Ey_pred` 解码回 forecasting 的值域空间：

```text
Ey_pred -> y_pred
```

其中：

```text
Ey_pred: (B, L, d_model)
y_pred:  (B, L, N, 1)
```

`y_pred` 用于 downstream finetuning 阶段计算 MAE / MSE 等预测误差。

### 6.2 使用的原始模块

`decode` 复用 Autoformer 原始 decoder 相关模块：

1. `self.decomp`：构造 seasonal / trend 初始化；
2. `self.dec_embedding`：构造 decoder input embedding；
3. `self.decoder`：执行 Autoformer decoder；
4. `trend_part + seasonal_part`：得到预测值；
5. `[:, -self.pred_len:, :]`：取最后 `pred_len` 个预测时间步；
6. `unsqueeze(-1)`：恢复 `(B, L, N, 1)` 格式。

### 6.3 decode 所需输入

虽然 `decode` 的核心输入是 `Ey_pred`，但 Autoformer decoder 还需要原始输入序列构造 seasonal/trend 初始化，因此 `decode` 还需要：

```text
input_seq
input_features
target_features
```

因此推荐接口为：

```python
def decode(self, Ey_pred, input_seq=None, input_features=None, target_features=None):
    ...
```

---

## 7. forward 重构设计

重构后的 forward 建议使用如下接口：

```python
def forward(
    self,
    input_seq,
    input_features=None,
    target_features=None,
    label=None,
    mode="finetune",
    *args,
    **kwargs,
):
    ...
```

### 7.1 pretrain 模式

当 `mode == "pretrain"` 时：

```python
Ex = self.encode(input_seq, input_features)
Ey = self.encode(label, target_features)
Ey_pred = self.predict(Ex)
return Ey, Ey_pred
```

返回：

```text
Ey:      (B, L, d_model)
Ey_pred: (B, L, d_model)
```

需要显式检查：

```python
Ey_pred.shape == Ey.shape
```

如果 shape 不一致，应直接报错，而不是静默 broadcast 或截断。

### 7.2 finetune 模式

当 `mode == "finetune"` 时：

```python
Ex = self.encode(input_seq, input_features)
Ey_pred = self.predict(Ex)
y_pred = self.decode(
    Ey_pred,
    input_seq=input_seq,
    input_features=input_features,
    target_features=target_features,
)
return y_pred
```

返回：

```text
y_pred: (B, L, N, 1)
```

这个输出 shape 应与原始 Autoformer forward 的输出 shape 保持一致。

### 7.3 默认行为兼容

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

## 8. 与现有训练框架的兼容性

本次只处理 Autoformer backbone，不需要修改以下文件：

```text
ModelFactory
configs/model/autoformer.yaml
src/experiments/main.py
src/engines/jepa_pretrain.py
```

原因：

1. `ModelFactory` 已经能实例化 Autoformer，不需要改变构造方式；
2. `configs/model/autoformer.yaml` 已经提供 Autoformer 所需参数；
3. `src/experiments/main.py` 已经根据 `train_mode` 分流到 pretrain 或 finetune engine；
4. `JEPAPretrainEngine` 设计上就是调用模型的 `mode="pretrain"`，并优化 JEPA representation loss。

为了兼容现有 engine 可能传入 `target_seq` 而不是 `label` 的情况，forward 中需要加入：

```python
if label is None:
    label = kwargs.get("target_seq", None)
```

这样 pretrain 阶段可以通过 `target_seq` 获取 label。

---

## 9. 实际需要修改的文件

本轮 Autoformer 重构只需要修改两个文件：

```text
src/models/time-series/autoformer.py
model_update_message/autoformer.md
```

其中：

- `src/models/time-series/autoformer.py` 是真正的代码修改；
- `model_update_message/autoformer.md` 是本说明文档，用于记录修改原因、shape 约定和人工确认事项。

---

## 10. 是否需要人工确认

需要。

原因是：

Autoformer 原始 decoder 中使用 historical encoder output 作为 cross representation。JEPA-style 重构后，finetune 路径中 decoder 使用的是 predictor 生成的 `Ey_pred` 作为 cross representation。

从 shape 和模块边界看，这个设计是合理的：

```text
Ey_pred: (B, L, d_model)
```

可以作为 decoder 的 cross hidden representation。

但从训练行为上看，是否稳定、是否优于直接原始 forward，需要通过实验确认。因此建议至少做两级检查：

1. shape-level smoke test；
2. 小规模训练测试。

---

## 11. 基本检查命令

### 11.1 语法检查

```bash
python -m py_compile src/models/time-series/autoformer.py
```

### 11.2 查看修改内容

```bash
git diff src/models/time-series/autoformer.py
git diff model_update_message/autoformer.md
```

### 11.3 提交修改

```bash
git add src/models/time-series/autoformer.py model_update_message/autoformer.md
git commit -m "Refactor Autoformer backbone for JEPA-style pretraining"
git push
```

---

## 12. 最小 shape check 示例

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
Ey:      (B, L, d_model)
Ey_pred: (B, L, d_model)
y_pred:  (B, L, N, 1)
```

---

## 13. 处理结果总结

模型名：Autoformer

文件位置：`src/models/time-series/autoformer.py`

是否能合理拆分为 encode / predict / decode：能

是否已有 encoder：有，`enc_embedding + encoder`

是否已有 predictor：原始模型没有，新增 `jepa_time_predictor`

是否已有 decoder：有，`dec_embedding + decoder + final projection/slicing`

无法合理拆分的部分：无

无法拆分的原因：无

采取的修改：

1. 新增 `_to_btn(...)`，统一处理 `(B,T,N,1)`、`(B,T,N)`、`(B,N,T)`；
2. 新增 `encode(...)`，复用 `enc_embedding + encoder`；
3. 新增 `predict(...)`，使用 `nn.Linear(self.his_len, self.pred_len)` 做时间维预测；
4. 新增 `_build_decoder_inputs(...)`，封装 Autoformer decoder 所需的 seasonal/trend 初始化；
5. 新增 `decode(...)`，复用 `dec_embedding + decoder` 输出值域预测；
6. 重构 `forward(...)`，支持 `mode="pretrain"` 和 `mode="finetune"`。

pretrain forward 返回：

```text
Ey, Ey_pred
```

其中：

```text
Ey:      (B, L, d_model)
Ey_pred: (B, L, d_model)
```

finetune forward 返回：

```text
y_pred: (B, L, N, 1)
```

是否需要人工确认：需要

原因：虽然 shape 和模块边界合理，但 decoder 使用 `Ey_pred` 作为 cross representation 后的训练稳定性和最终预测性能需要实验验证。
