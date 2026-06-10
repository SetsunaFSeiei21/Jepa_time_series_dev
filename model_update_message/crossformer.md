# Crossformer 的 JEPA-style Backbone 重构说明

## 1. 基本信息

- 模型名称：Crossformer
- 代码位置：`src/models/time-series/crossformer.py`
- 相关 layer 文件：`src/models/time-series/layers/Crossformer_EncDec.py`
- 建议说明文件位置：`model_update_message/crossformer.md`
- 本次处理目标：只对 Crossformer 的 backbone / forward 接口进行 JEPA-style 重构，不修改训练脚本、不修改 dataloader、不修改 ModelFactory。

本次重构的核心目标是让 Crossformer 同时支持两种训练模式：

1. `mode="pretrain"`：用于 JEPA-style pretraining，返回 `(Ey, Ey_pred)`，用于 embedding-space representation 对齐。
2. `mode="finetune"`：用于 downstream forecasting finetuning，返回最终预测结果 `y_pred`。

默认 `mode="finetune"`，因此原有 downstream 训练代码在不显式传入 `mode` 时，仍然走 finetune 路径。

---

## 2. 原始 Crossformer 的输入输出约定

当前项目中的 Crossformer 已经被初步接入为 `BaseModel` 风格，配置文件为：

```yaml
_target_: 'src.models.time-series.crossformer.Crossformer'

seg_len: 8
win_size: 4
factor: 10
d_model: 64
d_ff: 128
n_heads: 4
e_layers: 3
dropout: 0.1
baseline: false
```

当前项目 dataloader 的输入输出约定为：

```text
input_seq: (B, T, N, 1)
target_seq: (B, L, N, 1)
output:    (B, L, N, 1)
```

其中：

- `B` 是 batch size；
- `T` 是历史输入步长，即 `his_len`；
- `L` 是预测步长，即 `pred_len`；
- `N` 是节点数 / 变量数；
- 最后一维 `1` 是 value channel。

Crossformer 内部使用的是：

```text
x_seq: (B, T, N)
```

随后经过 DSW embedding 得到：

```text
x_embed: (B, N, in_seg_num, d_model)
```

其中：

```text
in_seg_num = pad_in_len / seg_len
```

---

## 3. 当前代码存在的基础问题

当前 `crossformer.py` 中直接调用：

```python
self.encoder = Encoder(
    e_layers,
    win_size,
    d_model,
    ...
)

self.decoder = Decoder(
    seg_len,
    e_layers + 1,
    d_model,
    ...
)
```

但当前仓库中的 `Crossformer_EncDec.py` 里：

```python
class Encoder(nn.Module):
    def __init__(self, attn_layers):
        ...

class Decoder(nn.Module):
    def __init__(self, layers):
        ...
```

也就是说，当前 `crossformer.py` 调用的 constructor 参数和真实 layer 文件不匹配。

因此本轮重构不仅要加入 `encode / predict / decode`，还要修正 Crossformer encoder/decoder 的构造方式：

1. 使用 `scale_block(...)` 构造 encoder layer list；
2. 使用 `DecoderLayer(...)` 构造 decoder layer list；
3. 再传给 `Encoder([...])` 和 `Decoder([...])`。

此外，`Crossformer_EncDec.py` 里的导入：

```python
from layers.SelfAttention_Family import TwoStageAttentionLayer
```

建议改为：

```python
from .SelfAttention_Family import TwoStageAttentionLayer
```

否则在包内相对导入时容易出现 `ModuleNotFoundError: No module named 'layers'`。

---

## 4. 是否能够合理拆分为 encode / predict / decode

结论：Crossformer 可以合理拆分为 `encode / predict / decode` 三个阶段，但需要注意它的 encoder output 是 multi-scale list。

原因如下：

1. Crossformer 有明确的 encoder 路径：
   ```text
   DSW_embedding + enc_pos_embedding + pre_norm + encoder
   ```
2. Crossformer 有明确的 decoder 路径：
   ```text
   dec_pos_embedding + decoder + slicing + baseline + unsqueeze
   ```
3. `Encoder.forward(...)` 返回的是 multi-scale representation list：
   ```text
   enc_out_list = [scale_0, scale_1, ..., scale_e]
   ```
4. `Decoder.forward(...)` 原本就是消费这个 multi-scale list：
   ```python
   predict_y = self.decoder(dec_in, enc_out_list)
   ```

因此 Crossformer 的 JEPA representation 不是单个 hidden tensor，而是一组 multi-scale hidden tensors。

---

## 5. encode 阶段设计

### 5.1 功能

`encode(...)` 用于将 input 或 label 编码到 Crossformer 的 multi-scale embedding space。

对于输入序列：

```text
input_seq -> Ex_list
```

对于 label / target 序列：

```text
label -> Ey_list
```

这里的 `Ex_list` 和 `Ey_list` 不是最终值域预测结果，而是 Crossformer encoder 输出的 multi-scale hidden representation。

### 5.2 支持的输入格式

为了兼容现有 dataloader 和通用 JEPA 设定，`encode` 前保留 `_to_btn(...)` 作为 shape 转换辅助函数。

支持以下三种输入格式：

```text
(B, T, N, 1)  # 当前项目 dataloader 格式
(B, T, N)     # Crossformer 内部格式
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

### 5.3 当前 v0 的长度约束

当前实验 setting 中：

```text
T == L
```

因此可以复用同一个 input encoder 对 input 和 label 编码。

本轮 v0 版本显式要求：

```text
seq.size(1) == self.in_len
```

也就是 label 的长度必须等于 `his_len`。如果未来 `pred_len != his_len`，则需要单独设计 target encoder 或 target positional embedding，不能直接复用当前 input encoder。

### 5.4 encode 输出 shape

`encode(...)` 返回一个 list：

```text
Ex_list = [
    (B, N, seg_num_0, d_model),
    (B, N, seg_num_1, d_model),
    ...,
    (B, N, seg_num_e, d_model),
]
```

其中：

- `seg_num_0 = pad_in_len / seg_len`；
- 后续 scale 的 `seg_num_i` 由 Crossformer 的 `SegMerging` 决定；
- 每个 scale 的 hidden dim 都是 `d_model`。

---

## 6. predict 阶段设计

### 6.1 功能

`predict(...)` 用于从 input multi-scale embedding `Ex_list` 预测 label multi-scale embedding `Ey_pred_list`：

```text
Ex_list -> Ey_pred_list
```

其中 `Ey_pred_list` 中每个 scale 的 shape 必须与 `Ey_list` 对应 scale 完全一致，从而可以计算 JEPA representation loss。

### 6.2 为什么不做时间维 Linear(t -> l)

Crossformer 的时间维已经被切成 segment，并且经过 multi-scale encoder 后变成：

```text
(B, N, seg_num_i, d_model)
```

在当前 v0 setting 中，`T == L`，因此 input 和 label 的每个 scale 具有相同的 `seg_num_i`。

因此 predictor 不需要做 segment 数量映射，只需要在每个 scale 上做 hidden-space MLP：

```text
d_model -> d_model
```

本轮使用：

```python
self.jepa_predictors = nn.ModuleList([
    nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.Linear(d_model, d_model),
    )
    for _ in range(e_layers + 1)
])
```

其中 `e_layers + 1` 对应 `Encoder.forward(...)` 返回的 multi-scale representation 数量。

### 6.3 predictor 输出 shape

对于每个 scale：

```text
Ex_i:      (B, N, seg_num_i, d_model)
Ey_pred_i: (B, N, seg_num_i, d_model)
```

因此：

```text
Ey_pred_list[i].shape == Ey_list[i].shape
```

---

## 7. JEPA pretrain loss 的 tensor 化方式

由于 Crossformer encoder 返回 multi-scale list，而当前 JEPA engine 通常希望接收 tensor 形式的 `(Ey, Ey_pred)`，因此本次新增：

```python
_pack_multiscale(...)
```

将 multi-scale list 打包成一个 tensor：

```text
[(B,N,seg_0,d), (B,N,seg_1,d), ...]
->
(B, N, total_feature_dim)
```

其中：

```text
total_feature_dim = sum_i(seg_i * d_model)
```

pretrain forward 返回：

```text
Ey:      (B, N, total_feature_dim)
Ey_pred: (B, N, total_feature_dim)
```

这样可以直接用于：

```python
MSE(Ey_pred, Ey)
```

或 SmoothL1 / SIGReg-enhanced JEPA loss。

---

## 8. decode 阶段设计

### 8.1 功能

`decode(...)` 用于将 predicted label multi-scale representation 解码回 forecasting 的值域空间：

```text
Ey_pred_list -> y_pred
```

其中：

```text
Ey_pred_list: list[(B, N, seg_num_i, d_model)]
y_pred:       (B, L, N, 1)
```

### 8.2 使用的原始模块

`decode` 复用 Crossformer 原始 decoder 路径：

1. 使用 `dec_pos_embedding` 构造 decoder input；
2. 调用 `self.decoder(dec_in, Ey_pred_list)`；
3. 对输出取前 `self.out_len` 个时间步；
4. 如果 `baseline=True`，加入历史均值 baseline；
5. `unsqueeze(-1)` 恢复 `(B, L, N, 1)`。

---

## 9. forward 重构设计

重构后的 forward 使用如下接口：

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

### 9.1 pretrain 模式

当 `mode == "pretrain"` 时：

```python
Ex_list = self.encode(input_seq)
Ey_list = self.encode(label)
Ey_pred_list = self.predict(Ex_list)

Ey = self._pack_multiscale(Ey_list)
Ey_pred = self._pack_multiscale(Ey_pred_list)

return Ey, Ey_pred
```

返回：

```text
Ey:      (B, N, total_feature_dim)
Ey_pred: (B, N, total_feature_dim)
```

### 9.2 finetune 模式

当 `mode == "finetune"` 时：

```python
Ex_list = self.encode(input_seq)
Ey_pred_list = self.predict(Ex_list)
y_pred = self.decode(Ey_pred_list, input_seq=input_seq)
return y_pred
```

返回：

```text
y_pred: (B, L, N, 1)
```

### 9.3 默认行为兼容

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

## 10. 与现有训练框架的兼容性

本次只处理 Crossformer backbone，不需要修改以下文件：

```text
src/experiments/main.py
src/engines/jepa_pretrain.py
configs/model/crossformer.yaml
src/factories/model.py
```

原因：

1. `configs/model/crossformer.yaml` 已经存在；
2. `ModelFactory` 已经注册 `Crossformer`；
3. `main.py` 已经根据 `train_mode` 分流到 pretrain 或 finetune engine；
4. `JEPAPretrainEngine` 设计上就是调用模型的 `mode="pretrain"`，并优化 JEPA representation loss。

本轮需要修改：

```text
src/models/time-series/crossformer.py
src/models/time-series/layers/Crossformer_EncDec.py
model_update_message/crossformer.md
```

---

## 11. 是否需要人工确认

需要。

原因是：

Crossformer 原始 decoder 使用历史输入 encoder output 作为 cross representation。JEPA-style 重构后，finetune 路径中 decoder 使用的是 predictor 生成的 `Ey_pred_list` 作为 cross representation。

从 shape 和模块边界看，这个设计是合理的，因为 decoder 原本消费的就是 multi-scale encoder representation list。

但从训练行为上看，是否稳定、是否优于直接原始 forward，需要通过实验确认。因此建议至少做两级检查：

1. shape-level smoke test；
2. 小规模训练测试。

此外，当前 v0 版本显式假设：

```text
his_len == pred_len
```

如果未来 `his_len != pred_len`，需要单独处理 target-side encoder / predictor 的 segment 数量映射。

---

## 12. 基本检查命令

### 12.1 语法检查

```bash
python -m py_compile src/models/time-series/layers/Crossformer_EncDec.py
python -m py_compile src/models/time-series/crossformer.py
```

### 12.2 查看修改内容

```bash
git diff src/models/time-series/crossformer.py
git diff src/models/time-series/layers/Crossformer_EncDec.py
git diff model_update_message/crossformer.md
```

### 12.3 提交修改

```bash
git add src/models/time-series/crossformer.py src/models/time-series/layers/Crossformer_EncDec.py model_update_message/crossformer.md
git commit -m "Refactor Crossformer backbone for JEPA-style pretraining"
git push
```

---

## 13. 最小 shape check 示例

建议写一个最小测试，确认 pretrain 和 finetune 两条路径都能跑通。

```python
Ey, Ey_pred = model(
    input_seq=input_seq,
    target_seq=target_seq,
    mode="pretrain",
)

assert Ey.shape == Ey_pred.shape

y_pred = model(
    input_seq=input_seq,
    mode="finetune",
)

assert y_pred.shape == target_seq.shape
```

期望 shape：

```text
Ey:      (B, N, total_feature_dim)
Ey_pred: (B, N, total_feature_dim)
y_pred:  (B, L, N, 1)
```

---

## 14. 处理结果总结

模型名：Crossformer

文件位置：`src/models/time-series/crossformer.py`

是否能合理拆分为 encode / predict / decode：能，但需要以 multi-scale list 作为内部 representation

是否已有 encoder：有，`DSW_embedding + enc_pos_embedding + pre_norm + encoder`

是否已有 predictor：原始模型没有，新增 `jepa_predictors`

是否已有 decoder：有，`dec_pos_embedding + decoder + slicing + baseline`

无法合理拆分的部分：无

无法拆分的原因：无

采取的修改：

1. 修正 `Crossformer_EncDec.py` 的相对导入；
2. 修正 `crossformer.py` 中 Encoder / Decoder 的构造方式；
3. 新增 `_to_btn(...)`，统一处理 `(B,T,N,1)`、`(B,T,N)`、`(B,N,T)`；
4. 新增 `_pad_input(...)`，封装 Crossformer input padding；
5. 新增 `_build_decoder_input(...)`，封装 decoder query；
6. 新增 `_pack_multiscale(...)`，将 multi-scale list 打包成 JEPA loss 可用 tensor；
7. 新增 `encode(...)`；
8. 新增 `predict(...)`；
9. 新增 `decode(...)`；
10. 重构 `forward(...)`，支持 `mode="pretrain"` 和 `mode="finetune"`。

pretrain forward 返回：

```text
Ey, Ey_pred
```

其中：

```text
Ey:      (B, N, total_feature_dim)
Ey_pred: (B, N, total_feature_dim)
```

finetune forward 返回：

```text
y_pred: (B, L, N, 1)
```

是否需要人工确认：需要

原因：

1. decoder 使用 `Ey_pred_list` 而不是原始历史 `enc_out_list` 作为 cross representation，训练稳定性和最终预测性能需要实验确认；
2. 当前 v0 版本依赖 `his_len == pred_len`，未来若 `his_len != pred_len`，需要单独设计 target encoder 或 segment/time predictor。
