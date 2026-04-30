# ASTGCN JEPA Backbone Refactor Report

## Model Information

- **Model name:** ASTGCN
- **Original file position:** `src/models/astgcn.py`
- **New proposed file:** `src/models/astgcn_jepa.py`
- **New class name:** `ASTGCN_JEPA`
- **Target use case:** JEPA-style hidden-space pretraining and downstream finetuning for time series forecasting.

## Whether ASTGCN Can Be Split into `encode / predict / decode`

ASTGCN can be reasonably split into `encode`, `predict`, and `decode`, but with one important constraint: ASTGCN's temporal and spatial attention layers contain fixed `seq_len`-dependent parameters. Therefore, the current refactor is valid when the sequence passed into `encode` has length `self.his_len`.

In the current experiment settings, `his_len == pred_len` for both `short` and `long` tasks, so both `input_seq` and `target_seq` can be encoded by the same ASTGCN encoder. If future settings use `his_len != pred_len`, this implementation should not be used directly for JEPA pretraining without further modification.

## Existing Components

### Existing Encoder

Yes.

The original ASTGCN backbone before `final_conv` can be treated as the encoder:

```python
for block in self.BlockList:
    x = block(x)
```

The output of `BlockList` is a meaningful hidden representation after temporal attention, spatial attention, Chebyshev graph convolution, temporal convolution, residual connection, and normalization.

### Existing Predictor

No.

The original ASTGCN directly decodes hidden features into value-space predictions through `final_conv`. It does not contain a hidden-space predictor. Therefore, the refactor adds two lightweight trainable predictor layers:

```python
self.jepa_time_predictor = nn.Linear(self.encoder_time_len, self.encoder_time_len)
self.jepa_dim_predictor = nn.Linear(self.encoder_out_dim, self.encoder_out_dim)
```

Both are registered as `self.xxx`, so their parameters are included in `model.parameters()`.

### Existing Decoder

Yes.

The original `final_conv` is the value-space decoder:

```python
self.final_conv = nn.Conv2d(
    self.encoder_time_len,
    self.pred_len,
    kernel_size=(1, nb_time_filter),
)
```

It maps hidden representation to forecasting predictions.

## Shape Design

### Input Shapes

- `input_seq`: `(B, T, N, 1)`
- `target_seq` / `label`: `(B, L, N, 1)`
- `input_features`: optional `(B, T, F)` or `(B, T, N, F)`
- `target_features`: optional `(B, L, F)` or `(B, L, N, F)`

The model first concatenates sequence values and temporal/external features:

```text
(B, T, N, 1) + (B, T, N, F) -> (B, T, N, input_dim)
```

Then it rearranges the tensor into the original ASTGCN format:

```text
(B, T, N, input_dim) -> (B, N, input_dim, T)
```

### Encoder Output

The `encode` function returns:

```text
Ex / Ey: (B, encoder_time_len, N, encoder_out_dim)
```

where:

```text
encoder_time_len = int(self.his_len / time_stride)
encoder_out_dim = nb_time_filter
```

For the default ASTGCN config:

```yaml
nb_time_filter: 64
time_stride: 1
```

so:

```text
encoder_time_len = his_len
encoder_out_dim = 64
```

### Predictor Output

The `predict` function maps:

```text
Ex:      (B, encoder_time_len, N, encoder_out_dim)
Ey_pred: (B, encoder_time_len, N, encoder_out_dim)
```

Therefore:

```text
Ey_pred.shape == Ey.shape
```

This satisfies the JEPA pretraining requirement.

### Decoder Output

The `decode` function maps:

```text
Ey_pred: (B, encoder_time_len, N, encoder_out_dim)
y_pred:  (B, pred_len, N, output_dim)
```

For the current forecasting task, `output_dim = 1`, so:

```text
y_pred: (B, pred_len, N, 1)
```

This matches the original ASTGCN forecasting output shape.

## Forward Behavior

### Pretraining Mode

```python
Ey, Ey_pred = model(
    input_seq=batch.input_seq,
    input_features=batch.input_features,
    target_seq=batch.target_seq,
    target_features=batch.target_features,
    mode="pretrain",
)
```

Returns:

```text
Ey:      encoded target representation
Ey_pred: predicted target representation from encoded input
```

Both have shape:

```text
(B, encoder_time_len, N, encoder_out_dim)
```

### Finetuning Mode

```python
y_pred = model(
    input_seq=batch.input_seq,
    input_features=batch.input_features,
    mode="finetune",
)
```

Returns:

```text
y_pred: (B, pred_len, N, 1)
```

### Default Mode

The default `mode` is:

```python
mode="finetune"
```

Therefore, calling the model without a `mode` argument keeps forecasting behavior compatible with the downstream finetuning engine.

## Important Limitation

ASTGCN is not fully sequence-length agnostic. Its attention layers include parameters whose shapes depend on `seq_len`, for example:

```python
self.be = nn.Parameter(torch.FloatTensor(1, seq_len, seq_len))
self.Ve = nn.Parameter(torch.FloatTensor(seq_len, seq_len))
self.W1 = nn.Parameter(torch.FloatTensor(seq_len))
self.W2 = nn.Parameter(torch.FloatTensor(in_channels, seq_len))
```

Because of this, `encode(label)` is only safe when:

```text
label.shape[1] == self.his_len
```

In the current short/long settings, this condition is satisfied because:

```text
his_len == pred_len
```

If future experiments use `his_len != pred_len`, possible solutions are:

1. create a separate target encoder initialized with `seq_len = pred_len`;
2. redesign ASTGCN attention layers to support dynamic sequence lengths;
3. perform JEPA alignment only on sequence windows with equal input and target lengths.

## Integration Notes

Since this refactor uses a new model file and class name, the repo needs two small integration changes before direct Hydra usage.

### 1. Add Model Config

Create:

```text
configs/model/astgcn_jepa.yaml
```

Suggested content:

```yaml
_target_: 'src.models.astgcn_jepa.ASTGCN_JEPA'
order: 3
nb_block: 2
nb_chev_filter: 64
nb_time_filter: 64
time_stride: 1
```

### 2. Register Model Name in `ModelFactory`

Add this entry to `_BUILDER_REGISTRY` in `src/factories/model.py`:

```python
"ASTGCN_JEPA": ASTGCNBuilder,
```

This keeps graph processing unchanged, because ASTGCN and ASTGCN_JEPA both use Chebyshev graph polynomials through the existing `ASTGCNBuilder`.

## Smoke Test Commands

After placing the new model file and adding config/factory registration, run:

```bash
PYTHONPATH=. python -m src.experiments.pretrain \
  dataset=pems03 \
  model=astgcn_jepa \
  task=short \
  exp=jepa_pretrain \
  exp.max_epochs=1 \
  exp.save_freq=1 \
  exp.num_workers=0 \
  hydra.run.dir=debug_smoke/astgcn_pretrain
```

Then run:

```bash
PYTHONPATH=. python -m src.experiments.finetune \
  dataset=pems03 \
  model=astgcn_jepa \
  task=short \
  exp=jepa_finetune \
  exp.max_epochs=1 \
  exp.num_workers=0 \
  'exp.pretrain_ckpt=""' \
  hydra.run.dir=debug_smoke/astgcn_finetune
```

Finally, test checkpoint loading:

```bash
PYTHONPATH=. python -m src.experiments.finetune \
  dataset=pems03 \
  model=astgcn_jepa \
  task=short \
  exp=jepa_finetune \
  exp.max_epochs=1 \
  exp.num_workers=0 \
  exp.pretrain_ckpt=/public/home/202411094934/python_project/Spatio-Temporal-Library/debug_smoke/astgcn_pretrain/jepa_pretrain_epoch_1.pth \
  hydra.run.dir=debug_smoke/astgcn_finetune_from_pretrain
```

## Final Decision Table

```text
模型名：ASTGCN
文件位置：src/models/astgcn.py
是否能合理拆分为 encode / predict / decode：可以，但要求 encode 输入长度等于 self.his_len
是否已有 encoder：有，BlockList 可作为 encoder
是否已有 predictor：无，新增 jepa_time_predictor 和 jepa_dim_predictor
是否已有 decoder：有，final_conv 可作为 decoder
无法合理拆分的部分：无；但存在序列长度限制
无法拆分的原因：不适用
采取的修改：新增 ASTGCN_JEPA，封装 encode/predict/decode，forward 支持 mode="pretrain" 和 mode="finetune"
pretrain forward 返回：(Ey, Ey_pred)，二者 shape 均为 (B, encoder_time_len, N, encoder_out_dim)
finetune forward 返回：y_pred，shape 为 (B, pred_len, N, 1)
是否需要人工确认：是
原因：如果未来 his_len != pred_len，ASTGCN 固定 seq_len attention 参数会使 target encoding 不再安全，需要人工设计 target encoder 或动态 attention
```
