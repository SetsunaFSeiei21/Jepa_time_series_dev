# PatchTST

PatchTST ([A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730),
Nie et al., ICLR 2023; [official code](https://github.com/yuqinie98/PatchTST)) forecasts every
variable of a multivariate series on its own (channel independence). Each series is split into
overlapping patches of `patch_len` steps taken every `stride` steps; a patch is one token of a
Transformer encoder, which keeps the number of tokens small and gives each token local semantics.
A linear head maps the flattened encoder output to the horizon. All series share the encoder
weights; the head is shared too, or one per variable (`individual`).

## Encoder, predictor and decoder

- **Encoder**: replication padding of `stride` steps at the end of the series, patching
  (`unfold`), linear patch embedding plus a learned positional encoding, token dropout, and
  `e_layers` post-norm Transformer layers with residual attention (each layer adds the previous
  layer's pre-softmax attention scores, as in the official code). Every (node, channel) series
  is encoded independently.
- **Latent**: `(b, n, c, l, d)`: for each node `n` and channel `c`, the `l` patch tokens of
  width `d = d_model` at the encoder's output. With one data channel this is `(b, n, 1, l, d)`.
  `l = (his_len + stride - patch_len) // stride + 1` (6 for `his_len`/`patch_len`/`stride`
  = 12/4/2, 8 for 96/24/12, 16 for 192/24/12).
- **Predictor**: the shared `LatentPredictor` mixes the `l` patches of each series (the
  second-to-last axis), then the `d` channels of each patch. Patches are the time axis of the
  latent, so mixing them is how the latent of the input window can be moved forward to the
  latent of the following window. Series are never mixed, which keeps channel independence.
- **Decoder**: the official flatten head. Each series' tokens are flattened width-major
  (`(d l)`, the order of `nn.Flatten` on `(d, l)`) and mapped linearly to `pred_len` steps, by one
  weight shared by all nodes or, with `individual`, one weight per node. It reads only the latent.

## Design choices for JEPA

- **LayerNorm in the encoder.** The official encoder normalizes with BatchNorm over the
  `b * n * c` tokens of a batch. JEPA compares the online encoder (train mode, batch statistics)
  with the EMA target encoder (eval mode, running statistics), and micro-batches can be tiny with
  only a few variables (7 for ETT), so BatchNorm makes the two encoders compute different
  functions and the train/eval gap large. LayerNorm over `d_model` behaves the same in both modes
  and for any batch size.
- **Time features are not used** (`custom_time_features = True`). A token is a patch of a single
  series; the official model has no time input and channel independence leaves no natural place
  for one. `x_mark` / `y_mark` are ignored.
- **The target window is encoded like the input window**: same padding, patching and encoder.
  Since `his_len == pred_len` for every task, both windows give the same number of patches.
- **No instance normalization inside the model.** The official model applies RevIN around the
  network; here the trainer z-scores every input window over time (`train.instance_norm`) and
  normalizes the following window with the same statistics, in supervised training and in JEPA
  alike. A second normalization inside the model would be redundant, and in JEPA it would
  normalize the target window with its own statistics, removing exactly the change of level
  and scale the predictor has to anticipate.

## Differences from the reference implementation

- **LayerNorm instead of BatchNorm** in both sublayers of every encoder layer (see above).
- **No RevIN module** inside the model; the trainer's per-window normalization replaces it.
  RevIN's optional affine parameters are absent (off by default in the official code,
  `affine=0`).
- **No series-decomposition variant** (the official `decomposition=True` option with separate
  trend and residual backbones); the default single backbone is implemented.
- **Padding at the end of the series**, as in the official code (`ReplicationPad1d((0, stride))`).
  `unfold` drops the steps that do not complete a patch at the end of the padded series; with
  end padding these are padded copies, while padding at the front would drop the newest
  observed steps whenever `(his_len - patch_len) % stride != 0` (e.g. 128 / 24 / 12 loses the
  last 8 steps).
- The per-node heads are stored as one `(n, pred_len, d * l)` weight applied with a batched
  `einsum`, instead of a Python loop over `n` `nn.Linear` modules; the function and the
  initialization (uniform in `±1/sqrt(d * l)`, as `nn.Linear`) are the same.
- Head dropout (`head_dropout`, 0 by default in the official code) is omitted.

## Hyperparameters

`configs/model/patchtst.yaml`, plus `configs/model_task/patchtst_<task>.yaml` for the patching.

| Key | Meaning | Default |
|---|---|---|
| `d_model` | token width | 128 |
| `n_heads` | attention heads (`d_model` must be divisible) | 8 |
| `d_ff` | hidden width of the feed-forward block | 256 |
| `e_layers` | encoder layers | 2 |
| `dropout` | token, attention-output, feed-forward and residual dropout | 0.05 |
| `attn_dropout` | dropout on the attention weights | 0.0 |
| `individual` | one head per node instead of a shared head | false |
| `patch_len` | steps per patch | 4 (short), 24 (long, long192) |
| `stride` | steps between patches, also the end padding | 2 (short), 12 (long, long192) |

The head is shared by default. One head per node (`individual: true`) has
`n * pred_len * d_model * l` weights: with `d_model = 128` about 87M for PEMS07 (883 nodes) on
the long task, against about 0.1M for the shared head and a few hundred thousand for the encoder.
Its step time stays close to the shared head's (the per-node heads are one batched product), but
its parameters and optimizer state dominate memory and checkpoints on large graphs, and a
per-node head would make the forecast depend on node identity rather than on the encoder.

## Verification

- `tests/test_models.py -k patchtst`: the shared contract tests (shapes, gradients, identity
  predictor at initialization, train/eval consistency, time-feature policy).
- `tests/models/test_patchtst.py`:
  - the untrained model matches a re-implementation of the official forward (its own
    `ReplicationPad1d`, `unfold`, permute / reshape, residual multi-head attention, per-variable
    `nn.Flatten` + `Linear` heads), with BatchNorm replaced by the same LayerNorm modules, for
    12/4/2, 96/24/12 (shared and individual heads) and 128/24/12;
  - every input step, including the newest, influences the latent (end padding);
  - a forecast depends only on its own series (channel independence).
- Supervised and JEPA pretraining runs of one epoch on PEMS08 (short) and ETTh1 (long), and
  fine-tuning from the pretrained checkpoint.
