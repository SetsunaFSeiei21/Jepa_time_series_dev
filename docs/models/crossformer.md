# Crossformer

Crossformer ([Zhang & Yan, *Crossformer: Transformer Utilizing Cross-Dimension Dependency for
Multivariate Time Series Forecasting*, ICLR 2023](https://openreview.net/forum?id=vSVLM2j9eie),
[official code](https://github.com/Thinklab-SJTU/Crossformer)) is a Transformer whose tokens
are segments of single variates. The Dimension-Segment-Wise (DSW) embedding cuts each variate
into segments of `seg_len` steps and embeds every segment with a linear map plus a learned
position of its own. Two-Stage Attention (TSA) layers then attend across the segments of each
variate (cross-time stage) and across variates through a few learned router vectors per
segment (cross-dimension stage, linear in the number of variates). The encoder is
hierarchical: between blocks, `win_size` adjacent segments are merged into one, so every block
sees a coarser scale. The decoder starts from learned positions of the forecast segments; its
layer `i` cross-attends to encoder scale `i` and forecasts the horizon, and the forecasts of
all layers are summed.

Implementation: [`src/models/timeseries/crossformer.py`](../../src/models/timeseries/crossformer.py).

## Encoder, predictor and decoder

A variate is one (node, channel) pair, so there are `v = n * c` variates.

- **Encoder** (`encode`): the window is padded at the front to a multiple of `seg_len` (by
  repeating its first value), cut into `l_0 = ceil(his_len / seg_len)` segments per variate,
  embedded (linear map + learned position + LayerNorm), and passed through `e_layers` scale
  blocks. Block 0 has no merging; every later block first merges `win_size` segments
  (`l_i = ceil(l_{i-1} / win_size)`). Each block holds `block_depth` TSA layers.
  The latent is the list of the `e_layers + 1` scales, the embedding and every block output,
  scale `i` of shape `(b, v, l_i, d_model)`: one vector per variate and segment.
- **Predictor**: one shared-width `LatentPredictor` per scale, mixing the `l_i` segments of
  each variate (token axis) and the `d_model` channels; batch and variates are kept apart.
  Segments are the time axis of a scale, and mapping a window's latent to that of the
  following window is a shift in time. Variates interact inside the TSA layers, and the
  decoder reads every variate's scale on its own (its cross-attention runs per variate), so
  the predictor does not mix them. Scales have different segment counts, hence one predictor
  each.
- **Decoder** (`decode`): reads only the predicted scales (plus the input window's mean if
  `baseline`). The learned decoder positions `(v, l_out, d_model)`, `l_out = ceil(pred_len /
  seg_len)`, pass through `e_layers + 1` decoder layers; layer `i` applies TSA over the
  decoder segments, cross-attention to scale `i` of the same variate, an MLP, and a linear map
  of every decoder segment to `seg_len` steps. The layer forecasts are summed and cropped to
  `pred_len`.
- **`latent_tokens`** concatenates the tokens of all scales:
  `(b, v * (l_0 + ... + l_e), d_model)`.

## Design choices for JEPA

- **Same encoder for input and target windows.** The positions and routers are sized by the
  number of input segments, so `encode` accepts windows of `his_len` steps only; every task has
  `his_len == pred_len`, and the target window yields scales of the same shapes.
- **The latent is every scale, not only the last one.** Decoder layer `i` reads scale `i`
  (including scale 0, the embedding before any attention), so all scales must be predicted.
  The JEPA loss compares all tokens with equal weight; the finest scales hold most of them
  (`l = 12, 12, 3, 1` for 96 steps with `seg_len 8`: 43% of the tokens in scale 0, 43% in
  scale 1; `2, 2, 1, 1` for 12 steps with `seg_len 6`).
- **Time features**: none (`custom_time_features = True`). The original model uses no time
  features: the decoder is driven by learned positions alone, and the DSW embedding embeds a
  segment of one variate, so time channels appended to every node would be embedded as extra
  variates rather than as features of the series. The model ignores `x_mark` and `y_mark`.
- **Normalization and dropout.** The model uses LayerNorm only, and every dropout (attention
  weights, residual branches) is an `nn.Dropout` module, so it computes the same function in
  train and eval mode and for any batch size.
- **Attention over a single key.** Coarse scales can have a single segment (e.g. scale 3 for
  96 steps, scales 2 and 3 for 12 steps). Attention over one key has softmax weight 1 whatever
  the queries and keys, so the query and key projections of such an attention (time attention
  of that scale, decoder cross-attention to it, and likewise for a single variate or a single
  router) can never act and would only ever receive a zero gradient. They are not built; the
  layer computes `out(dropout(1) * value(keys))`, which is exactly the original function.
- **Position embeddings dominate the latent at initialization.** The learned positions are
  drawn from `N(0, 1)` per variate and segment, as in the reference code, while the linear
  segment embedding of a z-scored window has a variance of about 0.36. On real windows (PEMS08,
  12 steps; ETTh1, 96 steps), the positions make up 73% of the embedding's variance, and 78-82%
  of the variance of every scale's tokens is input-independent (the same for all windows). The
  input and the target window share these position codes, so part of the JEPA objective can be
  met by input-independent features, and SIGReg does not object since the codes of different
  tokens look like independent Gaussian samples. Without a remedy this shortcut is taken: after
  one epoch of JEPA pretraining the input-independent share of the latent tokens (variance over
  tokens of the token mean over windows, divided by the total token variance) rises from 78% to
  87% on PEMS08 and to 84% on ETTh1.

  The model keeps the reference initialization, so that the untrained model is the original
  Crossformer and supervised training from scratch is the original baseline. Instead, JEPA
  pretraining of Crossformer compares centred tokens
  ([`configs/pretrain_model/jepa_crossformer.yaml`](../../configs/pretrain_model/jepa_crossformer.yaml),
  see [JEPA pretraining](../pretraining/jepa.md)): a running mean of every target token
  position over training windows, i.e. the input-independent part, is subtracted from the
  predicted and the target tokens before the alignment loss and SIGReg. After one epoch the
  input-independent share then falls from 78% to 47% on PEMS08 and to 45% on ETTh1: the encoder
  is trained on what depends on the window.

## Differences from the reference implementation

- **Single-key attentions have no query / key projections** (see above): same function, no
  parameters that could never be trained.
- **Segment merging with fewer segments than padding.** When a scale's segment count is not a
  multiple of `win_size`, the reference code appends its last `missing` segments again. With
  fewer than `missing` segments (e.g. merging a single segment with `win_size 4`, which happens
  for 12-step windows), its slice is too short and the code fails. Here the indices before the
  first segment are clamped to it, which reduces to the reference rule whenever that rule is
  defined.
- **Several channels per node** are treated as separate variates; the reference code has one
  channel per variate.
- **`seg_len` depends on the task**, as in the reference scripts, which choose a segment length
  that divides the input length: 8 by default (96 and 192 steps), 6 for 12-step windows
  (`configs/model_task/crossformer_short.yaml`). Other lengths are padded as in the reference
  code.

## Hyperparameters

`configs/model/crossformer.yaml` (+ `configs/model_task/crossformer_short.yaml`):

| Key | Meaning | Default |
|---|---|---|
| `seg_len` | time steps per segment | 8 (6 for `task=short`) |
| `win_size` | segments merged into one between consecutive encoder scales | 4 |
| `factor` | router vectors per segment in the cross-dimension stage | 10 |
| `d_model` | token width, latent width | 128 |
| `d_ff` | hidden width of the TSA feed-forward blocks | 256 |
| `n_heads` | attention heads | 4 |
| `e_layers` | encoder blocks; the decoder has `e_layers + 1` layers | 3 |
| `block_depth` | TSA layers per encoder block | 1 |
| `dropout` | dropout of attention weights, residual branches and feed-forward outputs | 0.1 |
| `baseline` | add the input window's mean to the forecast | false |

The reference scripts use `d_model 256`, `d_ff 512`, `n_heads 4`, `e_layers 3`, `win_size 2`,
`factor 10`, `dropout 0.2`, `block_depth 1`.

## Verification

- `tests/test_models.py -k crossformer`: shapes, a gradient for every parameter, predictor
  identity at initialization, latent shapes of input and target windows, train/eval
  consistency, on the 12- and 96-step tasks.
- `tests/models/test_crossformer.py`: with random LayerNorm parameters, the untrained model
  equals a port of the official forward (front padding, DSW embedding, TSA, official
  SegMerging, decoder sum, cropping) for 96, 90 (front padding and cropping) and 192 steps
  (merging pads two segments); single-key attentions are evaluated in the port with freshly
  drawn query / key projections and give identical results (max. difference 0). Replacing the
  merge padding by repeating the last segment changes the 192-step output by up to 0.15, so
  the test is sensitive to it. The test also checks the scale sizes and `latent_tokens`.
- Variance of the latent on 64 real training windows (instance-normalized as in training),
  at initialization and after pretraining: numbers above.
- One-epoch supervised and JEPA pretraining runs on PEMS08 (12 steps) and ETTh1 (96 steps),
  and supervised runs initialized from the pretrained models: PEMS08 test MAE 19.52 (scratch)
  and 19.44 (pretrained); ETTh1 test MAE 0.400 and 0.402 (normalized space).
