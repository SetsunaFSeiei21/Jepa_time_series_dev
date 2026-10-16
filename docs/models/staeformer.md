# STAEformer

STAEformer ([STAEformer: Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for
Traffic Forecasting](https://arxiv.org/abs/2308.10425), Liu et al., CIKM 2023; official code
[XDZhelheim/STAEformer](https://github.com/XDZhelheim/STAEformer)) shows that a plain
Transformer forecasts traffic well once its input tokens are rich enough. Every (step, node)
token is the concatenation of a linear projection of the observed values, learned
time-of-day and day-of-week embeddings, a learned node embedding and the paper's
spatio-temporal adaptive embedding, a free parameter of shape (input steps, nodes, width).
Vanilla Transformer layers then attend along time for each node, and along nodes for each step;
a linear head maps the result to the forecast.

## Encoder, predictor and decoder

- **Encoder**: the token embedding, the temporal Transformer layers and the spatial Transformer
  layers. Its latent is the final token sequence of every node, `(b, n, t, d)` with
  `t = his_len` and `d` the token width (sum of the embedding widths, 128 by default).
- **Predictor**: the shared `LatentPredictor` with `num_tokens = his_len`, `dim = d`. It mixes the
  `t` tokens of each node (then the `d` channels of each token), keeping nodes apart. Mixing
  time is what turns the latent of one window into that of the next; the encoder has already
  exchanged information between nodes, so the predictor need not.
- **Decoder**: the original output head, reading only `z`. With `use_mixed_proj: true` one linear
  map takes the flattened `(t, d)` history of each node to its `pred_len × out_channels`
  forecast. With `use_mixed_proj: false` a linear map over time (`his_len → pred_len`, shared by
  all channels) is followed by a linear map `d → out_channels` per step.

## Design choices for JEPA

- **Time features** (`custom_time_features`): the temporal embedding tables are a core part of
  the model, so STAEformer does not use the shared time encoding. The time-of-day and
  day-of-week columns of `x_mark` (values `index / size` in `[0, 1)`) are turned back into
  integer indices and look up tables of `time_feature_sizes` rows (steps per day; 7). Day of
  month and day of year are not used, as in the paper. A table is omitted when its width is 0
  or the dataset lacks the feature.
- **Target window**: the encoder reads a window and its own time features only, so the
  following window is encoded exactly like the input window, with `y_mark` as its time
  features.
- **Window length**: the adaptive embedding has one entry per step of a `his_len` window, so the
  encoder only accepts windows of `his_len` steps. JEPA pretraining requires
  `his_len == pred_len` (true for all tasks), and step `i` of the target window then receives
  the same adaptive embedding as step `i` of the input window.
- **Normalization and dropout**: the model only uses LayerNorm. Dropout (0.3) is applied through
  `nn.Dropout` modules after attention and after the feed-forward block; the JEPA trainer
  switches it off in the online encoder (the EMA target encoder runs in eval mode), and
  supervised training keeps it.

## Differences from the reference implementation

- Time-feature indices are obtained by rounding `feature × size` and clamping to `size − 1`
  instead of truncating. On exact `index / size` inputs both give the same index; rounding is
  robust to floating-point error just below an integer. The day of week comes from the same
  `[0, 1)` feature as the time of day instead of a separate integer channel.
- Attention uses `scaled_dot_product_attention` on separated heads, numerically equivalent to
  the official head splitting along the batch axis.
- Hyperparameters differ from the paper's defaults (see below); the architecture does not.

## Hyperparameters

| Key | Meaning | Default |
|---|---|---|
| `input_embedding_dim` | width of the linear projection of the data channels | 32 |
| `tod_embedding_dim` | width of the time-of-day table (0 disables) | 16 |
| `dow_embedding_dim` | width of the day-of-week table (0 disables) | 16 |
| `spatial_embedding_dim` | width of the per-node embedding (0 disables) | 32 |
| `adaptive_embedding_dim` | width of the per-(step, node) adaptive embedding (0 disables) | 32 |
| `feed_forward_dim` | hidden width of the Transformer feed-forward block | 256 |
| `num_heads` | attention heads; must divide the token width | 4 |
| `num_layers` | Transformer layers along time, and as many along nodes | 2 |
| `use_mixed_proj` | output head: one map from the flattened history (true) or time then channel maps (false) | true |
| `dropout` | dropout after attention and feed-forward | 0.3 |

The paper's defaults are 24 / 24 / 24 / 0 / 80 embedding widths, feed-forward 256, 3 layers and
dropout 0.1.

## Verification

- `tests/test_models.py -k staeformer`: the shared contract tests (shapes, gradients, identity
  predictor at initialization, train/eval consistency, time-feature policy).
- `tests/models/test_staeformer.py`: the untrained model is compared with a re-implementation of
  the official forward pass (official transposes, head splitting along the batch axis, floored
  time-of-day index and integer day-of-week channel) that reads the same weights, on the short
  and long tasks and for both output heads; outputs agree to 1e-5. A probe confirmed the test
  fails when the decoder flattens `(t, d)` in the wrong order or a wrong time column is read.
  A second test checks the latent layout `(b, n, t, d)` and that the predictor mixes `t`.
- One-epoch supervised training, JEPA pretraining and fine-tuning from the pretrained
  checkpoint run on PEMS08 (short task).
