# Adding a model

A model is one class deriving from `src.models.base.BaseModel` plus one yaml in
`configs/model/`. Nothing else needs registering: `build_model` instantiates the yaml's
`_target_` and the tests pick up every yaml automatically. Every model must work both as an
ordinary forecaster and as the backbone of JEPA pretraining, which is why it is written as
three parts: an encoder, a latent predictor and a decoder.

## Where things go

| Path | Content |
|---|---|
| `src/models/spatiotemporal/<name>.py` | models that use a graph or model nodes jointly |
| `src/models/timeseries/<name>.py` | multivariate time-series models |
| `src/models/layers/` | building blocks shared across model families (graph convolution, series decomposition, the latent predictor) |
| `src/models/timeseries/layers/` | blocks shared by the Transformer-style time-series models |
| `configs/model/<name>.yaml` | `_target_`, optional `graph_transform`, every hyperparameter |
| `configs/model_task/<name>_<task>.yaml` | hyperparameters that depend on the task lengths (optional) |
| `docs/models/<name>.md` | how the model is split into encoder / predictor / decoder, and why |

## Interface

```python
class MyModel(BaseModel):
    def __init__(self, hidden_dim: int, ..., **common):
        super().__init__(**common)  # num_nodes, in_channels, out_channels, num_time_features,
                                    # his_len, pred_len, time_encoding, predictor
        ...
        self.predictor = self.make_predictor(num_tokens=..., dim=hidden_dim)

    def encode(self, x, x_mark, supports=None):
        # x:        (b, t, n, c)   normalized window; missing entries are 0
        # x_mark:   (b, t, f)      time features of the window, values in [0, 1)
        # supports: (b, k, n, n) or None
        # returns   the latent of the window, e.g. (b, n, l, d)
        h = self.add_time(x, x_mark)     # (b, t, n, in_channels), shared time encoding
        ...

    def predict(self, z):
        # latent of the window -> latent of the window that follows it (same shape)
        return self.predictor(z)

    def decode(self, z, x, x_mark, y_mark, supports=None):
        # z: predicted latent; returns (b, pred_len, n, out_channels) in the normalized space
        ...
```

`BaseModel.forward` computes `decode(predict(encode(x)))` and is never overridden.

- **The encoder** maps one window to its latent. JEPA pretraining calls it on the input window
  (with the trained model) and on the following window of the same length (with an EMA copy of
  the model), so it must not depend on the window being an input or a target. Both windows have
  `his_len` steps; all tasks have `his_len == pred_len`.
- **The predictor** maps the latent of the input window to a prediction of the latent of the
  following window. Build it with `self.make_predictor(num_tokens, dim)`: a `LatentPredictor`
  (one MLP-Mixer block) whose widths are shared by all models (`configs/predictor/`). It mixes
  the `num_tokens` tokens along the second-to-last axis and the `dim` channels along the last
  axis; any leading axes (batch, nodes) are kept apart. Arrange the latent so that the axis to
  mix is second to last, typically time steps, patches or segments of a node. The predictor is
  the identity at initialization, so an untrained model computes `decode(encode(x))`, which must
  be exactly the original architecture: training from scratch is then a clean baseline.
- **The decoder** maps a predicted latent to the forecast. Most decoders read only `z`;
  decoder-style models (e.g. Autoformer) also read the input window or the future time
  features, which is why they are passed.
- **`latent_tokens(z)`** turns a latent into the tokens `(b, l, d)` the JEPA loss compares. The
  default `rearrange(z, "b ... d -> b (...) d")` fits any single tensor with the width last;
  override it otherwise (e.g. a list of multi-scale latents).
- Shape letters are fixed project-wide: `b` batch, `t` time, `p` predicted step, `n` node (a
  variable for time-series data), `c` channel, `f` time feature, `k` graph support, `l` latent
  token, `d` hidden width.
- The trainers handle normalization (dataset z-score and optional per-window instance norm),
  masking and the losses. A model only maps inputs to outputs; never read targets.
- All constructor arguments except the shared ones come from the yaml. `BaseModel` takes no
  `**kwargs`, so a misspelled yaml key fails loudly.

## Shared widths

All models use the same hidden width, so that comparisons are not confounded by model size and
every pretraining method learns latents of the same width: the main hidden width and the latent
width are `d = 128`, and feed-forward blocks are `2d = 256` wide. Widths that are specific to an
architecture (e.g. a bottleneck or the hidden layer of an output head) keep their usual ratio to
`d` or their reference value; each model page lists them.

## Layers that behave the same in train and eval mode

JEPA compares the latent of the online model (train mode) with that of the EMA target
encoder (eval mode), and micro-batches are as small as the GPU requires (the effective batch
is restored by gradient accumulation). So apart from dropout, a model must compute the same
function in both modes and for any batch size:

- **No BatchNorm**, or any other layer with batch statistics. Use LayerNorm.
- **Dropout only through `nn.Dropout` modules** (never `F.dropout`), so the JEPA trainer can
  switch it off in the online model (the target has none either).

## Time features

All models see time the same way, so that comparisons are not confounded by feature
engineering. `add_time` encodes the time features with the shared encoding selected by the
`time_encoding` config group (`cyclic` by default; `raw`, `embedding`, `none` for ablations; see
[`src/models/time_encoding.py`](../src/models/time_encoding.py)) and appends the result to every
node; `self.in_channels` already counts the appended channels. Never add another time embedding
inside a model.

A model whose design requires its own handling (e.g. a decoder driven by the future time
features `y_mark`, or a channel-independent model that cannot use them) is an exception: it sets
`custom_time_features = True` (`in_channels` stays the data channels and there is no
`time_encoding`), does not call `add_time`, and states why in its docstring. Exceptions are
listed in `CUSTOM_TIME_FEATURES` in [`tests/test_models.py`](../tests/test_models.py).

## Graphs

- A graph model declares `graph_transform` in its yaml, e.g.
  ```yaml
  graph_transform:
    _target_: src.graph.Chebyshev
    order: 3
  ```
  Available transforms are in [`src/graph/transforms.py`](../src/graph/transforms.py). The model
  then receives `num_supports` (= k) in `__init__` and the supports in `encode` / `decode`.
- Supports are **per sample**: `supports[i]` is the graph of sample `i`. A static dataset graph
  arrives expanded to `(b, k, n, n)` (no copy). So every graph product is batched, e.g.
  `einsum(h, supports, "b k c n t, b k n m -> b k c m t")`. Never store the graph in `__init__`.
- Learned (adaptive) adjacencies are ordinary parameters of the model.

## Code rules

- **einops for every layout change.** Use `rearrange` / `repeat` / `reduce` / `einsum` instead
  of `permute`, `transpose`, `view`, `reshape`, `squeeze`, `unsqueeze` or `torch.einsum` with
  opaque subscripts (`expand` and `.mT` stay fine). Write patterns with the project's letters.
- **No magic numbers.** Every width, depth, kernel size, embedding size, order, rate, ... is a
  constructor argument set in the yaml, with a short comment there when the name is not
  self-explanatory. Values derived from shapes (e.g. a receptive field) are computed, not
  configured. Purely numerical constants (an epsilon, a mask fill value) are named module-level
  constants with a comment.
- **Shape comments** on every `forward` / `encode` / `decode` and wherever a tensor changes
  meaning.
- **No dead code.** No unused parameters or modules (the tests fail on parameters without a
  gradient), no commented-out code, no prints, no try/except used for control flow.
- Initialize every parameter explicitly or through a standard module.
- Comments are written for a reader who knows the paper but nothing about this repository's
  history: say what the code does and why, never what it used to do.
- Docstring on the model class with the paper title and link, plus the reference code if any.
- `ruff check` and `ruff format` must pass on the files you touch.

## Testing

```bash
python -m pytest tests/test_models.py -k <name>    # contract tests for your model
python -m src.experiments.main model=<name> dataset=pems08 task=short train.max_epochs=1
python -m src.experiments.pretrain model=<name> dataset=pems08 task=short train.max_epochs=1
```

The contract tests check, on the short and the long task: output shape and finiteness, a
gradient for every parameter, that the predicted latent of the input window has the shape of
the target window's latent and equals the encoder output at initialization, identical
outputs in train and eval mode once dropout is off, that only the agreed exceptions bypass the
shared time encoding, and for graph models that each sample depends on its own graph only.

## Checklist

- [ ] class derives from `BaseModel` and implements `encode`, `predict` (via
      `make_predictor`) and `decode`; output shape as above
- [ ] the untrained model computes exactly the original architecture
- [ ] all hyperparameters in `configs/model/<name>.yaml` (and `model_task/` if task-dependent)
- [ ] no BatchNorm; dropout only as `nn.Dropout` modules
- [ ] graph used only through `supports`, batched per sample
- [ ] time features only through the shared encoding, unless an agreed exception
- [ ] layout changes written with einops; shape comments present
- [ ] no magic numbers, dead code or uninitialized parameters
- [ ] `pytest -k <name>` passes; supervised and pretraining runs work; `ruff` clean
- [ ] `docs/models/<name>.md` written
