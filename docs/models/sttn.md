# STTN

STTN ([Spatial-Temporal Transformer Networks for Traffic Flow Forecasting](https://arxiv.org/abs/2001.02908),
Xu et al., 2020) stacks spatio-temporal blocks, each made of a *spatial transformer* followed by
a *temporal transformer*. The spatial transformer applies self-attention across the nodes of
every time step, which learns dynamic, directed dependencies, and adds a graph diffusion
convolution over the fixed road graph. The temporal transformer applies self-attention across
the time steps of every node, capturing long-range temporal dependencies. Both use learned
positional embeddings (of nodes and of time steps). This implementation follows the PyTorch
STTN of [LargeST](https://github.com/liuxu77/LargeST) (`src/models/sttn.py`), the version
used by common benchmarks; the [official code](https://github.com/xumingxingsjtu/STTN) is
TensorFlow and differs in several details (see below).

## Encoder, predictor and decoder

- **Encoder** (`encode`): the shared time encoding is appended to every node (`add_time`), a
  linear layer projects the channels to `hidden_channels`, and `blocks` STTN blocks follow.
  Each block adds a node embedding, runs `layers_per_block` spatial layers (attention over
  nodes + pre-norm feed-forward, plus a diffusion convolution of the layer input over the
  forward and backward random-walk supports), adds a time-step embedding, runs
  `layers_per_block` causal temporal layers (attention over past steps + pre-norm
  feed-forward), applies LayerNorm and adds the block input. The latent is the output of the
  last block, arranged as `(b, n, t, d)` with `t = his_len` and `d = hidden_channels`: one
  token per node and time step.
- **Predictor**: a `LatentPredictor` with `num_tokens = his_len` and `dim = hidden_channels`.
  It mixes the time steps of each node (second-to-last axis) and the channels, keeping nodes
  apart. Time is the axis along which the input window and the following window differ, so
  the predictor maps the per-step representations of the past window to those of the next
  one; spatial mixing is already done inside the encoder.
- **Decoder** (`decode`): reads the last time step of each node, `z[:, :, -1]`, and applies
  `Linear(hidden_channels, end_channels) -> ReLU -> Linear(end_channels, pred_len *
  out_channels)`, giving all horizons at once. Because temporal attention is causal, the last
  step is the only token that has attended to the whole window, so it is the natural summary
  of the window.

## Design choices for JEPA

- **LayerNorm instead of BatchNorm.** Each block normalizes its output before the residual.
  BatchNorm statistics would differ between the online encoder (train mode, micro-batches of
  1-8 windows) and the EMA target encoder (eval mode, running statistics); LayerNorm over the
  `hidden_channels` of every token behaves identically in both modes and for any batch size.
- **Dropout only as `nn.Dropout` modules** (attention output, feed-forward, diffusion
  convolution), so the JEPA trainer can switch it off in the online encoder.
- **Time features** use the shared encoding appended to the input channels, exactly where the
  reference implementation concatenates its time-of-day / day-of-week channels before the
  start projection. STTN needs no custom handling.
- **`his_len == pred_len`.** The time-step embeddings and the predictor have `his_len`
  positions, so the target window (same length) is encoded by the same encoder.
- **Causal encoder.** The token of step `s` depends only on steps `<= s` of the window (spatial
  layers act per step, temporal attention is causal, LayerNorm is per token). The latent of
  the target window therefore holds, at each position, a summary of the target window up to
  that step, and the decoder's last token summarises the full predicted window.

## Differences from the reference implementation

Relative to the LargeST PyTorch code:

- **LayerNorm replaces `BatchNorm2d`** at the end of each block (see above).
- **Softmax is applied once in the causal temporal attention.** Masked (future) scores are set
  to `-inf` and normalized by a single softmax. Implementations that apply the softmax a
  second time map the masked zeros to `exp(0)` and give back about 42 % (12 steps) to 49 %
  (96 steps) of the attention mass to future positions, so the attention is no longer
  causal.
- **Positional embeddings are initialized as in the official code**: truncated normal with
  standard deviation 0.02 (truncated at two standard deviations), BERT's initializer. With a
  unit-variance initialization the embeddings would dominate the projected input (standard
  deviation about 0.5 at initialization). The official code concatenates its embeddings to
  the features; this implementation adds them, as LargeST does.
- The unused attention-weight dropout module of the reference is omitted (dropout acts on the
  attention output projection, as in the reference forward).
- The 1x1 convolutions of the reference are written as `nn.Linear` layers on channel-last
  tensors (same function).

The official TensorFlow code differs from the LargeST design in ways not reproduced here:
single-head post-norm transformers with ReLU feed-forward, a non-causal temporal attention,
concatenated rather than added positional embeddings (the spatial ones initialized from a
graph embedding), a gated sum `relu(attention + input + 0.1 * graph conv)` in the spatial
layer, and a single-step output layer that convolves over all `t` steps. This implementation
keeps the LargeST architecture, which is the multi-horizon version used in benchmarks.

## Hyperparameters

| Key | Meaning | Default |
|---|---|---|
| `graph_transform` | supports given to the diffusion convolution | `DoubleTransition` (k = 2), no added self-loops |
| `hidden_channels` | width of every token, `d` of the latent | 128 |
| `end_channels` | hidden width of the output head | 512 |
| `blocks` | number of STTN blocks | 2 |
| `layers_per_block` | spatial and temporal layers in each block | 1 |
| `heads` | attention heads (must divide `hidden_channels`) | 4 |
| `mlp_expand` | feed-forward width / `hidden_channels` | 2 |
| `diffusion_steps` | powers of each support in the diffusion convolution | 2 |
| `dropout` | dropout of attention outputs, feed-forwards and graph convolution | 0.1 |

## Verification

- `tests/test_models.py -k sttn` (contract tests on the short and long tasks): shapes, a
  gradient for every parameter, identity predictor at initialization, train/eval
  consistency, per-sample graphs, shared time encoding.
- `tests/models/test_sttn.py`: the untrained model equals, to `1e-5`, a line-by-line
  transcription of the LargeST forward in its `(b, c, n, t)` layout with per-support
  diffusion loops (with LayerNorm and a single softmax); perturbing step `s` of the input
  leaves the latent of steps `< s` unchanged.
- One-epoch runs on PEMS08 (short task, GPU): supervised test MAE 18.62; JEPA pretraining
  completes; supervised fine-tuning from the pretrained encoder, test MAE 18.38.
