# STGCN

STGCN ([Yu et al., *Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for
Traffic Forecasting*, IJCAI 2018](https://arxiv.org/abs/1709.04875),
[reference PyTorch code](https://github.com/hazdzz/STGCN)) is a fully convolutional
spatio-temporal forecaster. It stacks ST-Conv blocks with a "sandwich" structure: a gated (GLU)
temporal convolution, a Chebyshev graph convolution on the scaled Laplacian of the sensor graph
followed by ReLU, a second gated temporal convolution, LayerNorm over (nodes, channels) and
dropout. All temporal convolutions are unpadded, so each block shortens the window by
`2 (Kt - 1)` steps. An output block convolves over all remaining steps, normalizes and applies a
two-layer MLP per node. This is the Chebyshev variant (`STGCNChebGraphConv`) of the reference
code.

Implementation: [`src/models/spatiotemporal/stgcn.py`](../../src/models/spatiotemporal/stgcn.py).

## Encoder, predictor and decoder

- **Encoder** (`encode`): the shared time encoding is appended to every node and the window
  runs through the `num_blocks` ST-Conv blocks. The latent is the output of the last block,
  arranged as `(b, n, Ko, d)` with `Ko = his_len - 2 num_blocks (Kt - 1)` remaining steps
  (4, 88 and 184 for windows of 12, 96 and 192 steps with the defaults) and
  `d = block_channels[-1]`: one vector per node and per remaining step.
- **Predictor**: the shared `LatentPredictor` with `Ko` tokens, mixing the remaining steps of
  each node and the `d` channels. Nodes are kept apart: in the original model nodes interact
  only through the graph convolutions inside the blocks, and the output block treats each node
  separately (its LayerNorm aside). Mixing the time axis is what turns the latent of the input
  window into that of the following window, which is also what the output block's full-length
  temporal convolution does.
- **Decoder** (`decode`): reads only the predicted latent and applies the output block: a gated
  temporal convolution with kernel `Ko` (collapsing the time axis), LayerNorm over (nodes,
  channels), `fc_1`, ReLU, `fc_2`, which emits the `pred_len x out_channels` values of each node.

## Design choices for JEPA

- **Same encoder for input and target windows.** The output block's kernel length is `Ko`,
  which is tied to `his_len`. Every task has `his_len == pred_len`, so the target window is
  encoded by the same blocks and both latents have shape `(b, n, Ko, d)`. The output block is
  built for any `Ko >= 1`; the model asserts that the window is long enough.
- **The latent is taken before the output block.** The output block is a per-node readout that
  collapses time, so it forms the decoder, and the latent keeps the time axis for the predictor.
- **Time features** go through the shared time encoding (`add_time`), appended to the input
  channels; the model has no time handling of its own.
- **Normalization and dropout.** The original model already uses LayerNorm over (nodes,
  channels) of each step and no BatchNorm, so it computes the same function in train and eval
  mode and for any batch size. The dropout at the end of each ST-Conv block is an `nn.Dropout`
  module, which the JEPA trainer switches off in the online encoder.
- **Graph per sample.** The scaled Laplacian arrives as the single support `(b, 1, n, n)` and
  the Chebyshev recursion `T_k = 2 L T_{k-1} - T_{k-2}` is applied per sample.

## Differences from the reference implementation

- **Channel alignment without unused parameters.** The residual `Align` of the reference code
  creates a 1x1 convolution in every temporal and graph layer but applies it only when the layer
  shrinks the width (it zero-pads when the width grows and passes the input through when it is
  unchanged). Here the convolution exists only where it is applied, so the forward is identical
  and every parameter is trained.
- **Multi-step, multi-channel output.** The reference code predicts one horizon step; `fc_2`
  here emits `pred_len x out_channels` values per node (direct multi-step forecasting).
- **Time features** are appended as input channels (see above); the reference model reads the
  traffic channels only.
- **Output block.** The reference output block constructs a dropout module that its forward
  never applies; it is omitted here. The fully connected fallback the reference code uses when
  `Ko == 0` is not provided (it does not occur with `his_len >= 2 num_blocks (Kt - 1) + 1`).
- **Graph.** The support is the shared `ScaledLaplacian` transform with `self_loops: false`:
  `2 L / lambda_max - I`, `L = I - D^-1/2 A D^-1/2` of the symmetrized (elementwise max)
  weighted adjacency, with the exact `lambda_max`. This is the reference preprocessing
  (`sym_norm_lap` followed by `calc_chebynet_gso`) and the paper's.

## Hyperparameters

| Key | Meaning | Default |
|---|---|---|
| `graph_transform` | `ScaledLaplacian`, no self-loops | |
| `temporal_kernel_size` | `Kt`, steps of each unpadded temporal convolution | 3 |
| `chebyshev_order` | `Ks`, Chebyshev terms `T_0 .. T_{Ks-1}` | 3 |
| `num_blocks` | ST-Conv blocks | 2 |
| `block_channels` | widths of temporal conv, graph conv, temporal conv in each block | [128, 32, 128] |
| `head_channels` | output block: temporal conv width, hidden MLP width | [128, 128] |
| `dropout` | dropout at the end of every ST-Conv block | 0.1 |

## Verification

- `tests/test_models.py -k stgcn`: the shared contract tests (shapes, gradients for every
  parameter, latent shapes and identity predictor at initialization, train/eval consistency,
  per-sample graphs) pass on the short and long tasks.
- `tests/models/test_stgcn.py`: for windows of 12, 96 and 192 steps, the untrained model equals,
  in float64, a re-implementation of the reference forward (its permutes, zero-channel `Align`,
  GLU slicing and fixed-graph Chebyshev einsums) reading the same weights, with two data
  channels and the shared time encoding so that every alignment branch (pad, 1x1 convolution,
  identity) is exercised. The test also checks that the support produced by the yaml's graph
  transform equals the reference graph preprocessing on a random weighted directed graph.
- One-epoch supervised training, JEPA pretraining and fine-tuning from the pretrained
  checkpoint run on PEMS08 (12 steps); supervised training also runs on PEMS08 with 96 steps.
