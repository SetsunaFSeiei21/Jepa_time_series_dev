# ASTGCN

ASTGCN ([Guo et al., *Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic
Flow Forecasting*, AAAI 2019](https://doi.org/10.1609/aaai.v33i01.3301922),
[official code](https://github.com/guoshnBJTU/ASTGCN-r-pytorch)) forecasts traffic on a sensor
graph with a stack of spatial-temporal blocks. Each block first computes a temporal attention
(a `t x t` matrix from a bilinear score of the window) and a spatial attention (an `n x n`
matrix from a bilinear score of the time-attended window), then applies a Chebyshev graph
convolution whose polynomials `T_k` are reweighted elementwise by the spatial attention, a
temporal convolution and a residual 1x1 convolution, followed by ReLU and LayerNorm. A final
convolution maps the remaining time steps and channels of each node to all horizons at once.
This is the single-component variant (recent window only, no daily / weekly components), as in
the official ASTGCN-r code.

Implementation: [`src/models/spatiotemporal/astgcn.py`](../../src/models/spatiotemporal/astgcn.py).

## Encoder, predictor and decoder

- **Encoder** (`encode`): the shared time encoding is appended to every node, and the window
  runs through the `num_blocks` ASTGCN blocks. The latent is the output of the last block,
  arranged as `(b, n, t', d)` with `t' = ceil(his_len / time_stride)` (= `his_len` with the
  default stride 1) and `d = time_channels`: one vector per node and per time step.
- **Predictor**: the shared `LatentPredictor`, mixing the `t'` steps of each node (token axis)
  and the `d` channels. Nodes are kept apart: in the original model nodes interact only inside
  the blocks (spatial attention and graph convolution), and the output convolution is applied
  to each node separately. Mixing time is what maps the latent of the input window to that of
  the following window, which is also what the output convolution does.
- **Decoder** (`decode`): reads only the predicted latent. The original output convolution
  takes the `t'` steps as input channels and the `d` channels as kernel width, i.e. a linear
  map from each node's `(t', d)` latent to its `pred_len x out_channels` values.

## Design choices for JEPA

- **Same encoder for input and target windows.** The attention parameters are tied to the
  window length (`b_e`, `V_e` are `t x t`; `W_1`, `W_2` have a time axis), so `encode` accepts
  only windows of `his_len` steps. Every task has `his_len == pred_len`, so the target window
  is encoded by the same encoder and both latents have the same shape.
- **The latent is taken before the output convolution.** Everything after the last block is a
  per-node linear readout, so it forms the decoder, and the latent keeps the time axis that the
  predictor mixes.
- **Time features** go through the shared time encoding (`add_time`), appended to the input
  channels; the model has no time handling of its own.
- **Normalization and dropout.** The official block already uses LayerNorm over the channels
  of each (node, step), and the model has no dropout and no BatchNorm, so it computes the same
  function in train and eval mode and for any batch size.
- **Initialization that keeps the attentions trainable.** The score of each attention is a
  bilinear form, e.g. `((x^T U_1) U_2)(U_3 x)`, summed over channels and over all nodes (or
  steps), fed to a sigmoid. If its factors are all positive, the score grows with the number of
  nodes and channels and has one sign: the sigmoid saturates (in the second block, on PEMS08,
  nearly every score is saturated and the attention parameters receive an exactly zero
  gradient that never recovers). The factors `U_1, U_2, U_3, W_1, W_2, W_3` are therefore
  drawn from `N(0, 1/fan_in)`, `fan_in` being the axis each sums over, so that each maps
  unit-scale inputs to unit-scale outputs. The attention biases
  `b_e`, `b_s` and the convolution biases start at zero, LayerNorm at its default (weight 1,
  bias 0), and every other matrix (`V_e`, `V_s`, each `Theta_k`, convolution kernels) is
  Xavier-uniform.
- **Scaled attention scores.** Even from a sound initialization, the scores grow during training
  until the sigmoid saturates and the attention turns into a fixed gate that no longer learns
  (from the pretraining loss either). The score of the temporal attention sums over the `n`
  nodes and that of the spatial attention over the `t` steps, so, as dot-product attention
  divides by `sqrt(d)`, they are divided by `sqrt(n)` and `sqrt(t)`. With the initialization
  above the scaled scores start at O(1) whatever the graph size and window length. On PEMS08
  (12 steps, 3 epochs) this lowers the share of saturated scores (`|z| > 6.9`, sigmoid
  derivative below 1e-3) in the second block's temporal attention from 99% to 76% (score standard
  deviation 2531 to 36) and in the first block's from 44% to 14%, at the same accuracy (test MAE
  17.57 vs 17.62); the scores still grow with training, so the second temporal attention remains
  largely gated. The spatial attentions do not saturate in either version.
- **Memory.** The spatial attention does not depend on the time step, so `T_k ⊙ S` is formed
  once per sample, `(b, k, n, n)`, and the graph convolution covers all steps in one
  contraction. A per-step loop would keep one `n x n` product per step and polynomial for the
  backward pass (several GB for 883 nodes and 192 steps).

## Differences from the reference implementation

- **Initialization** (see above). The official code draws every 1-D parameter from U(0, 1),
  including the attention vectors, the LayerNorm weight and bias and the convolution biases,
  and the matrices `U_2`, `W_2`, `b_e`, `b_s` from Xavier-uniform. With these values the
  attention sigmoids saturate from the second block on.
- **Score scaling** (see above): the attention scores are divided by `sqrt(n)` (temporal) and
  `sqrt(t)` (spatial); the paper and the official code do not scale them.
- **Graph.** The Chebyshev polynomials come from the shared `Chebyshev` transform: the scaled
  normalized Laplacian `2 L / lambda_max - I`, `L = I - D^-1/2 A D^-1/2`, of the binarized,
  symmetrized dataset adjacency (no self-loops added), with the exact `lambda_max`. The
  official code uses the scaled combinatorial Laplacian `L = D - A` of the binary adjacency.
  Both have spectrum in `[-1, 1]`; the normalized one does not scale with node degrees.
- **Time features** are appended as input channels (see above); the official model reads the
  traffic channels only.
- **Output channels.** The output convolution emits `pred_len x out_channels` values per node;
  the official code predicts a single channel. With one output channel the two coincide.
- **Lengths with a temporal stride.** The number of steps after the first block is computed as
  `ceil(his_len / time_stride)`, the actual output length of the strided convolutions. The
  official code sizes the later blocks and the output convolution with `his_len // time_stride`,
  which is only correct when the stride divides the window. With the default stride 1 the two
  agree.
- The attention-weighted Chebyshev convolution is computed for all steps at once instead of in
  a loop over steps (same function).

## Hyperparameters

`configs/model/astgcn.yaml`:

| Key | Meaning | Default |
|---|---|---|
| `graph_transform` | Chebyshev polynomials of the scaled Laplacian; `order` = K | `Chebyshev`, order 3, no added self-loops |
| `num_blocks` | number of ASTGCN blocks | 2 |
| `cheb_channels` | output width of the Chebyshev graph convolution | 128 |
| `time_channels` | output width of the temporal convolution: block width and latent width `d` | 128 |
| `time_kernel_size` | temporal convolution width (odd; padded to keep the length) | 3 |
| `time_stride` | temporal stride of the first block | 1 |

The official configuration uses 64 channels for both widths.

## Verification

- `tests/test_models.py -k astgcn`: shapes, gradients for every parameter, predictor identity
  at initialization, latent shapes of input and target windows, train/eval consistency,
  per-sample graphs.
- `tests/models/test_astgcn.py`:
  - with random parameters (predictor kept the identity), the model equals a line-by-line port
    of the official forward (matrix products in the official order, Chebyshev convolution in a
    loop over steps and polynomials, the official output slicing), for time strides 1 and 2;
  - at initialization, fewer than 1% of the sigmoid pre-activations of every attention exceed
    6.9 in absolute value (derivative below 1e-3), for 20 to 300 nodes and 12 or 96 steps, and
    all score factors of the last block receive a gradient.
- On real PEMS08 windows (170 nodes, 12 and 96 steps) at initialization, the official
  initialization puts 94-100% of the scores of the first temporal attention and of both
  second-block attentions beyond 6 in absolute value (score standard deviation in the
  thousands for the second temporal attention, exactly zero gradient for the second block's
  attention vectors). With the initialization above, fewer than 0.5% of the scores exceed 6
  (largest 7.2, standard deviation at most 1.2) and no sigmoid gradient is exactly zero; the
  same holds for random inputs on 20 and 300 nodes.
- One-epoch supervised and JEPA pretraining runs on PEMS08, and a supervised run initialized
  from the pretrained encoder.
