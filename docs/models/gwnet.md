# Graph WaveNet

Graph WaveNet ([Wu et al., *Graph WaveNet for Deep Spatial-Temporal Graph Modeling*, IJCAI
2019](https://arxiv.org/abs/1906.00121), [official code](https://github.com/nnzhan/Graph-WaveNet))
forecasts traffic on a sensor graph with a stack of WaveNet layers. Each layer applies a gated,
dilated causal convolution along time (`tanh(filter) * sigmoid(gate)`) and then a diffusion
graph convolution over the given supports (forward and backward random walks) plus an
adaptive adjacency `softmax(relu(E1 E2))` learned from two node embeddings. Dilations double
within each block, so a few layers cover the whole input window. Every layer emits a skip
output; their sum is passed through a small MLP head that predicts all horizons at once.

Implementation: [`src/models/spatiotemporal/gwnet.py`](../../src/models/spatiotemporal/gwnet.py).

## Encoder, predictor and decoder

- **Encoder** (`encode`): the shared time encoding is appended to every node, the window is
  left-padded with zeros to the receptive field when it is shorter (`1 + (kernel_size - 1) *
  sum(dilations)` = 13 steps with the defaults), projected by `start_conv` and run through the
  WaveNet layers. The latent is the sum of the skip outputs, arranged as
  `(b, n, latent_steps, skip_channels)`: one vector per node and per remaining time step.
  `latent_steps = max(his_len, receptive_field) - receptive_field + 1`, i.e. 1 on the short
  task (12 steps) and 84 / 180 for 96 / 192-step windows. Each remaining step summarizes the
  receptive field that ends at it.
- **Predictor**: the shared `LatentPredictor`, mixing the `latent_steps` steps of each node
  (token axis) and the skip channels. Nodes are kept apart, as in the original model, where
  nodes interact only through the graph convolutions. Mixing time is what maps the latent of
  the input window to that of the following window. With a single step the token MLP has
  nothing to mix and the predictor acts on the channels.
- **Decoder** (`decode`): reads only the predicted latent. ReLU, then (when `latent_steps > 1`)
  an MLP over the steps that reduces them to one, then the original output head
  `end_conv_2(relu(end_conv_1(.)))`, which emits `pred_len * out_channels` values per node.

## Design choices for JEPA

- **LayerNorm instead of BatchNorm.** The normalization after each layer's residual addition
  is a LayerNorm over the residual channels of each (node, step). JEPA compares an online
  encoder in train mode with an EMA target encoder in eval mode, on micro-batches of a few
  windows; BatchNorm would normalize the two with different statistics (per-batch vs running).
- **Time features** go through the shared time encoding (`add_time`); the model has no time
  handling of its own. The official code feeds the time of day as an extra input channel,
  which the shared encoding subsumes.
- **Same encoder for input and target windows.** Both windows have `his_len` steps
  (`his_len == pred_len` on every task), so they have the same number of latent steps and the
  predictor maps between equal shapes.
- **The latent is taken before the output head.** The skip sum is the representation that all
  layers write into; everything after it is a per-node readout, so it forms the decoder.
- **Long windows.** The original model is built for windows no longer than its receptive
  field, where a single step remains. Longer windows leave several steps, which the time MLP
  (`time_mlp_hidden`) reduces to the one step the head expects. The head's ReLU is applied to
  the skip sum before this MLP. Placed after it, it would follow a layer with a single output
  per (node, channel); at initialization that output often has the same sign everywhere
  (it is dominated by the shared bias), and a negative sign zeros the forecast's dependence on
  the input and every gradient below the head. On short windows both placements coincide with
  the original model.
- **Gradient checkpointing** (`gradient_checkpointing: true`) recomputes each layer in the
  backward pass instead of storing its activations, so large graphs with long windows (e.g.
  883 nodes, 192 steps) fit in GPU memory. The RNG state is restored, so outputs and gradients
  are identical, dropout included.
- **Dropout** is the `nn.Dropout` inside the shared `DiffusionConv`, after each graph
  convolution as in the original, so the JEPA trainer can switch it off.

## Differences from the reference implementation

- LayerNorm instead of BatchNorm2d (see above).
- The residual of each layer is the layer input itself, with its gradient. Some
  implementations detach it, which cuts the gradient from deeper layers and the head to
  `start_conv`, to the earlier graph convolutions and to the adaptive adjacency along the
  residual path.
- The last layer has no graph convolution and normalization: the original computes them but
  never uses the result (only the skip output of the last layer is read). The forecast is
  unchanged and no parameter is left without a gradient.
- The adaptive adjacency embeddings are initialized from a standard normal, the
  `--randomadj` setting of the official training command. The official SVD initialization
  from the static graph would require storing the graph in the model, while graphs here are
  passed per sample.
- The 1x1 convolutions of the output head are written as linear layers on the channels-last
  latent (same function and same default initialization).
- The time MLP and the placement of the head's ReLU for windows longer than the receptive
  field (see above); the official code does not handle such windows.
- Hyperparameters: 128 residual and dilation channels (official: 32) and dropout 0.1
  (official: 0.3).

## Hyperparameters

`configs/model/gwnet.yaml`:

| Key | Meaning | Default |
|---|---|---|
| `graph_transform` | supports built from the dataset adjacency | `DoubleTransition` (2 supports), no added self-loops |
| `residual_channels` | width of the residual stream | 128 |
| `dilation_channels` | width of the gated temporal convolution output | 128 |
| `skip_channels` | width of the skip outputs, i.e. of the latent | 128 |
| `end_channels` | hidden width of the output head | 512 |
| `kernel_size` | temporal convolution width | 2 |
| `blocks` | number of blocks | 4 |
| `layers_per_block` | layers per block, dilations 1, 2, 4, ... | 2 |
| `diffusion_steps` | powers of each support in the graph convolution | 2 |
| `dropout` | dropout after each graph convolution | 0.1 |
| `adaptive_adj` | add the learned adjacency `softmax(relu(E1 E2))` | true |
| `adaptive_embedding_dim` | width of `E1` / `E2` | 10 |
| `time_mlp_hidden` | hidden widths of the MLP over the remaining steps (long windows only) | [64, 128, 64] |
| `gradient_checkpointing` | recompute each layer in backward to save memory | false |

## Verification

- `tests/test_models.py -k gwnet`: shapes, gradients for every parameter, predictor identity at
  initialization, latent shapes of input and target windows, train/eval consistency, per-sample
  graphs.
- `tests/models/test_gwnet.py`:
  - the untrained model equals a line-by-line port of the official forward (BatchNorm2d
    replaced by the same channel LayerNorm, time MLP after the head's ReLU for long windows) with
    copied weights, on 12- and 96-step windows, for the output and for the gradients of
    `start_conv` and of the adaptive embeddings;
  - on 96-step windows the forecast varies across nodes and `start_conv` receives a gradient
    for every one of 10 initialization seeds;
  - gradient checkpointing gives identical outputs and gradients in train mode with dropout.
- Placement of the head's ReLU, over 20 initialization seeds: placed after the time MLP, the
  encoder receives no gradient at all for 6 seeds on 96-step windows and 12 seeds on 192-step
  windows. Placed before it, this happens for no seed.
- One training step on 883 nodes with 192-step windows and batch 1 needs 3.2 GiB with gradient
  checkpointing. Without it, the same step ran out of memory at about 12.5 GiB.
- One-epoch supervised and JEPA pretraining runs on PEMS08, and a supervised run initialized
  from the pretrained encoder.
