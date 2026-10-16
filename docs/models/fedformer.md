# FEDformer

FEDformer ([FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series
Forecasting](https://arxiv.org/abs/2201.12740), Zhou et al., ICML 2022; official code
[MAZiqing/FEDformer](https://github.com/MAZiqing/FEDformer), also in
[Time-Series-Library](https://github.com/thuml/Time-Series-Library)) keeps Autoformer's
decomposition encoder-decoder (see [autoformer.md](autoformer.md)) and replaces
auto-correlation by frequency enhanced blocks: the sequence is transformed to a Fourier
(`version: fourier`) or Legendre multiwavelet (`version: wavelets`) basis, only a fixed subset
of `modes` frequencies is kept, a learned complex linear map mixes the channels of every kept
mode (self-attention) or attends between the modes of queries and keys (cross-attention), and the
result is transformed back. Keeping a few modes makes the blocks linear in the sequence length
and acts as a low-rank filter of the series. As in the official code, all encoder layers share
one block, and so do the decoder layers (one self- and one cross-attention block).

## Encoder, predictor and decoder

- **Encoder**: the token embedding (circular convolution over time of all `n × c` variates,
  sinusoidal positions and a linear embedding of the time features), `e_layers` decomposition
  encoder layers (frequency enhanced block and feed-forward network, each followed by trend
  removal) and the seasonal LayerNorm. The latent is `(b, t, d)` with `t = his_len`: one
  `d_model`-wide token per input step, the sequence the decoder cross-attends to.
- **Predictor**: the shared `LatentPredictor` with `num_tokens = his_len` and `dim = d_model`; it
  mixes the time-step tokens (the axis the frequency blocks transform) and then the channels.
- **Decoder**: the original decoder, initialized from the input window (seasonal part of the last
  `label_len` steps followed by zeros, their trend followed by the input mean) and the time
  features of these and of the predicted steps; its frequency enhanced cross-attention reads the
  predicted latent `z` in place of the encoder output. With the identity predictor of an
  untrained model this is the original forward.

## Design choices for JEPA

- **Time features** (`custom_time_features`): the decoder is driven by the time features of the
  predicted steps, which the shared time encoding of the input window cannot provide. They are
  embedded linearly (no bias) and added to every token (`timeF` of the reference code), with the
  dataset's number of time features as input width.
- **Target window**: the encoder reads a window and its own time features only; the following
  window is encoded alike (with `y_mark`). The encoder's Fourier modes are drawn for windows of
  `his_len` steps; JEPA requires `his_len == pred_len` (true for all tasks).
- **Fourier modes in the state dict**: with `mode_select: random` every block draws its modes at
  construction. They are buffers, so a checkpoint carries them and a model loaded from it uses
  the modes it was trained with, whatever random state the loading run is in.
- **The encoder must be able to influence the forecast**: the cross-attention is the decoder's
  only path to the encoder, and the reference scales it so that this path is practically closed
  (see the next section and Verification). Since JEPA pretraining only changes the encoder (and
  predictor), the double scaling is removed.
- **Normalization and dropout**: only LayerNorm; dropout only through `nn.Dropout` modules. The
  frequency blocks have no batch-dependent operation, so training and inference compute the
  same function.

## Differences from the reference implementation

- **Output scaling of the Fourier cross-attention.** The reference initializes the complex
  weights uniformly in `[0, 1/d_model²)` and *also* divides the block's output by
  `in_channels · out_channels = d_model²`. With `d_model = 64` (the width of the experiment in
  Verification; the scaling only gets smaller with a wider model) this leaves the encoder an
  influence of ~2e-6 on the forecast at initialization and ~2% after training, with a gradient
  norm ~1e-7 times the decoder's (Verification). The extra division is removed; the weight
  initialization is kept. Test accuracy is unchanged (slightly better), and the encoder then
  accounts for ~16% of the trained forecast.
- **Placement of the kept modes (`FourierBlock`).** The reference writes the output of the
  `i`-th kept mode at frequency index `i` of the output spectrum (its rank) instead of at its
  own frequency, which, with randomly selected modes, shifts every mode to a lower frequency.
  The paper's block maps each selected frequency to itself (pad the selected modes back to their
  positions before the inverse FFT), which is what is implemented. With `mode_select: low` both
  coincide.
- **Layout of the block output.** The reference returns the frequency blocks' output as
  `(b, heads, channels, t)` and the surrounding attention layer reshapes it with `view(b, t, -1)`,
  which reinterprets channel-major memory as time-major and scrambles time and channels. The
  output is returned as `(b, t, heads, channels)`, so that each token receives its own values.
- **Positional embedding**: sinusoidal positions are added as in Time-Series-Library's FEDformer
  (`positional_embedding: true`); the official FEDformer repository has none (set `false`).
- **Heads**: Fourier weights are per head with `n_heads` heads (Time-Series-Library); the official
  repository fixes 8 heads.
- **`label_ratio`**: the decoder re-reads the last `label_ratio × his_len` input steps (0.5, as in
  the paper and the reference scripts, so that the encoder, the part trained by pretraining,
  carries the rest of the history); the decoder's Fourier modes are drawn for its actual length
  `label_len + pred_len`.
- `n × c` variates (`in_channels == out_channels`), window normalization by the trainers, time
  features in `[0, 1)`, bias-free `nn.Linear` instead of kernel-1 `Conv1d` in the feed-forward
  network, and projections only for the attention inputs a block reads (the reference creates
  unused value projections).
- The wavelet version computes the Legendre multiwavelet filters directly in an orthonormal
  basis (Gauss-Legendre quadrature and Gram-Schmidt), which matches the reference's construction
  but stays accurate at order 8; its cross-attention keeps the reference's `1/d_model²` output
  scaling (it has no weights of its own to carry the scale).

## Hyperparameters

| key | meaning | default |
|---|---|---|
| `d_model` | token width (latent width) | 128 |
| `n_heads` | heads of the Fourier blocks | 4 |
| `d_ff` | hidden width of the feed-forward networks | 256 |
| `e_layers` / `d_layers` | encoder / decoder layers | 2 / 2 |
| `moving_avg` | moving-average kernel of the series decomposition | 25 |
| `label_ratio` | fraction of the input window the decoder re-reads | 0.5 |
| `embedding_kernel_size` | circular convolution embedding the variates | 3 |
| `trend_kernel_size` | circular convolution projecting each decoder layer's trend | 3 |
| `positional_embedding` | sinusoidal positions in the embeddings | true |
| `dropout` | dropout rate | 0.1 |
| `activation` | feed-forward activation (`relu`, `gelu`) | relu |
| `version` | `fourier` or `wavelets` | fourier |
| `modes` | Fourier modes kept per block (wavelets: per cross-attention level) | 32 |
| `mode_select` | `random` or `low` (lowest) frequencies, fourier version | random |
| `cross_activation` | on the cross-attention scores: `tanh` or `softmax` | tanh |
| `wavelet_order` | polynomial order k of the Legendre multiwavelets | 8 |
| `wavelet_modes` | Fourier modes of the self-attention wavelet kernels | 16 |
| `wavelet_channels` / `wavelet_cross_channels` | wavelet channels per order (self / cross) | 128 / 64 |
| `wavelet_blocks` | stacked wavelet kernels in self-attention | 1 |
| `wavelet_skip_levels` / `wavelet_cross_skip_levels` | coarsest levels left out (self / cross) | 1 / 0 |

## Verification

- `tests/test_models.py -k fedformer`: the contract tests pass on the short and long tasks. The
  wavelet version was checked by a probe for finite outputs, a gradient for every parameter,
  train/eval consistency and batch independence on both tasks.
- `tests/models/test_fedformer.py`: the untrained model equals `official_forward`
  (`tests/models/tslib_decomposition.py`, the reference forward written with its own tensor
  operations, with the three deviations above marked `DEVIATION`) for both tasks and
  `label_ratio` 1.0 and 0.5; Fourier modes are restored from a checkpoint into a model built
  under another seed; blocks are shared across layers. A probe building the reference's own
  layers (TSLib `FourierCorrelation`, `AutoCorrelationLayer`, `Autoformer_EncDec`, `Embed`), with
  only the placement and layout fixes patched in, matches the model with the reference scaling
  to within 2e-7 relative; without these two patches the outputs differ by ~7e-4 relative.
- Encoder influence (ETTh1, 96 → 96, 10 epochs, batch 32, one seed; influence = mean
  |decode(z of another window) − decode(z)| / mean |decode(z)| on the test set):

  | variant | `label_ratio` | test MSE / MAE | influence at init / trained | grad. norm encoder / decoder at init |
  |---|---|---|---|---|
  | reference scaling | 1.0 | 0.422 / 0.430 | 1.8e-6 / 0.020 | 1.6e-7 / 1.36 |
  | reference scaling | 0.5 | 0.421 / 0.430 | 2.6e-6 / 0.033 | 2.5e-7 / 1.39 |
  | without the extra division (default) | 1.0 | 0.421 / 0.428 | 2.3e-3 / 0.162 | 6.8e-4 / 1.36 |
  | without the extra division | 0.5 | 0.421 / 0.426 | 3.2e-3 / 0.163 | 1.0e-3 / 1.39 |
  | latent replaced by zeros (decoder only) | 1.0 | 0.430 / 0.434 | 0 / 0 | – |
  | latent replaced by zeros (decoder only) | 0.5 | 0.431 / 0.430 | 0 / 0 | – |

  With the reference scaling the trained model behaves almost like the decoder alone, so
  pretrained encoder weights could barely change its forecasts; without the extra division the
  encoder is used eight times more, at equal or better accuracy.
- One-epoch supervised, JEPA pretraining and fine-tuning runs (from the pretrained checkpoint)
  complete on PEMS08 (short) and ETTh1 (long).
