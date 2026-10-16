# Autoformer

Autoformer ([Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series
Forecasting](https://arxiv.org/abs/2106.13008), Wu et al., NeurIPS 2021; official code
[thuml/Autoformer](https://github.com/thuml/Autoformer), also in
[Time-Series-Library](https://github.com/thuml/Time-Series-Library)) is an encoder-decoder
Transformer with two ideas. Series decomposition is a layer inside the network: after every
sub-block, a moving average removes the trend, so attention works on the seasonal part and the
decoder accumulates the trends it removes. Auto-correlation replaces self-attention: it finds,
via FFT, the time delays at which queries and keys are most correlated and aggregates the values
rolled by these delays, so the series is connected period by period instead of point by point.
All variates of one time step form one token.

The encoder-decoder is `DecompositionTransformer` in
`src/models/timeseries/layers/decomposition_transformer.py`, shared with FEDformer.

## Encoder, predictor and decoder

- **Encoder**: the token embedding (circular convolution over time of all `n × c` variates plus
  a linear embedding of the time features), `e_layers` decomposition encoder layers
  (auto-correlation and feed-forward network, each followed by trend removal) and the seasonal
  LayerNorm. The latent is `(b, t, d)` with `t = his_len`: one `d_model`-wide token per input
  step, the seasonal representation of the window that the original decoder cross-attends to.
- **Predictor**: the shared `LatentPredictor` with `num_tokens = his_len` and `dim = d_model`.
  Tokens are time steps, so the predictor mixes time, which is what turns the latent of one
  window into that of the following window (the encoder's auto-correlation also mixes time).
- **Decoder**: the original decoder. It decomposes the input window `x`, starts from the
  seasonal part of the last `label_len` input steps followed by zeros and from their trend
  followed by the input mean, embeds the seasonal start with the time features of these steps
  and of the predicted steps (`x_mark`, `y_mark`), and runs `d_layers` decomposition decoder
  layers (self auto-correlation, auto-correlation with the latent, feed-forward; the removed
  trends are projected to the variates and accumulated). It cross-attends to the predicted
  latent `z` where the original attends to the encoder output; with the identity predictor
  of an untrained model, this is exactly the original forward.

## Design choices for JEPA

- **Time features** (`custom_time_features`): the decoder is driven by the time features of the
  predicted steps, which the shared time encoding of the input window cannot provide. Time
  features are embedded linearly (no bias) and added to every token in both embeddings, the
  `timeF` embedding of the reference code; its input width is the dataset's number of time
  features (`num_time_features`).
- **Target window**: the encoder reads a window and its own time features only, so the
  following window is encoded exactly like the input window (with `y_mark`). The latent has
  `his_len` tokens; JEPA requires `his_len == pred_len` (true for all tasks).
- **Auto-correlation delays per sample in training and inference**: the reference code picks
  one set of delays for the whole batch during training (from the batch-mean correlation, a
  speed-up) and per-sample delays at inference. In training mode a sample's output then depends
  on the other samples of its micro-batch, and the training-mode function differs from the
  inference-mode one. JEPA compares the online encoder (training mode) with the EMA target
  (inference mode), and micro-batches vary in size, so the model must compute the same
  per-sample function in both modes. The per-sample selection is the one the paper defines
  (delays from each series' own auto-correlation) and the one the reference uses for its
  reported test results, so it is used in both modes.
- **Normalization and dropout**: only LayerNorm (the seasonal LayerNorm subtracts the mean over
  time after it). Dropout is applied through `nn.Dropout` modules (embedding, after each
  attention, inside the feed-forward network), so the JEPA trainer can switch it off.

## Differences from the reference implementation

- Auto-correlation delays are chosen per sample in training too (see above).
- The decoder re-reads the last `label_len = label_ratio × his_len` input steps; the default is
  half the window (`label_ratio: 0.5`), as in the paper and the reference scripts. The more of
  the history the decoder reads itself, the less the forecast depends on the encoder, which is
  the part that pretraining trains; re-reading the whole window (`label_ratio: 1.0`) gave a
  lower error in a single-seed ETTh1 test (see Verification).
- `n × c` variates: each node-channel series is one variate; the trend is forecast in the input
  space, so `in_channels == out_channels`.
- The trainers normalize each window (`train.instance_norm`); time features lie in `[0, 1)`
  instead of the reference's `[-0.5, 0.5]`.
- Feed-forward layers use bias-free `nn.Linear` instead of kernel-1 bias-free `Conv1d` (the same
  map); the unused attention dropout module of the reference is absent.

## Hyperparameters

| key | meaning | default |
|---|---|---|
| `d_model` | token width (latent width) | 128 |
| `n_heads` | heads of auto-correlation | 4 |
| `d_ff` | hidden width of the feed-forward networks | 256 |
| `e_layers` / `d_layers` | encoder / decoder layers | 2 / 2 |
| `factor` | auto-correlation keeps `int(factor · ln(length))` delays | 3 |
| `moving_avg` | moving-average kernel of the series decomposition | 25 |
| `label_ratio` | fraction of the input window the decoder re-reads | 0.5 |
| `embedding_kernel_size` | circular convolution embedding the variates | 3 |
| `trend_kernel_size` | circular convolution projecting each decoder layer's trend | 3 |
| `dropout` | dropout rate | 0.1 |
| `activation` | feed-forward activation (`relu`, `gelu`) | relu |

## Verification

- `tests/test_models.py -k autoformer`: the contract tests (shapes, gradients, JEPA latents,
  train/eval consistency, time-feature policy) pass on the short and long tasks.
- `tests/models/test_autoformer.py`: the untrained model equals `official_forward`
  (`tests/models/tslib_decomposition.py`), a re-implementation of the reference forward with its
  own tensor operations and inference-phase auto-correlation, reading our weights, for both
  tasks and `label_ratio` 1.0 and 0.5; in training mode each sample's forecast equals the one
  computed for the sample alone. In addition, a probe that builds the reference's own layers
  (TSLib `Autoformer_EncDec`, `AutoCorrelation`, `Embed`) and copies our weights into them
  agrees to within 4e-7 relative (eval mode, his_len 12 and 96).
- How much the forecast depends on the encoder (ETTh1, 96 → 96, 10 epochs, batch 32, one seed;
  influence = mean |decode(z of another window) − decode(z)| / mean |decode(z)| on the test
  set): 0.50 at initialization and 0.38 after training; test MSE / MAE 0.421 / 0.418, against
  0.423 / 0.430 with the latent replaced by zeros (decoder only) and 0.475 / 0.448 with
  `label_ratio: 0.5`.
- One-epoch supervised, JEPA pretraining and fine-tuning runs (from the pretrained checkpoint)
  complete on PEMS08 (short) and ETTh1 (long).
