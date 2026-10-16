# iTransformer

iTransformer ([iTransformer: Inverted Transformers Are Effective for Time Series
Forecasting](https://arxiv.org/abs/2310.06625), Liu et al., ICLR 2024; official code
[thuml/iTransformer](https://github.com/thuml/iTransformer)) applies an unchanged Transformer
encoder to the inverted dimension of a multivariate series. Instead of one token per time step,
every variate (here: every node) becomes one token, embedded by a linear map of its whole input
window. Self-attention then models the correlations between variates, the feed-forward network
learns a representation of each variate's series, and a linear head maps each token to that
variate's forecast.

## Encoder, predictor and decoder

- **Encoder**: the inverted embedding (`Linear(his_len × in_channels → d_model)` applied to each
  node's flattened window, then dropout), `e_layers` post-norm Transformer encoder layers
  (full self-attention across nodes, feed-forward network per token) and the encoder's final
  LayerNorm. The latent is the token sequence `(b, n, d)`: one `d_model`-wide token per node,
  summarizing that node's window in the context of all other nodes.
- **Predictor**: the shared `LatentPredictor` with `num_tokens = n` (the number of nodes) and
  `dim = d_model`. The latent has no time axis left (each token already spans the whole
  window), so the predictor mixes the node tokens, the same axis the encoder's attention mixes,
  and then the channels of each token.
- **Decoder**: the original projection head, `Linear(d_model → pred_len × out_channels)` applied
  to each node token. It reads only `z`.

## Design choices for JEPA

- **Time features**: the model uses the shared time encoding, appended by `add_time` to every
  node's channels. A node's token is therefore embedded from its data series together with the
  time-encoding series of the window; since the time channels are identical for all nodes, they
  contribute the same learned vector to every token. The official code instead appends each
  time feature as one more token (embedded by the same linear map as the variates). That design
  is not adopted because (a) it would make iTransformer an exception to the time encoding shared
  by all models, while nothing in the architecture requires it (the official experiments on
  PEMS and Solar run without time features at all), (b) the extra tokens would enter the JEPA
  latent, or would have to be dropped from it, although the target window's time features are
  known in advance, and (c) in supervised training the choice matters little and neither option
  wins everywhere (table under Verification).
- **Latent**: only node tokens exist, so every token of the latent is compared by the JEPA loss.
- **Target window**: the encoder reads one window and its own time features, so the following
  window is encoded exactly like the input window, with `y_mark` as its time features. The
  embedding is defined for windows of `his_len` steps; JEPA pretraining requires
  `his_len == pred_len` (true for all tasks).
- **Normalization and dropout**: only LayerNorm is used. Dropout is applied through `nn.Dropout`
  modules (after the embedding, on the attention weights, inside and after the feed-forward
  network), so the JEPA trainer can switch it off in the online encoder.

## Differences from the reference implementation

- Time features are part of each node token instead of separate tokens (see above). Without
  time features (`time_encoding=none`) the model is exactly the official one.
- Several data channels per node (`c > 1`) are flattened into a single node token
  (`his_len × c` inputs) and the head predicts `pred_len × out_channels` values per node; the
  official code only has one channel per variate.
- The official model z-scores each input window itself (`use_norm`); here the trainers do it
  (`train.instance_norm`), so the model starts from the normalized window.
- The feed-forward network uses `nn.Linear` instead of kernel-1 `Conv1d`, which is the same map.

## Hyperparameters

| Key | Meaning | Default |
|---|---|---|
| `d_model` | width of the node tokens and of the latent | 128 |
| `n_heads` | attention heads; must divide `d_model` | 4 |
| `d_ff` | hidden width of the feed-forward networks | 256 |
| `e_layers` | encoder layers | 4 |
| `dropout` | dropout rate of embedding, attention weights and feed-forward network | 0.1 |
| `activation` | feed-forward activation, `relu` or `gelu` | relu |

## Verification

- `tests/test_models.py -k itransformer`: the shared contract tests (shapes, gradients, identity
  predictor at initialization, train/eval consistency, time-feature policy).
- `tests/models/test_itransformer.py`: with `time_encoding=none`, the untrained model (short and
  long task) equals a re-implementation of the official `Model.forecast` with `x_mark_enc=None`
  (official permutes and views, `torch.einsum` attention, kernel-1 `Conv1d` feed-forward)
  reading our weights, to 1e-5.
- Supervised training, JEPA pretraining and fine-tuning from the pretrained checkpoint run on
  PEMS08 (short) and ETTh1 (long).
- Time-feature options, supervised, test MAE averaged over 2 seeds (ETT: 96 → 96, 20 epochs,
  normalized MAE; PEMS08: 12 → 12, 10 epochs, raw MAE):

  | Time features | ETTh1 | ETTh2 | PEMS08 |
  |---|---|---|---|
  | shared encoding in every node token (this model) | 0.412 | 0.356 | **17.18** |
  | shared encoding channels as extra tokens | 0.402 | 0.351 | 17.47 |
  | raw features as extra tokens (official) | 0.400 | **0.340** | 17.65 |
  | none | **0.394** | 0.347 | 17.81 |

  Time features help on PEMS08, where the chosen option is best; on ETT they do not help with
  any option, and the differences are of the order of the seed-to-seed spread (up to 0.04 on
  ETTh2).
