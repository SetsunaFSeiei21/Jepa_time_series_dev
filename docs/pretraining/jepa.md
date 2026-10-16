# JEPA pretraining

A joint-embedding predictive architecture (JEPA) learns representations by predicting, in latent
space, the representation of one part of the data from another part. Here the two parts are
consecutive windows of a series: the encoder of a forecasting model is trained so that the latent
of the window that follows the input window can be predicted from the latent of the input window.
After pretraining, the whole model (encoder, predictor and decoder) is trained on the forecasting
loss as usual, starting from the pretrained weights.

Code: [`src/training/jepa.py`](../../src/training/jepa.py),
[`src/training/sigreg.py`](../../src/training/sigreg.py),
[`src/training/pretrainer.py`](../../src/training/pretrainer.py); configuration:
[`configs/pretrain/jepa.yaml`](../../configs/pretrain/jepa.yaml) and
[`configs/train/pretrain.yaml`](../../configs/train/pretrain.yaml).

## The backbone split

Every model is written as an encoder, a latent predictor and a decoder
(see [adding_a_model.md](../adding_a_model.md)); forecasting computes
`decode(predict(encode(x)))`.

- `encode(x)` maps a window of `his_len` steps to its latent.
- `predict(z)` maps the latent of a window to a prediction of the latent of the next window.
  All models use the same predictor, a single MLP-Mixer block
  ([`src/models/layers/predictor.py`](../../src/models/layers/predictor.py)): a residual MLP that
  mixes the latent's tokens (time steps, patches or segments, chosen by each model), followed by
  a residual MLP over the latent channels. Both residual branches start at zero, so the
  predictor is the identity at initialization. A model that has never been pretrained therefore
  computes `decode(encode(x))`, which is the original architecture; supervised training from
  scratch is an unmodified baseline, and pretraining changes only the starting point.
- `decode(z)` maps a predicted latent to the forecast.
- `latent_tokens(z)` flattens a latent into tokens `(b, l, d)` for the loss.

Because the encoder is applied to the target window too, every task uses input and target
windows of the same length (`his_len == pred_len`).

## Training objective

For a batch of windows `x` (input) and `y` (the `pred_len` steps that follow):

1. Both windows are normalized together: dataset z-score, then, when `train.instance_norm` is
   on, a z-score with the mean and standard deviation of the *input* window. The target window
   is thus expressed relative to its own past, exactly as the forecasting decoder sees it, and
   no statistic of `y` reaches the input. The window standard deviation is floored at 1 (the
   dataset standard deviation, `INSTANCE_MIN_STD` in `src/training/trainer.py`): a nearly flat
   input window would otherwise scale its target window to values of order 1e3 and let a few
   windows dominate the alignment loss. Missing entries are set to 0.
2. The online model computes `predicted = latent_tokens(predict(encode(x, x_mark)))`.
3. The target encoder, an exponential moving average (EMA) of the online model that receives no
   gradient and runs in eval mode, computes `target = latent_tokens(encode(y, y_mark))`.
4. The loss is

   ```
   loss = mse(predicted, target) + sigreg_weight * SIGReg(predicted)
   ```

   The alignment term pulls the prediction towards the target latent. On its own it admits a
   trivial solution, a constant latent. SIGReg (sketched isotropic Gaussian regularization,
   from LeJEPA, https://arxiv.org/abs/2511.08544) prevents it: the predicted tokens of the batch
   are projected on `num_slices` random unit directions, and on every direction the Epps-Pulley
   statistic measures how far the projected values are from a standard normal distribution by
   comparing their empirical characteristic function with `exp(-t^2/2)` on a grid of
   `num_points` values of `t` in `[0, t_max]`. The statistic is averaged over directions; it is
   minimal when the embeddings are an isotropic standard Gaussian, which a collapsed latent is
   not. The directions are redrawn at every step from a counter-seeded generator, so runs are
   reproducible.

5. After every optimizer update, each target parameter `w_t` moves towards the online parameter
   `w_o`: `w_t <- m w_t + (1 - m) w_o`, with a momentum `m` that follows a cosine schedule from
   `ema.start` at the first update to `ema.end` at the last. Buffers are copied.

## Design choices

- **The target encoder and the online encoder must compute the same function.** The target runs
  in eval mode. Dropout in the online model would add noise to its prediction that no predictor
  can remove, a floor under the alignment loss; with `disable_online_dropout` the trainer puts
  every `nn.Dropout` of the online model in eval mode during pretraining (supervised training
  keeps the model's dropout). For the same reason models contain no BatchNorm or other layer
  whose output depends on the batch: the contract tests check that every model gives identical
  outputs in train mode without dropout and in eval mode.
- **SIGReg on the prediction only.** The target receives no gradient, so regularizing it would
  only change the reported loss.
- **SIGReg on every latent token, with a strength independent of the token count.** All tokens
  of the batch are samples of the test, e.g. `b x n x t` tokens for a graph model whose latent
  keeps nodes and time steps; their number differs by orders of magnitude between models,
  datasets and micro-batch sizes. SIGReg is therefore the Epps-Pulley *distance* (the weighted
  squared difference of the characteristic functions), which measures the embedding
  distribution and does not depend on the number of samples. The reference implementation
  multiplies it by the number of samples (the usual normalization of a test statistic); with
  that, the same `sigreg_weight` would regularize a model with many tokens far more strongly
  than one with few, and a configuration more strongly when its effective batch is split into
  larger micro-batches. The characteristic function is summed over chunks of samples that are
  recomputed in the backward pass, so the memory of SIGReg does not grow with the number of
  tokens (hundreds of thousands for large graphs with long windows).
- **One predictor design for all models**, with the same expansion factors
  ([`configs/predictor/default.yaml`](../../configs/predictor/default.yaml)), so that differences
  between backbones are not due to differently sized predictors.
- **The pretrained checkpoint is the whole online model** (encoder, predictor and decoder). The
  decoder is not trained by the JEPA loss and keeps its initialization; forecasting training then
  updates all parameters.
- **Optional centering.** A model whose latent tokens carry a large component that is the same
  for every window (e.g. a learned positional embedding per token) can lower the alignment loss by
  reproducing that component rather than the content of the window. With `centering.enabled`, a
  running mean of every target token position over training batches (momentum
  `centering.momentum`, independent of the micro-batch size; not updated during evaluation) is
  subtracted from the predicted and the target tokens before the alignment loss and SIGReg. It
  is switched on per model with `configs/pretrain_model/jepa_<model>.yaml`; currently for
  Crossformer only (see its page).

## Choice of the SIGReg weight

The weight is the same for all models. It was chosen on PEMS08 (12 steps) with one epoch of
pretraining of six models (GWNet, STGCN, STAEformer, Crossformer, iTransformer, PatchTST) and
weights 0, 0.1, 1 and 10, by the effective rank of the predicted latent tokens on validation
windows (the exponential of the entropy of the normalized eigenvalues of their covariance; 128
at most):

| Model | at init | 0 | 0.1 | 1 | 10 |
|---|---|---|---|---|---|
| GWNet | 10.4 | 3.4 | 3.6 | 6.0 | 14.0 |
| STGCN | 13.9 | 6.2 | 6.4 | 7.4 | 28.1 |
| STAEformer | 20.2 | 5.4 | 5.5 | 6.5 | 27.6 |
| Crossformer | 93.0 | 51.1 | 48.2 | 40.5 | 52.8 |
| iTransformer | 10.2 | 3.8 | 4.0 | 4.7 | 20.2 |
| PatchTST | 4.4 | 1.2 | 1.2 | 1.5 | 13.9 |

Up to a weight of 1 the latent concentrates on a few directions (dimensional collapse) within
the first epoch, below its rank at initialization; with 10 every model keeps or raises its rank.
(Crossformer's rank at initialization is inflated by its random positional embeddings.) A larger
weight also raises the alignment loss, so the weight trades off how predictable and how spread
out the latent is; its effect on forecasting after fine-tuning has to be checked on the target
tasks.

## Hyperparameters

`configs/pretrain/jepa.yaml`:

| Key | Meaning | Default |
|---|---|---|
| `sigreg_weight` | weight of SIGReg relative to the alignment loss | 10 |
| `sigreg.num_slices` | random projection directions per step | 128 |
| `sigreg.t_max` | Epps-Pulley integration range `[-t_max, t_max]` | 3.0 |
| `sigreg.num_points` | integration points on `[0, t_max]` (odd) | 75 |
| `ema.start`, `ema.end` | target momentum at the first and the last update | 0.999, 1.0 |
| `disable_online_dropout` | switch off dropout in the online model | true |
| `centering.enabled` | subtract the running mean of each token position (per model) | false (Crossformer: true) |
| `centering.momentum` | momentum of that running mean per training batch | 0.99 |

`configs/train/pretrain.yaml` sets the optimization of pretraining (10 epochs, learning rate
1e-4, checkpoints every 2 epochs); all other settings (batch size, gradient accumulation,
clipping, scheduler) are those of supervised training in `configs/train/default.yaml`.

## Running

```bash
python -m src.experiments.pretrain model=gwnet dataset=pems08 task=short hydra.run.dir=runs/pt
python -m src.experiments.main model=gwnet dataset=pems08 task=short \
    train.init_checkpoint=runs/pt/pretrained_model.pt
```

The pretraining run logs, for every epoch, the training and validation losses (total,
alignment, SIGReg) and `latent_std`, the mean standard deviation of the predicted latent
channels, in `history.jsonl`; a `latent_std` falling towards 0 signals a collapsing latent.

## Verification

- `tests/test_models.py` checks for every model that the predicted latent of the input window
  has the shape of the target window's latent, that the predictor is the identity at
  initialization, and that train mode without dropout equals eval mode.
- `tests/test_training.py` checks that SIGReg separates Gaussian from scaled or collapsed
  embeddings and does not depend on the number of samples, the EMA schedule, centering, and that
  disabling dropout leaves the rest of the model in train mode.
- The SIGReg implementation was compared with the reference implementation
  (`lejepa.multivariate.SlicingUnivariateTest` with `lejepa.univariate.EppsPulley`): for the same
  projection directions its value equals the reference value divided by the number of samples.
