# Masked-reconstruction (MAE) pretraining

Masked autoencoding (He et al., *Masked Autoencoders Are Scalable Vision Learners*,
https://arxiv.org/abs/2111.06377) hides part of the input and trains an encoder and a decoder to
reconstruct it. Here the input is a forecasting model's input window: parts of every node's
series are hidden, the model's encoder reads the rest, and a decoder reconstructs the window.
The encoder must infer hidden values from the temporal context of the node, from other nodes and
from the time features, which is the kind of structure a forecaster needs. After pretraining,
the whole model is trained on the forecasting loss, starting from the pretrained weights.

Code: [`src/training/mae.py`](../../src/training/mae.py) (on top of the shared loop in
[`src/training/pretrainer.py`](../../src/training/pretrainer.py)); configuration:
[`configs/pretrain/mae.yaml`](../../configs/pretrain/mae.yaml) and
[`configs/train/pretrain.yaml`](../../configs/train/pretrain.yaml).

## Use of the backbone

Every model is an encoder, a latent predictor and a decoder (see
[adding_a_model.md](../adding_a_model.md)); forecasting computes `decode(predict(encode(x)))`.
MAE uses only these methods, so it works for every backbone without knowing the layout of its
latent:

- **Encoder**: `z = model.encode(x_masked, x_mark)`, the model's own encoder on the masked window.
- **Decoder**: `reconstruction = decoder.decode(z, x_masked, x_mark, x_mark)`. Because every task
  has `his_len == pred_len`, a model's decoder already maps a latent to a window of the input's
  length, `(b, his_len, n, c)`, in the normalized space. The time features passed as those of
  the window to produce are `x_mark`, since the window to reconstruct is the input window.
  Decoder-style models that also read the input window receive the masked one, so nothing
  hidden reaches the decoder either.
- **Predictor**: not used. It maps a window's latent to the latent of the *following* window,
  whereas reconstruction concerns the same window. It stays the identity, as at initialization.

With `separate_decoder: true` (default) the decoder is a copy of the model made before training;
only its `decode` path receives gradients, and it is discarded after pretraining, as the decoder
of a masked autoencoder usually is. The saved checkpoint is then the model with a pretrained
encoder and its decoder and predictor at their initial values, the same situation as after JEPA
pretraining, so the methods differ only in how the encoder was trained. With
`separate_decoder: false` the model's own decoder is trained to reconstruct and saved with it.

## Masking

For each sample and node, the window's `his_len` steps are split into `num_segments` equal
segments, and `round(mask_ratio * num_segments)` of them, chosen uniformly at random, are hidden
(on all channels of that node). Different nodes have different hidden segments, so a hidden value
can be inferred from the node's own visible steps and from other nodes at the same time.

Hidden entries are handled exactly as missing entries: their raw value is set to 0 and they are
added to the batch's missing mask *before* the batch is normalized. Consequently

- they are 0 in the model input (the missing-value convention of all models);
- the per-window instance normalization (`train.instance_norm`) computes its mean and standard
  deviation from the visible entries only;
- a graph computed from the batch would not see them either.

The target is the clean window, normalized with the same statistics as the input (dataset
z-score, then the instance statistics of the visible entries), so input and target live in the
same space.

## Loss

```
loss = criterion(reconstruction[scored], target[scored])
```

`scored` is the set of entries that were hidden **and** are observed in the data: visible entries
are not scored (copying them is trivial and would dilute the objective), and entries missing in
the data have no target. The criterion is the L1 loss by default (`loss` in the yaml, any
`torch.nn` loss can be configured). A batch without any scored entry contributes a zero loss.

Validation reports the same loss on the validation split with masks drawn from a generator
reseeded with `eval_seed` at every evaluation, so the validation loss of different epochs is
computed on identical masks and is directly comparable.

## Design choices

- **Hidden values must not influence the input through normalization.** If the instance
  statistics were computed on the complete window, the normalized window would sum to zero over
  time for every node, so the sum of the hidden values would be minus the sum of the visible
  ones, and the input would reveal the mean of the hidden values exactly (and, through the unit
  variance, their spread). The shortcut matters most for short windows: on PEMS08 (12 steps,
  4 hidden per node) the estimator "every hidden value = the mean implied by that constraint"
  reaches an L1 error of 0.74 against 0.82 for predicting the window mean when the hidden steps
  are scattered, and 0.57 against 0.82 when they form one 3-step span. Normalizing with the
  visible entries only removes this shortcut; `tests/test_pretraining.py` checks that the model
  input is unchanged when the hidden raw values change.
- **Segments rather than independent points.** Hiding isolated steps of a smooth series turns
  reconstruction into interpolation between neighbours. The table gives the L1 error on hidden
  entries (35 % hidden, visible-only statistics, random training windows) of predicting the
  window mean (0) and of linear interpolation between the nearest visible steps of the node:

  | Data, window | independent points: mean / interpolation | 12 segments: mean / interpolation |
  |---|---|---|
  | PEMS08, 12 steps | 1.09 / 1.02 | 1.03 / 0.95 (segments are single steps) |
  | ETTh1, 96 steps | 0.82 / 0.38 | 0.90 / 0.80 (8-step segments) |
  | ETTh1, 192 steps | 0.81 / 0.35 | 0.86 / 0.88 (16-step segments) |
  | ETTm1, 96 steps | 0.85 / 0.26 | 0.97 / 0.50 (8-step segments) |

  With independent points, interpolation removes 54-69 % of the error on the long windows, so
  the encoder could learn little beyond local smoothing. Expressing the masking unit as a number
  of segments per window keeps the difficulty comparable across task lengths: 12 segments are
  single steps on the 12-step short task, where 5-minute traffic windows are already hard to
  interpolate, and spans of 8 or 16 steps on the long tasks. Finely sampled data (ETTm1) remains
  partly interpolable at 12 segments; fewer, longer segments make the task harder there.
- **An exact number of hidden segments per node** rather than an independent coin flip per
  segment, so that every node of every window has the same amount of visible context.
- **Masked entries are 0, like missing ones.** The models have no mask token; 0 is the
  normalized mean and the value every model already receives for missing data, so no model
  needs a change.
- **The model's decoder as reconstruction head** instead of a separate head reading the latent:
  latents differ between models (per node and time step, per patch, per segment, multi-scale
  lists, no node axis at all), and only the model's decoder knows how to map its latent back to
  a window. A head that, e.g., projected the latent's tokens to values and resized them in time
  would treat patches or multi-scale segments as time steps.
- **Dropout stays on** in the model during pretraining, as in supervised training.

## Hyperparameters

`configs/pretrain/mae.yaml`:

| Key | Meaning | Default |
|---|---|---|
| `mask_ratio` | fraction of every node's segments that is hidden (rounded to a whole number of segments) | 0.35 |
| `num_segments` | equal time segments per window, the unit of masking; must divide `his_len` | 12 |
| `separate_decoder` | reconstruct with a discarded copy of the decoder (true) or train and save the model's decoder (false) | true |
| `loss` | reconstruction criterion on the scored entries | `torch.nn.L1Loss` |
| `eval_seed` | seed of the validation masks, reset at every evaluation | 0 |

With 12 segments and a ratio of 0.35, 4 segments (a third of the window) are hidden.
Optimization (epochs, learning rate, batch size, gradient accumulation, clipping, scheduler,
instance normalization) is shared with the other pretraining methods:
`configs/train/pretrain.yaml`, which extends `configs/train/default.yaml`.

## Running

```bash
python -m src.experiments.pretrain pretrain=mae model=stgcn dataset=pems08 task=short \
    hydra.run.dir=runs/mae
python -m src.experiments.main model=stgcn dataset=pems08 task=short \
    train.init_checkpoint=runs/mae/pretrained_model.pt
```

`history.jsonl` of the pretraining run holds the training and validation reconstruction loss of
every epoch.

## Verification

`tests/test_pretraining.py` (a small model and a toy dataset with missing values) checks that

- every node hides exactly `round(mask_ratio * num_segments)` whole segments;
- hidden entries are 0 in the model input, and changing their raw values changes neither the
  model input nor the target elsewhere (no leak through normalization);
- the loss depends on the target only at hidden, observed entries;
- training changes the encoder, leaves the predictor at its initialization and, with
  `separate_decoder: true`, the saved decoder too (with `false` the saved decoder is trained);
  the checkpoint loads into a freshly built model;
- the validation loss is deterministic.

The interpolation and normalization figures above come from reconstruction baselines evaluated
on random training windows of PEMS08, ETTh1 and ETTm1, normalized as in training.
Every backbone in `configs/model/` (PEMS08 short task for the spatio-temporal models, ETTh1 long
task for the time-series models) was pretrained for one epoch with this method and then trained
for one epoch on forecasting from the checkpoint. For STGCN on PEMS08 (3 epochs of pretraining,
3 of forecasting, one seed), the test MAE is 17.37 with `separate_decoder: true` and 17.29 with
`false`, against 17.50 from scratch: within the noise of a single short run, so the default
follows the usual MAE practice of discarding the reconstruction decoder.
