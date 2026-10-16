# Contrastive pretraining

Contrastive learning trains an encoder so that two randomly perturbed versions ("views") of the
same input have similar representations while the representations of different inputs stay
apart (SimCLR, Chen et al., https://arxiv.org/abs/2002.05709; for time series, e.g. TF-C,
https://arxiv.org/abs/2206.08496, which also perturbs the frequency content). Here the views are
made by removing random frequency components of every node's input window, the encoder is the
forecasting model's encoder, and the objective is the symmetric InfoNCE loss. After pretraining,
the whole model is trained on the forecasting loss, starting from the pretrained weights.

Code: [`src/training/contrastive.py`](../../src/training/contrastive.py) (on top of the shared
loop in [`src/training/pretrainer.py`](../../src/training/pretrainer.py)); configuration:
[`configs/pretrain/contrastive.yaml`](../../configs/pretrain/contrastive.yaml) and
[`configs/train/pretrain.yaml`](../../configs/train/pretrain.yaml).

## Use of the backbone

Every model is an encoder, a latent predictor and a decoder (see
[adding_a_model.md](../adding_a_model.md)). Contrastive pretraining uses only

- `model.encode(view, x_mark)`, the model's encoder, applied to each view, and
- `model.latent_tokens(z)`, which turns any latent into tokens `(b, l, d)`: the same tokens the
  JEPA loss compares (per node and time step, per patch, per segment, per variable, ..., as each
  model defines its latent; multi-scale latents are concatenated by the model's override).

The predictor and the decoder are not used and keep their initialization. The projection head
is a module of the trainer, not of the model, so the saved checkpoint `pretrained_model.pt`
contains exactly the model's parameters and loads into the plain model.

## Method

For a batch of input windows `x` (normalized as for forecasting: dataset z-score, then the
per-window instance z-score when `train.instance_norm` is on; missing entries 0):

1. **Two views.** For each view independently, the series of every sample, node and channel is
   transformed with a real FFT along time, every frequency component is removed with probability
   `freq_mask_ratio` (the zero frequency, i.e. the window mean, is always kept when
   `preserve_dc`), and the series is transformed back. Missing entries are reset to 0. Both
   views keep the window's time features and graph.
2. **Embeddings.** Each view is encoded, flattened to tokens with `latent_tokens`, and mapped by
   the projection head, a linear layer `d -> d` (`d` is read from the model's latent at start-up).
3. **Instances.** With `instances: token`, every token of every window of the batch is an
   instance: the positive of token `i` of view 1 is token `i` of view 2, and every other token
   of the batch is a negative. When the batch has more than `max_instances` tokens, a random
   subset of that size (the same positions in both views) is used. With `instances: window`, the
   tokens of a window are averaged and the windows of the batch are the instances.
4. **Loss.** With `a`, `b` the `(m, d)` embeddings of the two views, L2-normalized,

   ```
   logits = a @ b.T / temperature                     # cosine similarities, (m, m)
   loss   = (cross_entropy(logits, arange(m)) + cross_entropy(logits.T, arange(m))) / 2
   ```

   i.e. each embedding of view 1 must identify its own view-2 counterpart among the `m`
   candidates, and vice versa. The logged `accuracy` is the fraction of view-1 embeddings whose
   most similar view-2 embedding is their positive.

Validation reports the same loss and accuracy on the validation split, in eval mode, with views
(and token subsets) drawn from a generator reseeded with `eval_seed` at every evaluation, so the
values of different epochs are directly comparable.

## Design choices

- **Model-agnostic instances.** Latents have different layouts across models (some have a node
  axis, some only time steps or patches, some a list of scales), so the method defines instances
  on the tokens of `latent_tokens`, which every model provides, instead of on a node or time axis
  that only some models have. Token instances give thousands of negatives per batch for every
  model, including those whose latent has no node axis (with window instances such a model would
  get only `batch_size - 1` negatives). Negatives come from the micro-batch only: gradient
  accumulation does not add negatives.
- **Both views share the normalization** of the clean window (computed once, before the
  augmentation), so they differ only by the removed frequencies, not by their scale or offset.
  With `preserve_dc` both views also keep the window mean.
- **Missing entries stay 0 in the views.** The inverse FFT spreads every frequency over all
  steps, so without the reset a view would contain fabricated values at missing positions,
  where models expect 0.
- **No target window.** Only the input window is used; nothing of the window to forecast enters
  the views.
- **Dropout stays on** in train mode, so the two views of a training step also differ by their
  dropout masks; validation runs without dropout.
- **Time features are identical in the two views** of a window and usually differ between
  windows, so they help to tell a window's tokens from those of other windows. This is a
  property of the task rather than a leak (the time features are part of the input), and the
  values matter more: after 500 updates of STGCN on PEMS08, the validation loss is 6.72 with the
  real inputs and 6.98 when every window gets the same time features. Tokens of other nodes of
  the same window, which share its time features, are among the negatives too. Models whose
  latent carries learned node or time-of-day embeddings (e.g. STAEformer, Graph WaveNet) make
  tokens easy to identify: their validation accuracy exceeds 0.95 after one epoch on PEMS08,
  while the loss (about 3.3-4.0 against log(4096) = 8.3 at chance) keeps decreasing. With
  `instances: window` and a batch of 8 windows the task is nearly trivial (accuracy 1.0 after
  three epochs of STGCN), which is why token instances are the default.
- **A linear projection head**, as the simplest head that decouples the contrastive embedding
  from the latent the decoder reads.

## Hyperparameters

`configs/pretrain/contrastive.yaml`:

| Key | Meaning | Default |
|---|---|---|
| `freq_mask_ratio` | probability of removing each frequency component of each series in a view | 0.2 |
| `preserve_dc` | never remove the zero frequency (the window mean) | true |
| `temperature` | InfoNCE temperature on cosine similarities | 0.2 |
| `instances` | `token`: every latent token is an instance; `window`: the tokens of a window averaged | token |
| `max_instances` | token instances drawn per batch, which bounds the `(m, m)` similarity matrix (4096 instances: 64 MB per matrix in float32) | 4096 |
| `eval_seed` | seed of the validation views and token subsets, reset at every evaluation | 0 |

Optimization (epochs, learning rate, batch size, gradient accumulation, clipping, scheduler,
instance normalization) is shared with the other pretraining methods:
`configs/train/pretrain.yaml`, which extends `configs/train/default.yaml`. The batch size also
sets the number of window instances.

## Running

```bash
python -m src.experiments.pretrain pretrain=contrastive model=stgcn dataset=pems08 task=short \
    hydra.run.dir=runs/contrastive
python -m src.experiments.main model=stgcn dataset=pems08 task=short \
    train.init_checkpoint=runs/contrastive/pretrained_model.pt
```

`history.jsonl` of the pretraining run holds the training and validation loss and accuracy of
every epoch.

## Verification

`tests/test_pretraining.py` (a small model and a toy dataset with missing values) checks that

- with `freq_mask_ratio = 0` the augmentation is the identity, with `preserve_dc` it keeps every
  series' mean, and missing entries stay 0;
- the InfoNCE loss equals `log(m)` for collapsed embeddings, has the closed-form value for
  orthogonal embeddings, and increases when positives are mismatched;
- the two views differ, and validation views are reproducible;
- for both instance definitions, training changes the encoder and the projection head, leaves
  the decoder at its initialization, and saves a checkpoint without the head that loads into a
  freshly built model; the validation loss is deterministic.

Every backbone in `configs/model/` (PEMS08 short task for the spatio-temporal models, ETTh1 long
task for the time-series models) was pretrained for one epoch with this method and then trained
for one epoch on forecasting from the checkpoint. For STGCN on PEMS08 (3 epochs of pretraining,
3 of forecasting, one seed), the test MAE is 17.27 (token instances) and 17.22 (window
instances) against 17.50 from scratch.
