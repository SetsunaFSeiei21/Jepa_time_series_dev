# Documentation

This repository studies self-supervised pretraining, mainly JEPA (joint-embedding predictive
architecture), for forecasting models of spatio-temporal graphs (e.g. traffic sensors) and of
multivariate time series. Each forecasting model is written as an encoder, a latent predictor
and a decoder; pretraining trains the encoder (and predictor) without labels, then the whole
model is trained to forecast.

## Contents

| Document | Content |
|---|---|
| [adding_a_model.md](adding_a_model.md) | the model interface, the rules every model follows, the tests |
| [experiments.md](experiments.md) | data, configuration, running locally and on a Slurm cluster |
| [pretraining/jepa.md](pretraining/jepa.md) | JEPA pretraining: objective, target encoder, SIGReg |
| [pretraining/mae.md](pretraining/mae.md) | masked reconstruction pretraining |
| [pretraining/contrastive.md](pretraining/contrastive.md) | contrastive pretraining |
| [models/](models/) | one page per model: how it is split into encoder / predictor / decoder and why |

Models: spatio-temporal graph models [GWNet](models/gwnet.md), [STGCN](models/stgcn.md),
[ASTGCN](models/astgcn.md), [STTN](models/sttn.md), [STAEformer](models/staeformer.md);
time-series models [Autoformer](models/autoformer.md), [FEDformer](models/fedformer.md),
[Crossformer](models/crossformer.md), [iTransformer](models/itransformer.md),
[PatchTST](models/patchtst.md).

## Code layout

| Path | Content |
|---|---|
| `src/data/` | datasets in the unified on-disk format, sliding-window loaders, scalers |
| `src/graph/` | graph transforms (adjacency to the supports a model consumes) |
| `src/models/base.py` | `BaseModel`: the encoder / predictor / decoder interface |
| `src/models/layers/` | shared blocks, including the latent predictor |
| `src/models/spatiotemporal/`, `src/models/timeseries/` | the models |
| `src/training/` | supervised `Trainer`, the `Pretrainer` loop and the pretraining methods |
| `src/experiments/` | entry points: `main.py` (forecasting), `pretrain.py` (pretraining) |
| `configs/` | hydra configuration; every hyperparameter lives here |
| `tools/` | dataset conversion, GPU batch-size probing |
| `scripts/` | local smoke test and Slurm experiment scripts |
| `tests/` | unit tests and the contract tests every model must pass |

## Key conventions

- Forecasting is `decode(predict(encode(x)))`. The predictor is the identity at initialization,
  so a model trained from scratch is exactly the original architecture, and a pretrained model
  differs only in its starting point.
- Inputs and targets of a task have the same length, because pretraining encodes both windows
  with the same encoder.
- Every model behaves identically in train and eval mode apart from dropout (no BatchNorm), so
  that the JEPA target encoder (eval mode) and the trained encoder compute the same function and
  results do not depend on the micro-batch size.
- All models share the hidden and latent width `d = 128` (feed-forward blocks `2d`).
- All experiments use an effective batch of 8; when a configuration does not fit in GPU memory,
  a smaller micro-batch with gradient accumulation gives the same effective batch.
- Time features are encoded the same way for all models (`configs/time_encoding/`), except for a
  few models whose design requires their own handling (stated in their pages).
