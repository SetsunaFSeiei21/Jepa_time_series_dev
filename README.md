# Time-Series JEPA

Self-supervised (JEPA) pretraining for spatio-temporal and multivariate time-series forecasting
models. Every model is an encoder, a latent predictor and a decoder; it is pretrained on
unlabeled windows, then trained to forecast.

```bash
python -m src.experiments.pretrain model=gwnet dataset=pems08 task=short hydra.run.dir=runs/pre
python -m src.experiments.main model=gwnet dataset=pems08 task=short \
    train.init_checkpoint=runs/pre/pretrained_model.pt
```

See [docs/README.md](docs/README.md) for the documentation.
