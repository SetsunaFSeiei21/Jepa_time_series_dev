# Running experiments

## Environment

Python 3.12 with `torch`, `hydra-core`, `omegaconf`, `einops`, `numpy`, `pandas` (dataset
conversion), `pyyaml`, `tqdm`, and `pytest` / `ruff` for development.

## Data

Datasets are stored in a unified format, one directory per dataset under `data_root`
(`~/datasets` by default):

| File | Content |
|---|---|
| `data.npy` | `(T, N, C)` float32 series, missing entries filled with 0 |
| `mask.npy` | `(T, N, C)` bool, True where missing |
| `time_features.npy` | `(T, F)` calendar features in `[0, 1)` (time of day, day of week, ...) |
| `graph.npy` | `(N, N)` adjacency, for graph datasets |
| `meta.json` | name, shape, frequency, number of distinct values of every time feature |

`tools/convert_datasets.py` builds this format from the raw sources, e.g. from the STFM layout
(`data.npy`, `mask.npy`, `adj_mtx.npy`, `dataset_config.json` per dataset):

```bash
python tools/convert_datasets.py --stfm-root ~/STFM-datasets --out ~/datasets
```

The experiment protocol of a dataset (split ratios, normalization, the space in which losses
and metrics are computed) is in `configs/dataset/<name>.yaml`.

## Configuration

Everything is configured with hydra; any key can be overridden on the command line.

| Group | Selects |
|---|---|
| `dataset` | dataset and its protocol (`pems04`, `pems08`, `metr-la`, `etth1`, ...) |
| `model` | the model and all its hyperparameters |
| `task` | input / forecast lengths: `short` (12 -> 12), `long` (96 -> 96), `long192` (192 -> 192) |
| `train` | optimization: `default` for forecasting, `pretrain` for pretraining |
| `pretrain` | pretraining method: `jepa`, `mae`, `contrastive` |
| `time_encoding` | shared time-feature encoding: `cyclic` (default), `embedding`, `raw`, `none` |
| `predictor` | widths of the latent predictor shared by all models |
| `pretrain_model` | settings of a pretraining method for one model (`<method>_<model>.yaml`, optional) |

`configs/default.yaml` is the forecasting entry, `configs/pretrain.yaml` the pretraining entry.
Validation and test windows advance by `pred_len` steps (`task.eval_stride`), training windows
by one step (`train.train_stride`).

## Running locally

```bash
python -m src.experiments.main model=gwnet dataset=pems08 task=short            # from scratch
python -m src.experiments.pretrain model=gwnet dataset=pems08 task=short \
    pretrain=jepa hydra.run.dir=runs/gwnet_pre                                   # pretraining
python -m src.experiments.main model=gwnet dataset=pems08 task=short \
    train.init_checkpoint=runs/gwnet_pre/pretrained_model.pt                     # fine-tuning
```

A forecasting run writes `history.jsonl` (per-epoch training loss and validation metrics),
`best_model.pt` (the epoch with the best validation `train.monitor`) and `test_metrics.json`
(test metrics of that epoch). A pretraining run writes `history.jsonl` and
`pretrained_model.pt` (plus `pretrained_epoch_<e>.pt` every `train.save_every` epochs).

Runs use `train.device` (`cuda:0` by default) and stop with an error when it is a GPU and
none is available; pass `train.device=cpu` to train on the CPU.

`bash scripts/local/smoke_test.sh [model ...]` runs all tests and, for every model, one quick
epoch of training from scratch, of every pretraining method and of fine-tuning.

## Batch size

All experiments use an effective batch of 8 windows. Large graphs with long windows do not fit
on a GPU at once; such configurations run with a smaller micro-batch (`train.batch_size`) and
gradient accumulation (`train.accumulation_steps`), `batch_size x accumulation_steps = 8`. No
model has a layer that depends on the batch (no BatchNorm), so the result does not depend on how
the effective batch is split.

`tools/probe_batch_size.py` (run from the repository root with `PYTHONPATH=.`) measures, for
every model / dataset / task, the peak GPU memory of a supervised step and of a JEPA step for
micro-batches 1, 2, 4, 8, and writes the largest one that fits into `configs/batch_sizes.yaml`.
Run on the target GPU it measures directly; on a smaller GPU, pass `--memory-gb` and it
extrapolates linearly from the sizes that fit.
`tools/batch_overrides.py <model> <dataset> <task>` prints the matching overrides.

## Running on a Slurm cluster

1. Check `scripts/hpc/cluster.sh` (partition, account, resources, conda environment, data
   root). The defaults are those of the SCUT A800 partition; every value can also be overridden
   from the environment, e.g. `CONDA_ENV=myenv bash scripts/hpc/run_repeats.sh`.
2. Probe the micro-batches once: `bash scripts/hpc/probe_batch_size.sh` (updates
   `configs/batch_sizes.yaml`).
3. Submit the repeats: `bash scripts/hpc/run_repeats.sh`.

| Script | Role |
|---|---|
| `scripts/hpc/run_once.sh <model> <dataset> <task> <seed>` | one repeat: training from scratch, pretraining, fine-tuning from the pretrained checkpoint, in sequence |
| `scripts/hpc/run_repeats.sh` | submits one Slurm job running `run_once.sh` per model / dataset / task / seed |
| `scripts/hpc/slurm_job.sh` | job wrapper: loads modules and the conda environment, then runs its arguments |

A repeat writes `runs/<TAG>/scratch/<run>`, `runs/<TAG>/<METHOD>/pretrain/<run>` and
`runs/<TAG>/<METHOD>/finetune/<run>`, with `<run>` = `<model>/<dataset>/<task>/seed_<seed>`.
Finished stages are skipped and an unfinished stage restarts from scratch, so rerunning
`run_repeats.sh` after failures resubmits only the incomplete repeats; repeats whose job is
still queued or running are not submitted twice.

Settings, all read from the environment:

| Variable | Meaning | Default |
|---|---|---|
| `SEEDS` | seeds of the independent repeats | `0 1 2 3 4` |
| `MODELS`, `TASKS` | models and tasks to run | all 10 models; `short long` |
| `GRAPH_DATASETS` | datasets run with every model | the 6 traffic datasets |
| `TIME_SERIES_DATASETS` | datasets run with the time-series models only | the 7 long-term datasets |
| `METHOD` | pretraining method | `jepa` |
| `SCRATCH_EPOCHS`, `PRETRAIN_EPOCHS`, `FINETUNE_EPOCHS` | epochs of the three stages | 20, 10, 10 |
| `TAG` | name of the experiment under `runs/` | `jepa-repeat` |
| `DRY_RUN` | 1: print the `sbatch` commands without submitting | 0 |

Job logs go to `hpc_logs/<TAG>-<METHOD>-<model>-<dataset>-<task>-seed_<seed>.out`.
