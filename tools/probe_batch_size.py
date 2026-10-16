"""Largest micro-batch that fits in GPU memory for every model / dataset / task.

All experiments use the same effective batch (`--effective`, 8 by default). A configuration
whose training step does not fit with the full batch runs with a smaller micro-batch and
gradient accumulation, `batch_size x accumulation_steps = effective`. This tool measures, for
the candidate micro-batches (the divisors of the effective batch), the peak GPU memory of one
supervised step and of one JEPA pretraining step (online and target encoders, SIGReg, optimizer
state), and keeps the largest micro-batch whose larger peak fits in `--memory-gb` x `HEADROOM`
(0 when not even a batch of 1 fits or can be estimated). Models with a memory-saving option
(`MEMORY_SAVING_OPTIONS`) are probed again with it when the full batch does not fit, and the
option is recorded as `overrides` if it allows a larger micro-batch.

Run on the target GPU (`--memory-gb` omitted = its total memory), every candidate is tried
directly. On a smaller GPU, candidates that do not fit locally are extrapolated: the peak grows
linearly with the batch size, so two measured sizes give the slope and the intercept (one
measured size gives a conservative line through the origin).

    PYTHONPATH=. python tools/probe_batch_size.py --models gwnet stgcn --datasets pems07 pems08 \
        --tasks short long long192 --memory-gb 48 --out configs/batch_sizes.yaml

`tools/batch_overrides.py` turns the table into the hydra overrides of one configuration.
"""

import argparse
import copy
import gc
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from einops import rearrange
from hydra import compose, initialize_config_dir

from src.data import SpatioTemporalData
from src.models import build_graph, build_model
from src.training.jepa import disable_dropout
from src.training.sigreg import SIGReg

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
BYTES_PER_GB = 1024**3
# Fraction of the GPU memory a configuration may use: the allocator's fragmentation, the
# CUDA context and the dataset tensors kept on the GPU need the rest.
HEADROOM = 0.85
# Model options that trade compute for memory without changing the results (verified by the
# models' tests). They are switched on for a configuration when they allow a larger micro-batch.
MEMORY_SAVING_OPTIONS = {"gradient_checkpointing": "true"}


def compose_config(model: str, dataset: str, task: str, model_overrides: list[str]):
    overrides = [f"model={model}", f"dataset={dataset}", f"task={task}", *model_overrides]
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return compose("pretrain", overrides=overrides)


def peak_memory(cfg, data: SpatioTemporalData, batch_size: int, device: torch.device) -> float:
    """Peak bytes of one supervised step and one JEPA step at `batch_size` (the larger)."""
    graph = build_graph(cfg.model, data.graph.to(device) if data.graph is not None else None)
    model = build_model(
        cfg.model,
        cfg.time_encoding,
        cfg.predictor,
        graph,
        num_nodes=data.num_nodes,
        in_channels=data.num_channels,
        out_channels=data.num_channels,
        time_feature_sizes=data.time_feature_sizes,
        his_len=cfg.task.his_len,
        pred_len=cfg.task.pred_len,
    ).to(device)
    target_encoder = copy.deepcopy(model).requires_grad_(False).eval()
    sigreg = SIGReg(**cfg.pretrain.sigreg).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    shape = (batch_size, cfg.task.his_len, data.num_nodes, data.num_channels)
    x, y = torch.randn(shape, device=device), torch.randn(shape, device=device)
    x_mark = torch.rand(batch_size, cfg.task.his_len, data.num_time_features, device=device)
    y_mark = torch.rand(batch_size, cfg.task.pred_len, data.num_time_features, device=device)
    supports = None if graph is None else graph.supports.expand(batch_size, -1, -1, -1)

    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    F.l1_loss(model(x, x_mark, y_mark, supports), y).backward()
    optimizer.step()
    optimizer.zero_grad()

    disable_dropout(model)
    predicted = model.latent_tokens(model.predict(model.encode(x, x_mark, supports)))
    with torch.no_grad():
        target = target_encoder.latent_tokens(target_encoder.encode(y, y_mark, supports))
    loss = F.mse_loss(predicted, target) + sigreg(rearrange(predicted, "b l d -> (b l) d"))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return torch.cuda.max_memory_allocated(device)


def measure(cfg, data, batch_size, device) -> float | None:
    """Peak bytes, or None when the step runs out of memory."""
    try:
        return peak_memory(cfg, data, batch_size, device)
    except torch.OutOfMemoryError:
        return None
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def choose_batch(cfg, data, candidates, budget, device) -> dict:
    """Largest candidate whose (measured or extrapolated) peak fits in the budget."""
    measured = {}
    for batch_size in sorted(candidates):  # small first: they fit locally and give the slope
        peak = measure(cfg, data, batch_size, device)
        if peak is None:
            break
        measured[batch_size] = peak
    fits = {b: p for b, p in measured.items() if p <= budget}
    estimated = {}
    if measured:
        points = sorted(measured.items())[-2:]
        if len(points) == 1:
            # A line through the origin: an upper bound, since the batch-independent part
            # (weights, optimizer state) does not grow with the batch.
            points.insert(0, (0, 0.0))
        (b0, p0), (b1, p1) = points
        slope = (p1 - p0) / (b1 - b0)
        for batch_size in candidates:
            if batch_size not in measured:
                estimated[batch_size] = p1 + slope * (batch_size - b1)
                if estimated[batch_size] <= budget:
                    fits[batch_size] = estimated[batch_size]
    return {
        "batch_size": max(fits) if fits else 0,
        "measured_mb": {b: round(p / 2**20) for b, p in measured.items()},
        "estimated_mb": {b: round(p / 2**20) for b, p in estimated.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--effective", type=int, default=8, help="effective batch size")
    parser.add_argument("--memory-gb", type=float, help="target GPU memory; default: this GPU")
    parser.add_argument("--data-root", default="~/datasets")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, help="yaml table to update")
    args = parser.parse_args()

    device = torch.device(args.device)
    total = torch.cuda.get_device_properties(device).total_memory
    budget = HEADROOM * (args.memory_gb * BYTES_PER_GB if args.memory_gb else total)
    candidates = [b for b in range(1, args.effective + 1) if args.effective % b == 0]
    table = yaml.safe_load(args.out.read_text()) if args.out and args.out.exists() else {}
    for dataset_name in args.datasets:
        for model in args.models:
            for task in args.tasks:
                cfg = compose_config(model, dataset_name, task, [])
                data = SpatioTemporalData.load(Path(args.data_root) / cfg.dataset.name)
                result = choose_batch(cfg, data, candidates, budget, device)
                saving = [
                    f"model.{k}={v}" for k, v in MEMORY_SAVING_OPTIONS.items() if k in cfg.model
                ]
                if saving and result["batch_size"] < args.effective:
                    cfg = compose_config(model, dataset_name, task, saving)
                    with_saving = choose_batch(cfg, data, candidates, budget, device)
                    if with_saving["batch_size"] > result["batch_size"]:
                        result = {**with_saving, "overrides": saving}
                print(model, dataset_name, task, result, flush=True)
                table.setdefault(model, {}).setdefault(dataset_name, {})[task] = result
                if args.out:
                    args.out.write_text(yaml.safe_dump(table, sort_keys=True))


if __name__ == "__main__":
    main()
