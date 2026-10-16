"""Tests of the MAE and contrastive pretraining methods, on a tiny model and a toy dataset."""

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from einops import rearrange
from hydra import compose, initialize_config_dir
from torch import Tensor, nn

from src.data import SpatioTemporalData, build_loaders, fit_dataset_scaler
from src.models import BaseModel
from src.models.time_encoding import Cyclic
from src.training import PRETRAINERS
from src.training.contrastive import frequency_masking, info_nce
from src.training.mae import segment_mask
from src.training.pretrainer import FINAL_CHECKPOINT

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
STEPS, NODES, CHANNELS, WINDOW, BATCH, HIDDEN = 240, 5, 2, 12, 8, 16
MISSING_PROB, DROPOUT = 0.05, 0.1
TIME_FEATURE_SIZES = {"time_of_day": 24, "day_of_week": 7}
PREDICTOR = {"token_expansion": 2, "channel_expansion": 2}
SPLIT = [0.6, 0.2, 0.2]


class TinyModel(BaseModel):
    """Latent (b, n, t, d): a per-step embedding mixed along time; a per-step linear readout."""

    def __init__(self, hidden_dim: int, dropout: float, **common):
        super().__init__(**common)
        self.embed = nn.Linear(self.in_channels, hidden_dim)
        self.mix = nn.Linear(self.his_len, self.his_len)
        self.dropout = nn.Dropout(dropout)
        self.predictor = self.make_predictor(num_tokens=self.his_len, dim=hidden_dim)
        self.readout = nn.Linear(hidden_dim, self.out_channels)

    def encode(self, x: Tensor, x_mark: Tensor, supports: Tensor | None = None) -> Tensor:
        """(b, t, n, c) -> (b, n, t, d)"""
        h = rearrange(self.embed(self.add_time(x, x_mark)), "b t n d -> b n d t")
        return rearrange(self.dropout(torch.relu(self.mix(h))), "b n d t -> b n t d")

    def predict(self, z: Tensor) -> Tensor:
        return self.predictor(z)

    def decode(self, z, x, x_mark, y_mark, supports=None) -> Tensor:
        """(b, n, p, d) -> (b, p, n, c)"""
        return rearrange(self.readout(z), "b n p c -> b p n c")


def tiny_model() -> TinyModel:
    return TinyModel(
        hidden_dim=HIDDEN,
        dropout=DROPOUT,
        num_nodes=NODES,
        in_channels=CHANNELS,
        out_channels=CHANNELS,
        num_time_features=len(TIME_FEATURE_SIZES),
        his_len=WINDOW,
        pred_len=WINDOW,
        time_encoding=Cyclic(TIME_FEATURE_SIZES),
        predictor=PREDICTOR,
    )


@pytest.fixture
def dataset(tmp_path) -> SpatioTemporalData:
    rng = np.random.default_rng(0)
    steps = np.arange(STEPS)[:, None, None]
    series = 10 + 3 * np.sin(2 * np.pi * steps / 24 + rng.random((1, NODES, CHANNELS)))
    mask = rng.random((STEPS, NODES, CHANNELS)) < MISSING_PROB
    data = np.where(mask, 0.0, series + rng.normal(0, 0.3, series.shape)).astype(np.float32)
    marks = np.stack([(steps[:, 0, 0] % 24) / 24, (steps[:, 0, 0] // 24 % 7) / 7], axis=-1)
    np.save(tmp_path / "data.npy", data)
    np.save(tmp_path / "mask.npy", mask)
    np.save(tmp_path / "time_features.npy", marks.astype(np.float32))
    meta = {
        "name": "toy",
        "num_steps": STEPS,
        "num_nodes": NODES,
        "num_channels": CHANNELS,
        "time_features": TIME_FEATURE_SIZES,
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    return SpatioTemporalData.load(tmp_path)


def make_trainer(method: str, dataset: SpatioTemporalData, save_dir: Path, *overrides: str):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(
            "pretrain",
            overrides=[
                f"pretrain={method}",
                "train.device=cpu",
                "train.max_epochs=1",
                "train.save_every=1",
                f"train.batch_size={BATCH}",
                *overrides,
            ],
        )
    torch.manual_seed(0)
    loaders = build_loaders(
        dataset, SPLIT, WINDOW, WINDOW, 1, WINDOW, BATCH, BATCH, torch.device("cpu")
    )
    train_end = dataset.split_bounds(SPLIT)[0][1]
    scaler = fit_dataset_scaler(dataset.data, dataset.mask, train_end, "node")
    return PRETRAINERS[method](
        model=tiny_model(),
        graph=None,
        loaders=loaders,
        scaler=scaler,
        cfg=cfg.train,
        device=torch.device("cpu"),
        save_dir=save_dir,
        method=cfg.pretrain,
    )


def first_batch(trainer):
    return next(iter(trainer.loaders["val"]))


def eval_inputs(trainer, batch, seed: int = 0):
    """Inputs drawn with a fixed seed, as during evaluation."""
    trainer.model.eval()
    trainer.eval_generator.manual_seed(seed)
    return trainer.inputs(batch)


def changed(before: dict[str, Tensor], module: nn.Module, prefix: str) -> bool:
    after = module.state_dict()
    return any(not torch.equal(before[k], after[k]) for k in before if k.startswith(prefix))


# ---------------------------------------------------------------------------------------- MAE


def test_segment_mask_hides_whole_segments():
    b, t, n, segments, ratio = 4, 12, 3, 4, 0.5
    hidden = segment_mask((b, t, n), segments, ratio, torch.device("cpu"))
    assert hidden.shape == (b, t, n, 1)
    per_segment = rearrange(hidden[..., 0], "b (s r) n -> b n s r", s=segments)
    assert (per_segment.all(-1) | ~per_segment.any(-1)).all()  # a segment is hidden entirely
    assert (per_segment[..., 0].sum(-1) == round(ratio * segments)).all()


def test_mae_input_ignores_hidden_values(dataset, tmp_path):
    """Hidden entries are zero in the input and do not influence it through normalization."""
    trainer = make_trainer("mae", dataset, tmp_path)
    batch = first_batch(trainer)
    inputs = eval_inputs(trainer, batch)
    assert inputs.scored.any() and (inputs.x[inputs.scored] == 0).all()
    perturbed = replace(batch, x=batch.x + 100.0 * inputs.scored)
    other = eval_inputs(trainer, perturbed)
    assert torch.equal(other.x, inputs.x)
    assert torch.equal(other.target[~inputs.scored], inputs.target[~inputs.scored])
    assert not torch.equal(other.target[inputs.scored], inputs.target[inputs.scored])


def test_mae_loss_only_on_hidden_observed_entries(dataset, tmp_path):
    trainer = make_trainer("mae", dataset, tmp_path)
    batch = first_batch(trainer)
    inputs = eval_inputs(trainer, batch)
    assert not (inputs.scored & batch.x_mask).any()
    loss, _ = trainer.loss(inputs)
    noisy_target = inputs.target + 5.0 * torch.randn_like(inputs.target) * ~inputs.scored
    noisy_loss, _ = trainer.loss(replace(inputs, target=noisy_target))
    assert torch.isfinite(loss) and torch.equal(loss, noisy_loss)


@pytest.mark.parametrize("separate_decoder", [True, False])
def test_mae_checkpoint(dataset, tmp_path, separate_decoder):
    """The encoder is trained; the saved decoder is the pretrained one only when it is shared."""
    trainer = make_trainer(
        "mae", dataset, tmp_path, f"pretrain.separate_decoder={separate_decoder}"
    )
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    path = trainer.fit()
    assert changed(before, trainer.model, "embed.") and changed(before, trainer.model, "mix.")
    assert not changed(before, trainer.model, "predictor.")
    assert changed(before, trainer.model, "readout.") == (not separate_decoder)
    assert path == tmp_path / FINAL_CHECKPOINT
    tiny_model().load_state_dict(torch.load(path, weights_only=True))
    record = json.loads((tmp_path / "history.jsonl").read_text().splitlines()[-1])
    assert math.isfinite(record["train"]["loss"]) and math.isfinite(record["val"]["loss"])


def test_mae_validation_is_deterministic(dataset, tmp_path):
    trainer = make_trainer("mae", dataset, tmp_path)
    assert trainer.evaluate("val") == trainer.evaluate("val")


# -------------------------------------------------------------------------------- contrastive


def test_frequency_masking():
    x = torch.randn(4, 12, 3, 2)
    missing = torch.rand(x.shape) < 0.2
    x = x.masked_fill(missing, 0.0)
    assert torch.allclose(frequency_masking(x, missing, 0.0, True), x, atol=1e-6)
    view = frequency_masking(x, torch.zeros_like(missing), 0.5, preserve_dc=True)
    assert torch.allclose(view.mean(1), x.mean(1), atol=1e-6)
    assert not torch.allclose(view, x)
    assert (frequency_masking(x, missing, 0.5, True)[missing] == 0).all()


def test_info_nce():
    m, temperature = 16, 0.2
    collapsed = torch.ones(m, 4)
    loss, _ = info_nce(collapsed, collapsed, temperature)
    assert math.isclose(loss.item(), math.log(m), rel_tol=1e-5)
    distinct = torch.eye(m)
    loss, accuracy = info_nce(distinct, distinct, temperature)
    expected = -math.log(1 / (1 + (m - 1) * math.exp(-1 / temperature)))
    assert math.isclose(loss.item(), expected, rel_tol=1e-5) and accuracy == 1
    swapped = torch.roll(distinct, 1, dims=0)
    assert info_nce(distinct, swapped, temperature)[0] > math.log(m)


def test_contrastive_views_share_everything_but_x(dataset, tmp_path):
    trainer = make_trainer("contrastive", dataset, tmp_path)
    batch = first_batch(trainer)
    inputs = eval_inputs(trainer, batch)
    assert not torch.equal(inputs.x, inputs.x_other)
    assert (inputs.x[batch.x_mask] == 0).all() and (inputs.x_other[batch.x_mask] == 0).all()
    assert torch.equal(eval_inputs(trainer, batch).x_other, inputs.x_other)


@pytest.mark.parametrize("instances", ["token", "window"])
def test_contrastive_checkpoint(dataset, tmp_path, instances):
    trainer = make_trainer("contrastive", dataset, tmp_path, f"pretrain.instances={instances}")
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    head_before = trainer.head.weight.clone()
    path = trainer.fit()
    assert changed(before, trainer.model, "embed.") and changed(before, trainer.model, "mix.")
    assert not changed(before, trainer.model, "readout.")
    assert not torch.equal(trainer.head.weight, head_before)
    state = torch.load(path, weights_only=True)
    assert state.keys() == before.keys()  # the projection head is not saved
    tiny_model().load_state_dict(state)
    record = json.loads((tmp_path / "history.jsonl").read_text().splitlines()[-1])
    assert math.isfinite(record["val"]["loss"]) and 0 <= record["val"]["accuracy"] <= 1
    assert trainer.evaluate("val") == trainer.evaluate("val")
