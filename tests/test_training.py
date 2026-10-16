"""Tests of the training components shared by all models."""

import math

import torch
from omegaconf import OmegaConf
from torch import nn

from src.data import Batch, StandardScaler
from src.training.jepa import JEPATrainer, disable_dropout
from src.training.sigreg import SIGReg
from src.training.trainer import model_inputs

NUM_SLICES, T_MAX, NUM_POINTS = 64, 3.0, 17
SAMPLES, DIM = 4096, 16
# The distance of a large Gaussian sample is O(1 / samples); of a strongly scaled one, O(0.1).
GAUSSIAN_BOUND, SCALE = 1e-3, 4.0


def test_sigreg_prefers_isotropic_gaussian():
    sigreg = SIGReg(NUM_SLICES, T_MAX, NUM_POINTS)
    torch.manual_seed(0)
    gaussian = sigreg(torch.randn(SAMPLES, DIM))
    scaled = sigreg(SCALE * torch.randn(SAMPLES, DIM))
    collapsed = sigreg(torch.zeros(SAMPLES, DIM))
    assert gaussian < GAUSSIAN_BOUND
    assert scaled > 10 * gaussian and collapsed > 10 * gaussian


def test_sigreg_does_not_scale_with_the_number_of_samples():
    sigreg = SIGReg(NUM_SLICES, T_MAX, NUM_POINTS)
    torch.manual_seed(0)
    small = sigreg(SCALE * torch.randn(SAMPLES, DIM))
    sigreg.calls = 0  # same directions
    large = sigreg(SCALE * torch.randn(16 * SAMPLES, DIM))
    torch.testing.assert_close(small, large, rtol=0.05, atol=0)


def test_sigreg_redraws_directions():
    sigreg = SIGReg(NUM_SLICES, T_MAX, NUM_POINTS)
    x = torch.randn(SAMPLES, DIM) * torch.linspace(0.5, 2, DIM)
    assert sigreg(x) != sigreg(x)


def test_disable_dropout_keeps_train_mode():
    model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5), nn.LayerNorm(4))
    model.train()
    disable_dropout(model)
    assert model.training and model[0].training and model[2].training
    assert not model[1].training


class _Schedule:
    """The attributes `JEPATrainer.ema_momentum` reads."""

    def __init__(self, start: float, end: float, total_updates: int):
        self.method = OmegaConf.create({"ema": {"start": start, "end": end}})
        self.total_updates = total_updates
        self.num_updates = 0


def test_ema_momentum_schedule():
    start, end, total = 0.99, 1.0, 11
    schedule = _Schedule(start, end, total)
    momenta = []
    for update in range(total):
        schedule.num_updates = update
        momenta.append(JEPATrainer.ema_momentum(schedule))
    assert math.isclose(momenta[0], start) and math.isclose(momenta[-1], end)
    assert all(a <= b for a, b in zip(momenta, momenta[1:], strict=False))


class _Centering:
    """The attributes `JEPATrainer.center` reads."""

    def __init__(self, momentum: float):
        self.method = OmegaConf.create({"centering": {"momentum": momentum}})
        self.model = nn.Identity()
        self.token_mean = None

    def update_token_mean(self, target):
        return JEPATrainer.update_token_mean(self, target)


def test_centering_removes_the_window_independent_part():
    """Tokens = per-position constant + window content. After training batches of a single
    window (the smallest micro-batch), centering leaves the content only, and evaluation does not
    move the running mean."""
    momentum, steps, tokens, dim = 0.9, 200, 5, 3
    torch.manual_seed(0)
    position = 10 * torch.randn(tokens, dim)
    centering = _Centering(momentum)
    for _ in range(steps):
        target = position + torch.randn(1, tokens, dim)
        JEPATrainer.center(centering, target.clone(), target)
    torch.testing.assert_close(centering.token_mean, position, atol=1.0, rtol=0)
    content = torch.randn(4, tokens, dim)
    predicted, target = JEPATrainer.center(centering, position + content, position + content)
    assert predicted.abs().mean() < position.abs().mean() / 4
    torch.testing.assert_close(predicted, target)
    centering.model.eval()
    frozen = centering.token_mean.clone()
    JEPATrainer.center(centering, position + 100, position + 100)
    torch.testing.assert_close(centering.token_mean, frozen)


def test_instance_norm_std_floor():
    x = torch.zeros(2, 12, 3, 1)
    x[0] = 0.01 * torch.randn(12, 3, 1)  # nearly flat window: only centered
    x[1] = 5 * torch.randn(12, 3, 1)  # wide window: z-scored as usual
    batch = Batch(
        x=x,
        y=x.clone(),
        x_mask=torch.zeros_like(x, dtype=torch.bool),
        y_mask=torch.zeros_like(x, dtype=torch.bool),
        x_mark=torch.zeros(2, 12, 0),
        y_mark=torch.zeros(2, 12, 0),
    )
    inputs = model_inputs(batch, StandardScaler.identity(x.device), True, None)
    torch.testing.assert_close(inputs.instance.std[0], torch.ones(1, 3, 1))
    torch.testing.assert_close(inputs.instance.std[1], x[1].std(0, unbiased=False, keepdim=True))
