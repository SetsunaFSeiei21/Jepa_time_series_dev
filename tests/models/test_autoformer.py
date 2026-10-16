"""Autoformer: the untrained model equals the official forward pass, and auto-correlation
chooses its delays per sample in training and inference alike.

`official_forward` (tslib_decomposition.py) re-implements the official forward with the
reference code's tensor operations, reading the weights of our model; its auto-correlation is
the reference's inference-phase aggregation. Since the latent predictor is the identity at
initialization, `model(x)` must match it.
"""

import pytest
import torch
from einops import rearrange

from src.training.jepa import disable_dropout
from tests.models.tslib_decomposition import build, official_forward


@pytest.mark.parametrize("label_ratio", [1.0, 0.5])
@pytest.mark.parametrize("task", ["short", "long"])
def test_matches_official_forward(task, label_ratio):
    model, (x, x_mark, y_mark) = build("autoformer", task, label_ratio=label_ratio)
    model.eval()
    with torch.no_grad():
        ours = model(x, x_mark, y_mark)
        reference = official_forward(model, rearrange(x, "b t n c -> b t (n c)"), x_mark, y_mark)
    torch.testing.assert_close(rearrange(ours, "b p n c -> b p (n c)"), reference)


def test_delays_per_sample_in_training():
    """In training mode a sample's forecast does not depend on the rest of its batch."""
    model, (x, x_mark, y_mark) = build("autoformer", "long")
    model.train()
    disable_dropout(model)
    with torch.no_grad():
        both = model(x, x_mark, y_mark)
        alone = [
            model(x[i : i + 1], x_mark[i : i + 1], y_mark[i : i + 1]) for i in range(x.shape[0])
        ]
    torch.testing.assert_close(both, torch.cat(alone))
