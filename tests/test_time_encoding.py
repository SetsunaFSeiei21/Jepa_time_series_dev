import math

import pytest
import torch

from src.models.time_encoding import Cyclic, Embedding, NoTime, Raw

from .test_models import TIME_FEATURE_SIZES, random_marks, setup

NUM_FEATURES = len(TIME_FEATURE_SIZES)
EMBEDDING_DIM = 3
STEPS = 5


@pytest.mark.parametrize(
    "encoding,channels",
    [
        (Raw(TIME_FEATURE_SIZES), NUM_FEATURES),
        (Cyclic(TIME_FEATURE_SIZES), 2 * NUM_FEATURES),
        (Embedding(TIME_FEATURE_SIZES, EMBEDDING_DIM), EMBEDDING_DIM * NUM_FEATURES),
        (NoTime(TIME_FEATURE_SIZES), 0),
    ],
    ids=lambda v: type(v).__name__ if not isinstance(v, int) else str(v),
)
def test_shapes(encoding, channels):
    marks = random_marks(STEPS)
    assert encoding.num_channels == channels
    assert encoding(marks).shape == (*marks.shape[:2], channels)


@pytest.mark.parametrize(
    "encoding",
    [Raw({}), Cyclic({}), Embedding({}, EMBEDDING_DIM), NoTime({})],
    ids=lambda e: type(e).__name__,
)
def test_datasets_without_timestamps(encoding):
    assert encoding.num_channels == 0
    assert encoding(torch.zeros(2, STEPS, 0)).shape == (2, STEPS, 0)


def test_cyclic_is_continuous_across_the_period():
    last, first = torch.tensor([[[287 / 288]]]), torch.tensor([[[0.0]]])
    sizes = {"time_of_day": 288}
    gap = (Cyclic(sizes)(last) - Cyclic(sizes)(first)).abs().max()
    assert gap < 2 * math.pi / 288 + 1e-6


def test_embedding_looks_up_the_feature_index():
    sizes = {"time_of_day": 288, "day_of_week": 7}
    encoding = Embedding(sizes, EMBEDDING_DIM)
    marks = torch.tensor([[[287 / 288, 6 / 7], [0.0, 3 / 7]]])
    out = encoding(marks)
    torch.testing.assert_close(out[0, 0, :EMBEDDING_DIM], encoding.tables[0].weight[287])
    torch.testing.assert_close(out[0, 1, EMBEDDING_DIM:], encoding.tables[1].weight[3])


@pytest.mark.parametrize("time_encoding", ["raw", "cyclic", "embedding", "none"])
def test_shared_model_with_every_encoding(time_encoding):
    cfg, model, graph, (x, x_mark, y_mark) = setup("gwnet", "short", time_encoding)
    supports = graph.supports.expand(x.shape[0], -1, -1, -1)
    y = model(x, x_mark, y_mark, supports)
    assert y.shape == (x.shape[0], cfg.task.pred_len, *x.shape[2:])
    y.square().mean().backward()
    assert all(p.grad is not None for p in model.parameters())
