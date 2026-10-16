import json

import numpy as np
import pytest
import torch

from src.data import SpatioTemporalData, StandardScaler, build_loaders, fit_dataset_scaler

T, N, C, F = 200, 5, 1, 4
MISSING_PROB = 0.1
SPLIT = [0.6, 0.2, 0.2]


@pytest.fixture
def dataset(tmp_path) -> SpatioTemporalData:
    rng = np.random.default_rng(0)
    mask = rng.random((T, N, C)) < MISSING_PROB
    data = np.where(mask, 0.0, rng.normal(10, 3, (T, N, C))).astype(np.float32)
    np.save(tmp_path / "data.npy", data)
    np.save(tmp_path / "mask.npy", mask)
    np.save(tmp_path / "time_features.npy", rng.random((T, F)).astype(np.float32))
    np.save(tmp_path / "graph.npy", rng.random((N, N)).astype(np.float32))
    sizes = {"time_of_day": 24, "day_of_week": 7, "day_of_month": 31, "day_of_year": 366}
    meta = {
        "name": "toy",
        "num_steps": T,
        "num_nodes": N,
        "num_channels": C,
        "time_features": sizes,
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    return SpatioTemporalData.load(tmp_path)


def test_load(dataset):
    assert (dataset.num_steps, dataset.num_nodes, dataset.num_channels) == (T, N, C)
    assert dataset.num_time_features == F and dataset.graph.shape == (N, N)


def test_missing_time_features_give_empty_marks(dataset, tmp_path):
    (tmp_path / "time_features.npy").unlink()
    data = SpatioTemporalData.load(tmp_path)
    batch = data.windows(torch.tensor([0, 1]), his_len=4, pred_len=3)
    assert batch.x_mark.shape == (2, 4, 0)


@pytest.mark.parametrize("his_len,pred_len", [(12, 12), (24, 6)])
def test_windows_respect_splits(dataset, his_len, pred_len):
    bounds = dataset.split_bounds(SPLIT)
    assert bounds[0][0] == 0 and bounds[-1][1] == T
    loaders = build_loaders(dataset, SPLIT, his_len, pred_len, 1, pred_len, 16, 16, "cpu")
    for (lo, hi), (name, loader) in zip(bounds, loaders.items(), strict=True):
        starts = loader.starts
        assert (starts >= 0).all()
        assert (starts + his_len >= lo).all() and (starts + his_len + pred_len <= hi).all()
        if name == "train":
            assert len(starts) == hi - his_len - pred_len + 1
        else:  # eval windows tile the split: every step predicted once, except the tail
            covered = torch.zeros(T, dtype=torch.long)
            for s in starts.tolist():
                covered[s + his_len : s + his_len + pred_len] += 1
            assert covered.max() == 1 and (covered[max(lo, his_len) : hi] == 0).sum() < pred_len
        assert sum(b.x.shape[0] for b in loader) == len(starts)


def test_window_contents(dataset):
    batch = dataset.windows(torch.tensor([3, 50]), his_len=5, pred_len=2)
    torch.testing.assert_close(batch.x[1], dataset.data[50:55])
    torch.testing.assert_close(batch.y[1], dataset.data[55:57])
    assert torch.equal(batch.y_mask[0], dataset.mask[8:10])
    torch.testing.assert_close(batch.y_mark[0], dataset.time_features[8:10])


@pytest.mark.parametrize("mode,axes", [("node", (0,)), ("global", (0, 1))])
def test_dataset_scaler_ignores_missing(dataset, mode, axes):
    train_end = dataset.split_bounds(SPLIT)[0][1]
    scaler = fit_dataset_scaler(dataset.data, dataset.mask, train_end, mode)
    train = np.where(dataset.mask[:train_end], np.nan, dataset.data[:train_end]).astype(np.float64)
    np.testing.assert_allclose(scaler.mean, np.nanmean(train, axes, keepdims=True), rtol=1e-5)
    np.testing.assert_allclose(scaler.std, np.nanstd(train, axes, keepdims=True), rtol=1e-4)


def test_identity_scaler(dataset):
    scaler = fit_dataset_scaler(dataset.data, dataset.mask, 10, "none")
    torch.testing.assert_close(scaler.transform(dataset.data), dataset.data)


def test_instance_scaler_round_trip():
    x = torch.randn(4, 12, 5, 1)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, :3] = True
    x[mask] = 1e6  # garbage in missing entries must not affect the statistics
    scaler = StandardScaler.fit(x, mask, dims=(1,))
    assert scaler.mean.shape == (4, 1, 5, 1)
    torch.testing.assert_close(scaler.mean, x[:, 3:].mean(1, keepdim=True))
    torch.testing.assert_close(scaler.inverse_transform(scaler.transform(x)), x)


def test_constant_series_is_not_scaled():
    x = torch.ones(1, 10, 2, 1)
    scaler = StandardScaler.fit(x, torch.zeros_like(x, dtype=torch.bool), dims=(1,))
    torch.testing.assert_close(scaler.std, torch.ones_like(scaler.std))
