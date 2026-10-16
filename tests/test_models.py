"""Contract tests run for every model config in configs/model, on both tasks."""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from src.graph import Edgeless
from src.models import BaseModel, build_graph, build_model
from src.training.jepa import disable_dropout

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
MODELS = sorted(p.stem for p in (CONFIG_DIR / "model").glob("*.yaml"))
TASKS = ["short", "long"]
NUM_NODES, BATCH, CHANNELS = 8, 2, 1
EDGE_PROB = 0.3
# Hidden / latent width shared by all models (docs/adding_a_model.md, "Shared widths").
LATENT_WIDTH = 128
TIME_FEATURE_SIZES = {"time_of_day": 288, "day_of_week": 7, "day_of_month": 31, "day_of_year": 366}
# Agreed exceptions to the shared time encoding; each states its reason in its docstring.
CUSTOM_TIME_FEATURES = {"autoformer", "fedformer", "staeformer", "patchtst", "crossformer"}


def random_adjacency(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    weights = torch.rand(NUM_NODES, NUM_NODES, generator=generator)
    return weights * (torch.rand(NUM_NODES, NUM_NODES, generator=generator) < EDGE_PROB)


def random_marks(steps: int) -> torch.Tensor:
    """Valid time features: every feature is an integer index divided by its size."""
    sizes = torch.tensor(list(TIME_FEATURE_SIZES.values()))
    return torch.randint(0, 2**31, (BATCH, steps, len(sizes))).remainder(sizes) / sizes


def setup(model_name: str, task: str, time_encoding: str = "cyclic"):
    overrides = [f"model={model_name}", f"task={task}", f"time_encoding={time_encoding}"]
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose("default", overrides=overrides)
    torch.manual_seed(0)
    graph = build_graph(cfg.model, random_adjacency(seed=0))
    model = build_model(
        cfg.model,
        cfg.time_encoding,
        cfg.predictor,
        graph,
        num_nodes=NUM_NODES,
        in_channels=CHANNELS,
        out_channels=CHANNELS,
        time_feature_sizes=TIME_FEATURE_SIZES,
        his_len=cfg.task.his_len,
        pred_len=cfg.task.pred_len,
    )
    x = torch.randn(BATCH, cfg.task.his_len, NUM_NODES, CHANNELS)
    marks = random_marks(cfg.task.his_len), random_marks(cfg.task.pred_len)
    return cfg, model, graph, (x, *marks)


def static_supports(graph, batch_size: int):
    return None if graph is None else graph.supports.expand(batch_size, -1, -1, -1)


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("model_name", MODELS)
def test_forward_backward(model_name, task):
    cfg, model, graph, (x, x_mark, y_mark) = setup(model_name, task)
    model.train()
    y = model(x, x_mark, y_mark, static_supports(graph, BATCH))
    assert y.shape == (BATCH, cfg.task.pred_len, NUM_NODES, CHANNELS)
    assert torch.isfinite(y).all()

    y.square().mean().backward()
    unused = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not unused, f"parameters without gradient: {unused}"


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("model_name", MODELS)
def test_jepa_latents(model_name, task):
    """The predicted latent of the input window is comparable with the target window's latent,
    has the shared width, and the predictor starts as the identity (so the untrained model is the
    original one)."""
    cfg, model, graph, (x, x_mark, y_mark) = setup(model_name, task)
    supports = static_supports(graph, BATCH)
    model.eval()
    with torch.no_grad():
        z = model.encode(x, x_mark, supports)
        predicted = model.latent_tokens(model.predict(z))
        y = torch.randn(BATCH, cfg.task.pred_len, NUM_NODES, CHANNELS)
        target = model.latent_tokens(model.encode(y, y_mark, supports))
    assert predicted.dim() == 3 and predicted.shape[0] == BATCH
    assert predicted.shape[-1] == LATENT_WIDTH
    assert predicted.shape == target.shape
    torch.testing.assert_close(predicted, model.latent_tokens(z))


@pytest.mark.parametrize("model_name", MODELS)
def test_train_eval_consistency(model_name):
    """Apart from dropout, a model behaves the same in train and eval mode (no BatchNorm or
    other batch-dependent layers), so the JEPA target encoder (eval mode) matches the online
    encoder (train mode with dropout disabled), whatever the micro-batch size."""
    _, model, graph, (x, x_mark, y_mark) = setup(model_name, "short")
    supports = static_supports(graph, BATCH)
    with torch.no_grad():
        model.train()
        disable_dropout(model)
        train_out = model(x, x_mark, y_mark, supports)
        model.eval()
        eval_out = model(x, x_mark, y_mark, supports)
    torch.testing.assert_close(train_out, eval_out)


@pytest.mark.parametrize("model_name", MODELS)
def test_per_sample_graph(model_name):
    """Each sample must be driven by its own graph only."""
    cfg, model, graph, (x, x_mark, y_mark) = setup(model_name, "short")
    if graph is None:
        pytest.skip("model takes no graph")
    if isinstance(graph.transform, Edgeless):
        pytest.skip("edgeless ablation ignores the graph by design")
    model.eval()
    adj = torch.stack([random_adjacency(seed=1), random_adjacency(seed=2)])
    supports = graph.transform(adj)  # (2, k, n, n), different per sample
    with torch.no_grad():
        both = model(x, x_mark, y_mark, supports)
        first_alone = model(x[:1], x_mark[:1], y_mark[:1], supports[:1])
        second_with_first_graph = model(x, x_mark, y_mark, supports[[0, 0]])
    torch.testing.assert_close(both[:1], first_alone, rtol=1e-4, atol=1e-5)
    assert not torch.allclose(both[1], second_with_first_graph[1]), "graph of sample 1 is ignored"


@pytest.mark.parametrize("model_name", MODELS)
def test_time_feature_policy(model_name):
    """Only the agreed exceptions handle time themselves; all others use the shared encoding,
    and no model overrides `forward` (forecasting is always decode(predict(encode(x))))."""
    _, model, _, _ = setup(model_name, "short")
    model_class = type(model)
    assert model_class.custom_time_features == (model_name in CUSTOM_TIME_FEATURES)
    assert model_class.forward is BaseModel.forward
