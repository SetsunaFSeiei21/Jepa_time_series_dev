"""STAEformer: the untrained model equals the official forward pass.

`official_forward` re-implements the forward of the official code
(https://github.com/XDZhelheim/STAEformer, model/STAEformer.py) with its own tensor operations
(transposes, head splitting along the batch axis, flooring the time-of-day index, integer
day-of-week channel), reading the weights of our model. At initialization the latent predictor
is the identity, so `model(x)` must match it for both output heads.
"""

import pytest
import torch
from hydra import compose, initialize_config_dir

from src.models import build_model
from tests.test_models import CONFIG_DIR

NUM_NODES, BATCH, CHANNELS = 7, 3, 1
TIME_FEATURE_SIZES = {"time_of_day": 288, "day_of_week": 7, "day_of_month": 31, "day_of_year": 366}


def official_attention(attention, x, num_heads):
    """Official AttentionLayer on (batch, ..., length, model_dim)."""
    batch_size = x.shape[0]
    head_dim = x.shape[-1] // num_heads
    query, key, value = attention.query(x), attention.key(x), attention.value(x)
    query = torch.cat(torch.split(query, head_dim, dim=-1), dim=0)
    key = torch.cat(torch.split(key, head_dim, dim=-1), dim=0)
    value = torch.cat(torch.split(value, head_dim, dim=-1), dim=0)
    score = torch.softmax((query @ key.transpose(-1, -2)) / head_dim**0.5, dim=-1)
    out = torch.cat(torch.split(score @ value, batch_size, dim=0), dim=-1)
    return attention.out(out)


def official_layer(layer, x, dim, num_heads):
    """Official SelfAttentionLayer (eval mode: dropout is the identity)."""
    x = x.transpose(dim, -2)
    out = layer.norm1(x + official_attention(layer.attention, x, num_heads))
    out = layer.norm2(out + layer.feed_forward(out))
    return out.transpose(dim, -2)


def official_forward(model, x, steps_per_day, num_heads):
    """x: (batch, in_steps, nodes, input_dim + 2), the last two channels being the time-of-day
    fraction and the integer day of week, as in the official data pipeline."""
    batch_size, in_steps, num_nodes, _ = x.shape
    tod, dow = x[..., -2], x[..., -1]
    features = [model.input_proj(x[..., :-2])]
    features.append(model.time_embeddings["time_of_day"]((tod * steps_per_day).long()))
    features.append(model.time_embeddings["day_of_week"](dow.long()))
    features.append(model.node_embedding.expand(batch_size, in_steps, *model.node_embedding.shape))
    features.append(
        model.adaptive_embedding.expand(size=(batch_size, *model.adaptive_embedding.shape))
    )
    h = torch.cat(features, dim=-1)
    for layer in model.temporal_layers:
        h = official_layer(layer, h, 1, num_heads)
    for layer in model.spatial_layers:
        h = official_layer(layer, h, 2, num_heads)
    if model.use_mixed_proj:
        out = h.transpose(1, 2).reshape(batch_size, num_nodes, in_steps * model.model_dim)
        out = model.output_proj(out).view(batch_size, num_nodes, model.pred_len, -1)
        return out.transpose(1, 2)
    out = model.temporal_proj(h.transpose(1, 3))
    return model.output_proj(out.transpose(1, 3))


def build(task, *overrides):
    """Config and untrained STAEformer (eval mode) for a task, with yaml overrides."""
    overrides = ["model=staeformer", f"task={task}", *overrides]
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose("default", overrides=overrides)
    torch.manual_seed(0)
    model = build_model(
        cfg.model,
        cfg.time_encoding,
        cfg.predictor,
        None,
        num_nodes=NUM_NODES,
        in_channels=CHANNELS,
        out_channels=CHANNELS,
        time_feature_sizes=TIME_FEATURE_SIZES,
        his_len=cfg.task.his_len,
        pred_len=cfg.task.pred_len,
    )
    return cfg, model.eval()


@pytest.mark.parametrize("use_mixed_proj", [True, False])
@pytest.mark.parametrize("task", ["short", "long"])
def test_matches_official_forward(use_mixed_proj, task):
    cfg, model = build(task, f"model.use_mixed_proj={use_mixed_proj}")
    steps = cfg.task.his_len

    sizes = torch.tensor(list(TIME_FEATURE_SIZES.values()))
    index = torch.randint(0, 2**31, (BATCH, steps, len(sizes))).remainder(sizes)
    x_mark = index / sizes  # what the data pipeline provides: index / size in [0, 1)
    y_mark = torch.rand(BATCH, cfg.task.pred_len, len(sizes))  # not read by STAEformer
    x = torch.randn(BATCH, steps, NUM_NODES, CHANNELS)

    official_x = torch.cat(
        [
            x,
            x_mark[:, :, None, 0, None].expand(-1, -1, NUM_NODES, 1),  # time-of-day fraction
            index[:, :, None, 1, None].expand(-1, -1, NUM_NODES, 1).float(),  # day of week
        ],
        dim=-1,
    )
    with torch.no_grad():
        ours = model(x, x_mark, y_mark)
        official = official_forward(
            model, official_x, TIME_FEATURE_SIZES["time_of_day"], cfg.model.num_heads
        )
    assert ours.shape == (BATCH, cfg.task.pred_len, NUM_NODES, CHANNELS)
    torch.testing.assert_close(ours, official, rtol=1e-5, atol=1e-5)


def test_latent_layout():
    """The latent is (b, n, t, d) and the predictor mixes the t steps of each node."""
    cfg, model = build("short")
    x = torch.randn(BATCH, cfg.task.his_len, NUM_NODES, CHANNELS)
    x_mark = torch.zeros(BATCH, cfg.task.his_len, len(TIME_FEATURE_SIZES))
    with torch.no_grad():
        z = model.encode(x, x_mark)
    assert z.shape == (BATCH, NUM_NODES, cfg.task.his_len, model.model_dim)
    assert model.predictor.token_mlp.mlp[0].in_features == cfg.task.his_len
