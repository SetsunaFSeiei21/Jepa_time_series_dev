"""PatchTST: the untrained model equals the official forward pass.

`official_forward` re-implements the forward of the official code
(https://github.com/yuqinie98/PatchTST, PatchTST_supervised/layers/PatchTST_backbone.py) with
its own tensor operations (ReplicationPad1d at the end, unfold, permute / reshape, per-variable
nn.Flatten + Linear heads), reading the weights of our model. The official encoder uses
BatchNorm; ours uses LayerNorm, so the reference applies our LayerNorm modules at the same
places. At initialization the latent predictor is the identity, so `model(x)` must match it.
"""

import pytest
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from torch import nn

from src.models import build_model
from tests.test_models import CONFIG_DIR

NUM_NODES, BATCH, CHANNELS = 7, 3, 1
TIME_FEATURE_SIZES = {"time_of_day": 288, "day_of_week": 7, "day_of_month": 31, "day_of_year": 366}
# (task, his_len = pred_len, individual); 128 with patch_len 24 / stride 12 leaves a remainder
# after patching, which the end padding must not drop from the newest steps.
CASES = [("short", 12, True), ("long", 96, True), ("long", 96, False), ("long", 128, True)]


def build(task: str, length: int, individual: bool):
    overrides = [
        "model=patchtst",
        f"task={task}",
        f"task.his_len={length}",
        f"task.pred_len={length}",
        f"model.individual={individual}",
    ]
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
        his_len=length,
        pred_len=length,
    )
    # Move LayerNorm away from the identity so that its placement is tested too.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.LayerNorm):
                module.weight.add_(0.1 * torch.randn_like(module.weight))
                module.bias.add_(0.1 * torch.randn_like(module.bias))
    return cfg, model.eval()


def official_attention(attention, src, prev, n_heads):
    """Official _MultiheadAttention + _ScaledDotProductAttention with res_attention=True."""
    bs = src.size(0)
    d_k = src.size(-1) // n_heads
    q_s = attention.query(src).view(bs, -1, n_heads, d_k).transpose(1, 2)
    k_s = attention.key(src).view(bs, -1, n_heads, d_k).permute(0, 2, 3, 1)
    v_s = attention.value(src).view(bs, -1, n_heads, d_k).transpose(1, 2)
    attn_scores = torch.matmul(q_s, k_s) * d_k**-0.5
    if prev is not None:
        attn_scores = attn_scores + prev
    output = torch.matmul(F.softmax(attn_scores, dim=-1), v_s)
    output = output.transpose(1, 2).contiguous().view(bs, -1, n_heads * d_k)
    return attention.out(output), attn_scores


def official_forward(model, x, n_heads):
    """x: (batch, input length, channel) as in the official Model.forward (eval mode)."""
    z = x.permute(0, 2, 1)  # [bs x nvars x seq_len]
    z = nn.ReplicationPad1d((0, model.stride))(z)
    z = z.unfold(dimension=-1, size=model.patch_len, step=model.stride)
    z = z.permute(0, 1, 3, 2)  # [bs x nvars x patch_len x patch_num]

    # _TSTiEncoder
    n_vars = z.shape[1]
    z = model.patch_embedding(z.permute(0, 1, 3, 2))  # [bs x nvars x patch_num x d_model]
    u = torch.reshape(z, (z.shape[0] * z.shape[1], z.shape[2], z.shape[3]))
    src = u + model.position
    scores = None
    for layer in model.layers:
        src2, scores = official_attention(layer.attention, src, scores, n_heads)
        src = layer.attention_norm(src + src2)
        src = layer.ff_norm(src + layer.ff(src))
    z = torch.reshape(src, (-1, n_vars, src.shape[-2], src.shape[-1]))
    z = z.permute(0, 1, 3, 2)  # [bs x nvars x d_model x patch_num]

    # Flatten_Head
    flatten = nn.Flatten(start_dim=-2)
    if model.head_weight.dim() == 3:  # individual
        out = torch.stack(
            [
                F.linear(flatten(z[:, i]), model.head_weight[i], model.head_bias[i])
                for i in range(n_vars)
            ],
            dim=1,
        )
    else:
        out = F.linear(flatten(z), model.head_weight, model.head_bias)
    return out.permute(0, 2, 1)  # [Batch x Output length x Channel]


@pytest.mark.parametrize("task, length, individual", CASES)
def test_matches_official_forward(task, length, individual):
    cfg, model = build(task, length, individual)
    x = torch.randn(BATCH, length, NUM_NODES, CHANNELS)
    with torch.no_grad():
        ours = model(x, None, None)
        reference = official_forward(model, x[..., 0], cfg.model.n_heads)
    assert ours.shape == (BATCH, length, NUM_NODES, CHANNELS)
    torch.testing.assert_close(ours[..., 0], reference, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("task, length, individual", CASES)
def test_every_input_step_reaches_the_latent(task, length, individual):
    """The replication padding is at the end of the series, so the patches cover every step,
    in particular the newest ones."""
    _, model = build(task, length, individual)
    x = torch.randn(BATCH, length, NUM_NODES, CHANNELS, requires_grad=True)
    model.encode(x, None).sum().backward()
    covered = x.grad.abs().sum(dim=(0, 2, 3)) > 0
    assert covered.all(), f"steps without influence: {(~covered).nonzero().flatten().tolist()}"


def test_series_are_independent():
    """Channel independence: a forecast depends only on its own series."""
    _, model = build("long", 96, True)
    x = torch.randn(BATCH, 96, NUM_NODES, CHANNELS)
    changed = x.clone()
    changed[:, :, 0] += 1.0
    with torch.no_grad():
        diff = (model(x, None, None) - model(changed, None, None)).abs().sum(dim=(0, 1, 3))
    assert diff[0] > 0
    assert (diff[1:] == 0).all()
