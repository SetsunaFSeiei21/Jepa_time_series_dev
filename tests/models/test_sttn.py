"""STTN-specific checks: equivalence of the untrained model with a direct transcription of the
LargeST STTN forward (layout (b, c, n, t), per-support diffusion loop), and causality of the
encoder in time."""

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from src.graph import DoubleTransition
from src.models.spatiotemporal.sttn import STTN
from src.models.time_encoding import Cyclic

NUM_NODES, BATCH, CHANNELS, STEPS = 6, 3, 1, 12
FEATURE_SIZES = {"time_of_day": 288, "day_of_week": 7}
HYPER = dict(
    hidden_channels=16,
    end_channels=32,
    blocks=2,
    layers_per_block=1,
    heads=4,
    mlp_expand=2,
    diffusion_steps=2,
    dropout=0.1,
)


def make_model() -> STTN:
    torch.manual_seed(0)
    model = STTN(
        **HYPER,
        num_supports=DoubleTransition.num_supports,
        num_nodes=NUM_NODES,
        in_channels=CHANNELS,
        out_channels=CHANNELS,
        num_time_features=len(FEATURE_SIZES),
        his_len=STEPS,
        pred_len=STEPS,
        time_encoding=Cyclic(FEATURE_SIZES),
        predictor=dict(token_expansion=2, channel_expansion=2),
    )
    return model.eval()


def inputs(seed: int = 1):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(BATCH, STEPS, NUM_NODES, CHANNELS, generator=generator)
    marks = torch.rand(BATCH, STEPS, len(FEATURE_SIZES), generator=generator)
    adj = torch.rand(NUM_NODES, NUM_NODES, generator=generator)
    supports = DoubleTransition()(adj).expand(BATCH, -1, -1, -1)
    return x, marks, supports


def linear(x: Tensor, layer) -> Tensor:
    """1x1 convolution over the channel axis of (b, c, n, t)."""
    return torch.einsum("bcnt,oc->bont", x, layer.weight) + layer.bias[:, None, None]


def attention(x: Tensor, layer, causal: bool) -> Tensor:
    """LargeST Spatial/TemporalAttention on (B, L, C), softmax applied once."""
    B, L, C = x.shape
    h = layer.heads
    qkv = (x @ layer.qkv.weight.T).reshape(B, L, 3, h, C // h).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    attn = (q @ k.transpose(-2, -1)) * layer.scale
    if causal:
        mask = torch.tril(torch.ones(L, L))
        attn = attn.masked_fill(mask == 0, float("-inf"))
    attn = attn.softmax(dim=-1)
    out = (attn @ v).transpose(1, 2).reshape(B, L, C)
    return F.linear(out, layer.proj.weight, layer.proj.bias)


def feed_forward(x: Tensor, ff) -> Tensor:
    norm, fc1, _, _, fc2, _ = ff
    h = F.layer_norm(x, norm.normalized_shape, norm.weight, norm.bias, norm.eps)
    return F.linear(F.gelu(F.linear(h, fc1.weight, fc1.bias)), fc2.weight, fc2.bias)


def gcn(x: Tensor, supports: Tensor, conv, order: int) -> Tensor:
    """LargeST GCN on (b, c, n, t) with one graph per call (here the same for every sample)."""
    out = [x]
    for a in supports:
        x1 = torch.einsum("ncvl,vw->ncwl", x, a)
        out.append(x1)
        for _ in range(2, order + 1):
            x1 = torch.einsum("ncvl,vw->ncwl", x1, a)
            out.append(x1)
    return linear(torch.cat(out, dim=1), conv.mix)


def reference_forward(model: STTN, x: Tensor, marks: Tensor, supports: Tensor) -> Tensor:
    """LargeST STTN.forward, with LayerNorm in place of BatchNorm2d and one softmax."""
    time = model.time_encoding(marks)[:, :, None].expand(-1, -1, NUM_NODES, -1)
    h = torch.cat([x, time], dim=-1).transpose(1, 3)  # (b, c, n, t)
    h = linear(h, model.start)
    for block in model.blocks:
        residual = h
        b, c, n, t = h.shape
        # SpatialTransformer
        s = h.permute(0, 3, 2, 1).reshape(b * t, n, c) + block.spatial_embedding
        for layer in block.spatial_layers:
            s_in = s.reshape(b, t, n, c)
            s = attention(s, layer.attention, causal=False) + s
            s = feed_forward(s, layer.feed_forward) + s
            g = gcn(
                s_in.permute(0, 3, 2, 1), supports[0], layer.graph_conv, HYPER["diffusion_steps"]
            )
            s = g.permute(0, 3, 2, 1).reshape(b * t, n, c) + s
        h = s.reshape(b, t, n, c).permute(0, 3, 2, 1)
        # TemporalTransformer
        u = h.permute(0, 2, 3, 1).reshape(b * n, t, c) + block.temporal_embedding
        for layer in block.temporal_layers:
            u = attention(u, layer.attention, causal=True) + u
            u = feed_forward(u, layer.feed_forward) + u
        h = u.reshape(b, n, t, c).permute(0, 3, 1, 2)
        # norm over channels, then residual
        norm = block.norm
        h = F.layer_norm(h.permute(0, 2, 3, 1), (c,), norm.weight, norm.bias, norm.eps)
        h = h.permute(0, 3, 1, 2) + residual
    h = h[..., -1:]
    fc1, _, fc2 = model.end
    out = linear(F.relu(linear(h, fc1)), fc2)  # (b, pred_len * c_out, n, 1)
    return out[..., 0].reshape(b, STEPS, CHANNELS, NUM_NODES).transpose(2, 3)


def test_untrained_model_is_largest_sttn():
    model = make_model()
    x, marks, supports = inputs()
    with torch.no_grad():
        ours = model(x, marks, marks, supports)
        reference = reference_forward(model, x, marks, supports)
    torch.testing.assert_close(ours, reference, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("step", [0, STEPS // 2, STEPS - 1])
def test_encoder_is_causal_in_time(step):
    """The latent at step s depends on the window up to s only (each time step is mixed only
    with the past, through the causal temporal attention)."""
    model = make_model()
    x, marks, supports = inputs()
    changed = x.clone()
    changed[:, step] += 1.0
    with torch.no_grad():
        z = model.encode(x, marks, supports)  # (b, n, t, d)
        z_changed = model.encode(changed, marks, supports)
    torch.testing.assert_close(z[:, :, :step], z_changed[:, :, :step])
    assert not torch.allclose(z[:, :, step:], z_changed[:, :, step:])
