"""ASTGCN: equivalence with the reference code's forward and unsaturated attentions at init."""

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from src.graph import Chebyshev
from src.models.spatiotemporal.astgcn import ASTGCN
from src.models.time_encoding import Cyclic

TIME_FEATURE_SIZES = {"time_of_day": 288, "day_of_week": 7}
BATCH, CHANNELS, ORDER = 2, 1, 3
# Pre-activation beyond which the sigmoid's derivative s (1 - s) is below 1e-3.
SATURATION = 6.9


def make_model(num_nodes: int, seq_len: int, time_stride: int, width: int = 16) -> ASTGCN:
    return ASTGCN(
        num_blocks=2,
        cheb_channels=width,
        time_channels=width,
        time_kernel_size=3,
        time_stride=time_stride,
        num_supports=ORDER,
        num_nodes=num_nodes,
        in_channels=CHANNELS,
        out_channels=CHANNELS,
        num_time_features=len(TIME_FEATURE_SIZES),
        his_len=seq_len,
        pred_len=seq_len,
        time_encoding=Cyclic(TIME_FEATURE_SIZES),
        predictor=dict(token_expansion=2, channel_expansion=2),
    )


def make_inputs(num_nodes: int, seq_len: int, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    adj = torch.rand(num_nodes, num_nodes, generator=generator) < 0.3
    supports = Chebyshev(ORDER, self_loops=False)(adj.float()).expand(BATCH, -1, -1, -1)
    x = torch.randn(BATCH, seq_len, num_nodes, CHANNELS, generator=generator)
    x_mark = torch.rand(BATCH, seq_len, len(TIME_FEATURE_SIZES), generator=generator)
    return x, x_mark, supports


def reference_block(block, x, cheb_polynomials):
    """ASTGCN_block.forward of the reference code, step by step; x: (b, n, c, t)."""
    batch, num_nodes, num_features, seq_len = x.shape
    ta = block.temporal_attention
    lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), ta.u1), ta.u2)
    rhs = torch.matmul(ta.u3, x)
    # The reference score, divided by sqrt(n) (the model's documented score scaling).
    product = torch.matmul(lhs, rhs) / num_nodes**0.5
    temporal = F.softmax(torch.matmul(ta.v, torch.sigmoid(product + ta.bias)), dim=1)
    x_tat = torch.matmul(x.reshape(batch, -1, seq_len), temporal)
    x_tat = x_tat.reshape(batch, num_nodes, num_features, seq_len)

    sa = block.spatial_attention
    lhs = torch.matmul(torch.matmul(x_tat, sa.w1), sa.w2)
    rhs = torch.matmul(sa.w3, x_tat).transpose(-1, -2)
    product = torch.matmul(lhs, rhs) / seq_len**0.5  # divided by sqrt(t), likewise
    spatial = F.softmax(torch.matmul(sa.v, torch.sigmoid(product + sa.bias)), dim=1)

    outputs = []
    for step in range(seq_len):
        signal = x[:, :, :, step]
        output = 0
        for k, polynomial in enumerate(cheb_polynomials):
            rhs = (polynomial * spatial).permute(0, 2, 1).matmul(signal)
            output = output + rhs.matmul(block.cheb_conv.theta[k])
        outputs.append(output.unsqueeze(-1))
    gcn = F.relu(torch.cat(outputs, dim=-1))

    time_out = block.time_conv(gcn.permute(0, 2, 1, 3))
    residual = block.residual_conv(x.permute(0, 2, 1, 3))
    out = F.relu(residual + time_out).permute(0, 3, 2, 1)
    return block.norm(out).permute(0, 2, 3, 1)


@pytest.mark.parametrize("time_stride", [1, 2])
def test_matches_reference_forward(time_stride):
    """With arbitrary parameters, the untrained model (predictor = identity) computes the
    reference code's forward on the time-augmented input."""
    num_nodes, seq_len = 6, 12
    torch.manual_seed(0)
    model = make_model(num_nodes, seq_len, time_stride).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not name.startswith("predictor."):  # keep the predictor the identity
                p.normal_(std=0.3)
    x, x_mark, supports = make_inputs(num_nodes, seq_len)
    with torch.no_grad():
        ours = model(x, x_mark, x_mark, supports)
        h = rearrange(model.add_time(x, x_mark), "b t n c -> b n c t")
        polynomials = list(supports[0])
        for block in model.blocks:
            h = reference_block(block, h, polynomials)
        out = model.output_conv(h.permute(0, 3, 1, 2))[:, :, :, -1]  # (b, p, n)
    torch.testing.assert_close(ours, out.unsqueeze(-1), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("num_nodes, seq_len", [(20, 12), (300, 12), (20, 96), (100, 96)])
def test_attention_not_saturated_at_init(num_nodes, seq_len, monkeypatch):
    """The sigmoid of every attention starts in its responsive range, and the score factors
    of the last block receive a gradient."""
    torch.manual_seed(0)
    model = make_model(num_nodes, seq_len, time_stride=1, width=64)
    pre_activations = []
    sigmoid = torch.sigmoid

    def recording_sigmoid(z):
        pre_activations.append(z.detach())
        return sigmoid(z)

    monkeypatch.setattr(torch, "sigmoid", recording_sigmoid)
    x, x_mark, supports = make_inputs(num_nodes, seq_len)
    model(x, x_mark, x_mark, supports).square().mean().backward()
    assert len(pre_activations) == 2 * len(model.blocks)
    for z in pre_activations:
        assert (z.abs() > SATURATION).float().mean() < 0.01
    last = model.blocks[-1]
    for factor in ("u1", "u2", "u3"):
        assert getattr(last.temporal_attention, factor).grad.abs().max() > 0
    for factor in ("w1", "w2", "w3"):
        assert getattr(last.spatial_attention, factor).grad.abs().max() > 0
