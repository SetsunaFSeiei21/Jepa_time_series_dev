"""Graph WaveNet: the untrained model is the original forward, the long-input head trains from
every seed, and gradient checkpointing changes nothing."""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange
from hydra import compose, initialize_config_dir
from torch import nn

from src.models import build_graph, build_model
from src.models.spatiotemporal.gwnet import GWNET

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
NUM_NODES, BATCH, CHANNELS = 6, 2, 1
TIME_FEATURE_SIZES = {"time_of_day": 288, "day_of_week": 7}
EDGE_PROB = 0.4
SEEDS = range(10)


def build(task: str, seed: int = 0, **overrides) -> tuple[GWNET, torch.Tensor, tuple]:
    """Small GWNET on a random graph, its supports and random inputs."""
    model_overrides = [f"model.{key}={value}" for key, value in overrides.items()]
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose("default", overrides=["model=gwnet", f"task={task}", *model_overrides])
    torch.manual_seed(seed)
    adj = torch.rand(NUM_NODES, NUM_NODES) * (torch.rand(NUM_NODES, NUM_NODES) < EDGE_PROB)
    graph = build_graph(cfg.model, adj)
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
    t = cfg.task.his_len
    sizes = torch.tensor(list(TIME_FEATURE_SIZES.values()))
    marks = torch.randint(0, 1000, (BATCH, t, len(sizes))).remainder(sizes) / sizes
    x = torch.randn(BATCH, t, NUM_NODES, CHANNELS)
    supports = graph.supports.expand(BATCH, -1, -1, -1)
    return model, supports, (x, marks, marks)


class OriginalGCN(nn.Module):
    """gcn of the reference code: [x, A x, A^2 x, ...] per support, 1x1 conv, dropout."""

    def __init__(self, c_in: int, c_out: int, support_len: int, order: int):
        super().__init__()
        self.order = order
        self.mlp = nn.Conv2d((order * support_len + 1) * c_in, c_out, kernel_size=1)

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = torch.einsum("ncvl,vw->ncwl", x, a)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x2 = torch.einsum("ncvl,vw->ncwl", x1, a)
                out.append(x2)
                x1 = x2
        return self.mlp(torch.cat(out, dim=1))


class ChannelNorm(nn.LayerNorm):
    """Stand-in for the reference BatchNorm2d (the one deliberate change): over channels."""

    def forward(self, x):
        return rearrange(super().forward(rearrange(x, "b c n t -> b n t c")), "b n t c -> b c n t")


class OriginalGWNet(nn.Module):
    """Forward of gwnet in https://github.com/nnzhan/Graph-WaveNet/blob/master/model.py
    (gcn_bool, addaptadj, random adaptive init), with every layer's graph convolution and
    norm, plus the time MLP after the head's ReLU for inputs longer than the receptive field.
    Input (b, c, n, t) and output (b, pred_len, n, 1) as in the reference."""

    def __init__(self, model: GWNET, num_supports: int):
        super().__init__()
        self.receptive_field = model.receptive_field
        self.nodevec1, self.nodevec2 = model.source_embedding, model.target_embedding
        self.start_conv = model.start_conv
        self.layers = model.layers
        d_res = model.start_conv.out_channels
        d_dil = model.layers[0].filter_conv.out_channels
        order = model.layers[0].graph_conv.steps
        self.gconv = nn.ModuleList()
        self.bn = nn.ModuleList()
        for layer in model.layers:
            gcn = OriginalGCN(d_dil, d_res, num_supports + 1, order)
            norm = ChannelNorm(d_res)
            if layer.residual_output:  # the model's own weights; the last layer keeps random ones
                gcn.mlp.weight.data = layer.graph_conv.mix.weight.data[..., None, None]
                gcn.mlp.bias.data = layer.graph_conv.mix.bias.data
                norm.load_state_dict(layer.norm.state_dict())
            self.gconv.append(gcn)
            self.bn.append(norm)
        self.time_mlp = model.time_mlp
        self.end_conv_1 = nn.Conv2d(*model.end_conv_1.weight.shape[::-1], kernel_size=1)
        self.end_conv_2 = nn.Conv2d(*model.end_conv_2.weight.shape[::-1], kernel_size=1)
        for conv, linear in [
            (self.end_conv_1, model.end_conv_1),
            (self.end_conv_2, model.end_conv_2),
        ]:
            conv.weight.data = linear.weight.data[..., None, None]
            conv.bias.data = linear.bias.data

    def forward(self, inputs, supports):
        in_len = inputs.size(3)
        x = inputs
        if in_len < self.receptive_field:
            x = F.pad(inputs, (self.receptive_field - in_len, 0, 0, 0))
        x = self.start_conv(x)
        skip = 0
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        new_supports = [*supports, adp]
        for i, layer in enumerate(self.layers):
            residual = x
            x = torch.tanh(layer.filter_conv(residual)) * torch.sigmoid(layer.gate_conv(residual))
            s = layer.skip_conv(x)
            skip = s + (skip[:, :, :, -s.size(3) :] if torch.is_tensor(skip) else skip)
            x = self.gconv[i](x, new_supports)
            x = x + residual[:, :, :, -x.size(3) :]
            x = self.bn[i](x)
        x = F.relu(skip)
        if self.time_mlp is not None:
            x = self.time_mlp(x)
        x = F.relu(self.end_conv_1(x))
        return self.end_conv_2(x)


@pytest.mark.parametrize("task", ["short", "long"])
def test_matches_original_forward(task):
    """decode(predict(encode(x))) at initialization is the reference forward, with the same
    gradients for the input projection and the adaptive adjacency (the residual path is live)."""
    model, supports, inputs = build(task)
    # float64: in float32 the summation order alone moves small gradient entries past the
    # default tolerance.
    model.double().eval()
    supports, (x, x_mark, y_mark) = supports.double(), (t.double() for t in inputs)
    original = OriginalGWNet(model, supports.shape[1]).double().eval()

    params = [model.start_conv.weight, model.source_embedding]
    y = model(x, x_mark, y_mark, supports)
    grads = torch.autograd.grad(y.square().sum(), params)

    inputs = rearrange(model.add_time(x, x_mark), "b t n c -> b c n t")
    y_original = original(inputs, list(supports[0]))  # (b, pred_len, n, 1)
    grads_original = torch.autograd.grad(y_original.square().sum(), params)

    torch.testing.assert_close(y, y_original)
    for grad, grad_original in zip(grads, grads_original, strict=True):
        torch.testing.assert_close(grad, grad_original)


@pytest.mark.parametrize("seed", SEEDS)
def test_long_input_head_trains_from_init(seed):
    """With his_len > receptive field the time MLP reduces the remaining steps; the forecast
    must depend on the input and pass gradients to the encoder from every initialization."""
    model, supports, (x, x_mark, y_mark) = build("long", seed=seed)
    assert model.latent_steps > 1
    y = model(x, x_mark, y_mark, supports)
    assert y.std(dim=2).min() > 0, "forecast is the same for every node"
    y.square().mean().backward()
    assert model.start_conv.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("task", ["short", "long"])
def test_gradient_checkpointing_is_exact(task):
    """Checkpointing recomputes each layer in backward: same outputs, same gradients, even with
    dropout active."""
    results = []
    for flag in [False, True]:
        model, supports, inputs = build(task, gradient_checkpointing=flag)
        model.train()
        torch.manual_seed(1)
        y = model(*inputs, supports)
        y.square().mean().backward()
        results.append((y, [p.grad for p in model.parameters()]))
    (y, grads), (y_ckpt, grads_ckpt) = results
    torch.testing.assert_close(y, y_ckpt)
    for grad, grad_ckpt in zip(grads, grads_ckpt, strict=True):
        torch.testing.assert_close(grad, grad_ckpt)
