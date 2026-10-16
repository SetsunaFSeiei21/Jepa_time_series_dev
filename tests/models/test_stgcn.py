"""STGCN: the untrained model equals the reference forward pass, for every task length.

`reference_forward` re-implements the forward of the reference code
(https://github.com/hazdzz/STGCN, model/models.py STGCNChebGraphConv and model/layers.py) with
its own tensor operations (permutes, zero-channel concatenation in Align, slicing the GLU halves,
the Chebyshev recursion on one fixed graph shift operator), reading the weights of our model.
At initialization the latent predictor is the identity, so `model(x)` must match it.
`reference_gso` re-implements the reference graph preprocessing (calc_gso with sym_norm_lap,
then calc_chebynet_gso), which must match the support our yaml's graph transform produces.
"""

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from src.models import build_graph, build_model
from tests.test_models import CONFIG_DIR

NUM_NODES, BATCH, CHANNELS = 7, 3, 2
TIME_FEATURE_SIZES = {"time_of_day": 288, "day_of_week": 7, "day_of_month": 31, "day_of_year": 366}
EDGE_PROB = 0.4


def reference_gso(adj: np.ndarray) -> np.ndarray:
    """calc_gso(adj, 'sym_norm_lap') followed by calc_chebynet_gso, in dense numpy."""
    adj = np.maximum(adj, adj.T)
    row_sum = adj.sum(axis=1)
    with np.errstate(divide="ignore"):
        deg_inv_sqrt = np.power(row_sum, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    identity = np.eye(adj.shape[0])
    laplacian = identity - deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]
    eigval_max = np.linalg.norm(laplacian, 2)
    if eigval_max >= 2:
        return laplacian - identity
    return 2 * laplacian / eigval_max - identity


def reference_align(align, x, c_out):
    c_in = x.shape[1]
    if c_in > c_out:
        return align.conv(x)
    if c_in < c_out:
        batch_size, _, timestep, n_vertex = x.shape
        zeros = torch.zeros([batch_size, c_out - c_in, timestep, n_vertex]).to(x)
        return torch.cat([x, zeros], dim=1)
    return x


def reference_temporal(layer, x):
    c_out = layer.conv.out_channels // 2
    x_in = reference_align(layer.align, x, c_out)[:, :, layer.kernel_size - 1 :, :]
    x_causal_conv = layer.conv(x)
    x_p = x_causal_conv[:, :c_out, :, :]
    x_q = x_causal_conv[:, -c_out:, :, :]
    return torch.mul(x_p + x_in, torch.sigmoid(x_q))


def reference_graph(layer, x, gso):
    x_gc_in = reference_align(layer.align, x, layer.bias.shape[0])
    h = torch.permute(x_gc_in, (0, 2, 3, 1))
    x_list = [h]
    if layer.order > 1:
        x_list.append(torch.einsum("hi,btij->bthj", gso, h))
    for k in range(2, layer.order):
        x_list.append(torch.einsum("hi,btij->bthj", 2 * gso, x_list[k - 1]) - x_list[k - 2])
    h = torch.stack(x_list, dim=2)
    h = torch.einsum("btkhi,kij->bthj", h, layer.weight) + layer.bias
    return torch.add(h.permute(0, 3, 1, 2), x_gc_in)


def reference_forward(model, x, gso):
    """x: (batch, channels incl. time encoding, steps, nodes) -> (batch, steps, nodes, out)."""
    for block in model.blocks:
        x = reference_temporal(block.temporal_conv_1, x)
        x = torch.relu(reference_graph(block.graph_conv, x, gso))
        x = reference_temporal(block.temporal_conv_2, x)
        x = block.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    head = model.output
    x = reference_temporal(head.temporal_conv, x)
    x = head.norm(x.permute(0, 2, 3, 1))
    x = head.fc_2(torch.relu(head.fc_1(x))).permute(0, 3, 1, 2)  # (b, p * c, 1, n)
    x = x.reshape(x.shape[0], -1, model.out_channels, x.shape[-1])  # (b, p, c, n)
    return x.permute(0, 1, 3, 2)


@pytest.mark.parametrize("task", ["short", "long", "long192"])
def test_untrained_model_matches_reference(task):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose("default", overrides=["model=stgcn", f"task={task}"])
    generator = torch.Generator().manual_seed(0)
    adj = torch.rand(NUM_NODES, NUM_NODES, generator=generator)
    adj = adj * (torch.rand(NUM_NODES, NUM_NODES, generator=generator) < EDGE_PROB)
    graph = build_graph(cfg.model, adj.double())
    gso = torch.from_numpy(reference_gso(adj.double().numpy()))
    torch.testing.assert_close(graph.supports[0, 0], gso)

    torch.manual_seed(0)
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
    ).double()
    model.eval()
    x = torch.randn(BATCH, cfg.task.his_len, NUM_NODES, CHANNELS, dtype=torch.float64)
    sizes = torch.tensor(list(TIME_FEATURE_SIZES.values()))
    marks = [
        torch.randint(0, 2**31, (BATCH, steps, len(sizes))).remainder(sizes) / sizes
        for steps in (cfg.task.his_len, cfg.task.pred_len)
    ]
    x_mark, y_mark = (m.double() for m in marks)
    supports = graph.supports.expand(BATCH, -1, -1, -1)
    with torch.no_grad():
        ours = model(x, x_mark, y_mark, supports)
        inputs = model.add_time(x, x_mark).permute(0, 3, 1, 2)
        expected = reference_forward(model, inputs, gso)
    assert ours.shape == (BATCH, cfg.task.pred_len, NUM_NODES, CHANNELS)
    torch.testing.assert_close(ours, expected)
