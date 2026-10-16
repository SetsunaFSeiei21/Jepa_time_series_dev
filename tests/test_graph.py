import pytest
import torch

from src.data import Batch
from src.graph import (
    Binary,
    Chebyshev,
    Concat,
    DoubleTransition,
    Identity,
    ScaledLaplacian,
    StaticGraph,
    ops,
)

NUM_NODES = 12
EDGE_PROB = 0.3


@pytest.fixture
def adj() -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    weights = torch.rand(3, NUM_NODES, NUM_NODES, generator=generator, dtype=torch.float64)
    return weights * (torch.rand(3, NUM_NODES, NUM_NODES, generator=generator) < EDGE_PROB)


def test_random_walk_rows_sum_to_one(adj):
    adj = ops.fill_diagonal(adj, 1.0)
    torch.testing.assert_close(ops.random_walk(adj).sum(-1), torch.ones(3, NUM_NODES).double())


def test_isolated_nodes_stay_zero():
    adj = torch.zeros(NUM_NODES, NUM_NODES)
    assert (ops.random_walk(adj) == 0).all()
    assert (ops.symmetric_normalize(adj) == 0).all()


def test_scaled_laplacian_spectrum(adj):
    eigenvalues = torch.linalg.eigvalsh(ops.scaled_laplacian(ops.fill_diagonal(adj, 1.0)))
    assert eigenvalues.min() >= -1 - 1e-6 and eigenvalues.max() <= 1 + 1e-6
    torch.testing.assert_close(eigenvalues[..., -1], torch.ones(3).double())


def test_chebyshev_recurrence(adj):
    laplacian = ops.scaled_laplacian(adj)
    t = ops.chebyshev_polynomials(laplacian, order=4)
    torch.testing.assert_close(t[:, 0], torch.eye(NUM_NODES).double().expand(3, -1, -1))
    torch.testing.assert_close(t[:, 1], laplacian)
    torch.testing.assert_close(t[:, 3], 2 * laplacian @ t[:, 2] - t[:, 1])


@pytest.mark.parametrize(
    "transform",
    [
        Identity(),
        Binary(),
        DoubleTransition(),
        ScaledLaplacian(),
        Chebyshev(order=3),
        Concat([Chebyshev(order=3), Binary()]),
    ],
    ids=lambda t: type(t).__name__,
)
def test_transform_is_per_sample(transform, adj):
    supports = transform(adj)
    assert supports.shape == (3, transform.num_supports, NUM_NODES, NUM_NODES)
    one_by_one = torch.cat([transform(adj[i : i + 1]) for i in range(3)])
    torch.testing.assert_close(supports, one_by_one)


def test_self_loops_flag(adj):
    assert (Identity(self_loops=True)(adj).diagonal(dim1=-2, dim2=-1) == 1).all()
    torch.testing.assert_close(Identity(self_loops=False)(adj)[:, 0], adj)


def test_static_graph_is_shared(adj):
    graph = StaticGraph(adj[0], DoubleTransition())
    batch = Batch(*(torch.zeros(5, 1) for _ in range(6)))
    supports = graph(batch)
    assert supports.shape == (5, 2, NUM_NODES, NUM_NODES)
    torch.testing.assert_close(supports[4], DoubleTransition()(adj[:1])[0])
