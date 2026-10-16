"""Graph transforms: adjacency (b, n, n) -> supports (b, k, n, n) consumed by a model.

A model selects its transform in its yaml under `graph_transform` (hydra `_target_`). Models
that take a graph receive `num_supports` (= k) at construction and `supports` in forward.
"""

from collections.abc import Sequence

import torch
from torch import Tensor

from . import ops


class GraphTransform:
    """Base class; subclasses implement `_supports` and set `num_supports`."""

    num_supports: int

    def __init__(self, self_loops: bool = True):
        self.self_loops = self_loops

    def __call__(self, adj: Tensor) -> Tensor:
        if self.self_loops:
            adj = ops.fill_diagonal(adj, 1.0)
        supports = self._supports(adj)
        assert supports.shape[-3] == self.num_supports, (supports.shape, self.num_supports)
        return supports

    def _supports(self, adj: Tensor) -> Tensor:
        raise NotImplementedError


class Identity(GraphTransform):
    """The adjacency itself."""

    num_supports = 1

    def _supports(self, adj: Tensor) -> Tensor:
        return adj.unsqueeze(-3)


class Binary(GraphTransform):
    """0/1 connectivity of the adjacency."""

    num_supports = 1

    def _supports(self, adj: Tensor) -> Tensor:
        return ops.binarize(adj).unsqueeze(-3)


class SymmetricNormalized(GraphTransform):
    """D^-1/2 A D^-1/2 of the symmetrized adjacency, with self-loops by default (GCN)."""

    num_supports = 1

    def _supports(self, adj: Tensor) -> Tensor:
        return ops.symmetric_normalize(torch.maximum(adj, adj.mT)).unsqueeze(-3)


class DoubleTransition(GraphTransform):
    """Forward and backward random-walk matrices (DCRNN, Graph WaveNet)."""

    num_supports = 2

    def _supports(self, adj: Tensor) -> Tensor:
        return torch.stack([ops.random_walk(adj), ops.random_walk(adj.mT)], dim=-3)


class ScaledLaplacian(GraphTransform):
    """Scaled Laplacian of the symmetrized adjacency (STGCN)."""

    num_supports = 1

    def __init__(self, lambda_max: float | None = None, self_loops: bool = True):
        super().__init__(self_loops)
        self.lambda_max = lambda_max

    def _supports(self, adj: Tensor) -> Tensor:
        return ops.scaled_laplacian(adj, self.lambda_max).unsqueeze(-3)


class Chebyshev(GraphTransform):
    """Chebyshev polynomials T_0 .. T_{order-1} of the scaled Laplacian (ChebNet, ASTGCN)."""

    def __init__(
        self,
        order: int,
        binary: bool = True,
        lambda_max: float | None = None,
        self_loops: bool = True,
    ):
        super().__init__(self_loops)
        self.num_supports = order
        self.binary = binary
        self.lambda_max = lambda_max

    def _supports(self, adj: Tensor) -> Tensor:
        if self.binary:
            adj = ops.binarize(adj)
        laplacian = ops.scaled_laplacian(adj, self.lambda_max)
        return ops.chebyshev_polynomials(laplacian, self.num_supports)


class Edgeless(GraphTransform):
    """`transform` of a graph without edges, whatever the adjacency: a graph model then mixes no
    nodes and keeps the rest of its architecture (ablation of the spatial information).

    Normalizations need a fixed scale on an edgeless graph, e.g. ScaledLaplacian(lambda_max=2):
    its normalized Laplacian is zero, so the scaled Laplacian is -I and every Chebyshev term is
    diagonal."""

    def __init__(self, transform: GraphTransform):
        super().__init__(self_loops=False)  # the child applies its own setting
        self.transform = transform
        self.num_supports = transform.num_supports

    def _supports(self, adj: Tensor) -> Tensor:
        supports = self.transform(torch.zeros_like(adj))
        assert torch.isfinite(supports).all(), (
            "set a fixed scale (e.g. lambda_max) for an edgeless graph"
        )
        return supports


class Concat(GraphTransform):
    """Supports of several transforms concatenated along k, in the given order."""

    def __init__(self, transforms: Sequence[GraphTransform]):
        super().__init__(self_loops=False)  # each child applies its own setting
        self.transforms = list(transforms)
        self.num_supports = sum(t.num_supports for t in self.transforms)

    def _supports(self, adj: Tensor) -> Tensor:
        return torch.cat([t(adj) for t in self.transforms], dim=-3)
