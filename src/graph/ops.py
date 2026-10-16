"""Graph operators on adjacency matrices with any number of leading batch dims: (..., n, n).

Row i of an adjacency lists the edges leaving node i.
"""

import torch
from torch import Tensor


def fill_diagonal(adj: Tensor, value: float) -> Tensor:
    eye = torch.eye(adj.shape[-1], dtype=torch.bool, device=adj.device)
    return adj.masked_fill(eye, value)


def binarize(adj: Tensor) -> Tensor:
    return (adj != 0).to(adj.dtype)


def random_walk(adj: Tensor) -> Tensor:
    """Row-normalized transition matrix D^-1 A; rows without edges stay zero."""
    degree = adj.sum(-1, keepdim=True)
    return torch.where(degree > 0, adj / degree, torch.zeros_like(adj))


def symmetric_normalize(adj: Tensor) -> Tensor:
    """D^-1/2 A D^-1/2 with D the row degrees; isolated nodes stay zero."""
    degree = adj.sum(-1)
    inv_sqrt = torch.where(degree > 0, degree.rsqrt(), torch.zeros_like(degree))
    return inv_sqrt[..., :, None] * adj * inv_sqrt[..., None, :]


def normalized_laplacian(adj: Tensor) -> Tensor:
    eye = torch.eye(adj.shape[-1], dtype=adj.dtype, device=adj.device)
    return eye - symmetric_normalize(adj)


def scaled_laplacian(adj: Tensor, lambda_max: float | None = None) -> Tensor:
    """2 L / lambda_max - I of the symmetrized graph, with spectrum in [-1, 1].

    `lambda_max=None` computes the largest eigenvalue exactly (one eigendecomposition per
    graph); 2 is the usual cheap upper bound for a normalized Laplacian.
    """
    laplacian = normalized_laplacian(torch.maximum(adj, adj.mT))
    if lambda_max is None:
        largest = torch.linalg.eigvalsh(laplacian.double())[..., -1].to(adj.dtype)
        scale = (2 / largest)[..., None, None]
    else:
        scale = 2 / lambda_max
    eye = torch.eye(adj.shape[-1], dtype=adj.dtype, device=adj.device)
    return scale * laplacian - eye


def chebyshev_polynomials(laplacian: Tensor, order: int) -> Tensor:
    """T_0 .. T_{order-1} of a (scaled) Laplacian, stacked as (..., order, n, n)."""
    eye = torch.eye(laplacian.shape[-1], dtype=laplacian.dtype, device=laplacian.device)
    polynomials = [eye.expand_as(laplacian), laplacian]
    while len(polynomials) < order:
        polynomials.append(2 * laplacian @ polynomials[-1] - polynomials[-2])
    return torch.stack(polynomials[:order], dim=-3)
