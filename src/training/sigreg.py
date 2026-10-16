"""SIGReg: sketched isotropic Gaussian regularization of embeddings (LeJEPA,
https://arxiv.org/abs/2511.08544; reference code https://github.com/rbalestr-lab/lejepa).

The embeddings are projected on random unit directions; on every direction the Epps-Pulley
statistic measures how far the projected samples are from a standard normal distribution, by
comparing their empirical characteristic function with exp(-t^2 / 2) on a grid of t. Averaged
over directions, the statistic is minimized when the embeddings are an isotropic standard
Gaussian, which prevents the collapse of a joint-embedding predictive architecture.

The statistic here is the weighted squared distance between the two characteristic functions,
a property of the embedding distribution that does not depend on the number of samples. (The
reference implementation multiplies it by the number of samples, the usual normalization of a
test statistic; then the regularization strength would grow with the number of latent tokens,
which differs by orders of magnitude between models and datasets, and with the micro-batch
size.)
"""

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

# Samples per chunk when summing the empirical characteristic function. The (s, k, points)
# phases are never held at once: each chunk's are recomputed in the backward pass, so the
# memory does not grow with the number of embeddings (a JEPA batch of a large graph has
# hundreds of thousands of tokens). Changing it only changes the order of the sums.
SAMPLES_PER_CHUNK = 4096


def characteristic_sums(x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
    """(s, k), (points,) -> sum over samples of cos(x t) and of sin(x t), each (k, points)"""
    xt = x[..., None] * t
    return xt.cos().sum(0), xt.sin().sum(0)


class EppsPulley(nn.Module):
    """Epps-Pulley distance of each column of (s, k), s samples of k one-dimensional series, to
    the standard normal distribution: the integral over t of |empirical cf(t) - exp(-t^2/2)|^2
    weighted by exp(-t^2/2).

    The integral over t in [-t_max, t_max] uses the trapezoidal rule on `num_points` points of
    [0, t_max] (the integrand is even).
    """

    def __init__(self, t_max: float, num_points: int):
        super().__init__()
        t = torch.linspace(0, t_max, num_points)
        step = t_max / (num_points - 1)
        weights = torch.full((num_points,), 2 * step)  # 2x: the negative half of the integral
        weights[[0, -1]] = step
        normal_cf = (-t.square() / 2).exp()
        self.register_buffer("t", t, persistent=False)
        self.register_buffer("normal_cf", normal_cf, persistent=False)
        self.register_buffer("weights", weights * normal_cf, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """(s, k) -> (k,)"""
        sums = [
            checkpoint(characteristic_sums, chunk, self.t, use_reentrant=False)
            for chunk in x.split(SAMPLES_PER_CHUNK)
        ]
        cos_mean = sum(c for c, _ in sums) / x.shape[0]
        sin_mean = sum(s for _, s in sums) / x.shape[0]
        error = (cos_mean - self.normal_cf).square() + sin_mean.square()
        return error @ self.weights


class SIGReg(nn.Module):
    """Mean Epps-Pulley distance of (s, d) embeddings over `num_slices` random directions.

    The directions are redrawn at every call from a counter-seeded generator, so a run is
    reproducible and successive steps test different directions.
    """

    def __init__(self, num_slices: int, t_max: float, num_points: int):
        super().__init__()
        self.num_slices = num_slices
        self.test = EppsPulley(t_max, num_points)
        self.calls = 0

    def forward(self, x: Tensor) -> Tensor:
        """(s, d) -> scalar"""
        generator = torch.Generator(device=x.device).manual_seed(self.calls)
        self.calls += 1
        directions = torch.randn(x.shape[1], self.num_slices, generator=generator, device=x.device)
        directions = directions / directions.norm(dim=0)
        return self.test(x @ directions).mean()
