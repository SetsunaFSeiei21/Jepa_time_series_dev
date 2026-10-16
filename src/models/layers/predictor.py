from torch import Tensor, nn


class ResidualMLP(nn.Module):
    """x + MLP(x) over the last axis. The output layer starts at zero, so the block is the
    identity at initialization and learns a correction from there."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        """(..., dim) -> (..., dim)"""
        return x + self.mlp(x)


class LatentPredictor(nn.Module):
    """JEPA predictor: maps the latent tokens of a window to those of the following window.

    One MLP-Mixer block: a residual MLP across the `num_tokens` tokens (mixing time steps,
    patches, ...) followed by a residual MLP across the `dim` latent channels. Both are the
    identity at initialization (see `ResidualMLP`), so a model computes `decode(encode(x))`
    until the predictor is trained. Any other axes (e.g. nodes) are kept separate.
    """

    def __init__(self, num_tokens: int, dim: int, token_expansion: int, channel_expansion: int):
        super().__init__()
        self.token_mlp = ResidualMLP(num_tokens, token_expansion * num_tokens)
        self.channel_mlp = ResidualMLP(dim, channel_expansion * dim)

    def forward(self, z: Tensor) -> Tensor:
        """(..., l, d) -> (..., l, d)"""
        z = self.token_mlp(z.mT).mT
        return self.channel_mlp(z)
