import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from torch import Tensor, nn

from src.models.base import BaseModel


def init_score_factor(factor: nn.Parameter) -> None:
    """N(0, 1/fan_in) entries for a factor of a bilinear attention score (U_1, U_2, U_3 or W_1,
    W_2, W_3), fan_in being the axis it sums over (the first one). Each factor then maps
    unit-scale inputs to unit-scale outputs; with the 1/sqrt(length) scaling of the score (see
    `TemporalAttention`), the scores are O(1) whatever the numbers of nodes, steps and channels,
    and the sigmoid they feed starts unsaturated."""
    nn.init.normal_(factor, std=factor.shape[0] ** -0.5)


def init_conv(conv: nn.Conv2d) -> None:
    nn.init.xavier_uniform_(conv.weight)
    nn.init.zeros_(conv.bias)


class TemporalAttention(nn.Module):
    """Temporal attention E = softmax(V_e sigmoid(((x^T U_1) U_2) (U_3 x) / sqrt(n) + b_e)) over
    steps.

    The bilinear score sums over the n nodes. Like the 1/sqrt(d) of dot-product attention, the
    1/sqrt(n) keeps its scale independent of the graph size; without it the scores of large
    graphs grow during training until the sigmoid saturates (zero gradient, a fixed gate).
    """

    def __init__(self, in_channels: int, num_nodes: int, seq_len: int):
        super().__init__()
        self.score_scale = num_nodes**-0.5
        self.u1 = nn.Parameter(torch.empty(num_nodes))
        self.u2 = nn.Parameter(torch.empty(in_channels, num_nodes))
        self.u3 = nn.Parameter(torch.empty(in_channels))
        self.bias = nn.Parameter(torch.empty(seq_len, seq_len))
        self.v = nn.Parameter(torch.empty(seq_len, seq_len))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        for factor in (self.u1, self.u2, self.u3):
            init_score_factor(factor)
        nn.init.xavier_uniform_(self.v)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        """x: (b, c, n, t) -> (b, t, s), normalized over t (the attended source steps)"""
        lhs = einsum(x, self.u1, "b c n t, n -> b t c")
        lhs = einsum(lhs, self.u2, "b t c, c n -> b t n")
        rhs = einsum(x, self.u3, "b c n t, c -> b n t")
        product = einsum(lhs, rhs, "b t n, b n s -> b t s") * self.score_scale
        scores = einsum(self.v, torch.sigmoid(product + self.bias), "t r, b r s -> b t s")
        return F.softmax(scores, dim=1)


class SpatialAttention(nn.Module):
    """Spatial attention S = softmax(V_s sigmoid(((x W_1) W_2) (W_3 x)^T / sqrt(t) + b_s)) over
    nodes; the score sums over the t steps (see `TemporalAttention` for the scaling)."""

    def __init__(self, in_channels: int, num_nodes: int, seq_len: int):
        super().__init__()
        self.score_scale = seq_len**-0.5
        self.w1 = nn.Parameter(torch.empty(seq_len))
        self.w2 = nn.Parameter(torch.empty(in_channels, seq_len))
        self.w3 = nn.Parameter(torch.empty(in_channels))
        self.bias = nn.Parameter(torch.empty(num_nodes, num_nodes))
        self.v = nn.Parameter(torch.empty(num_nodes, num_nodes))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        for factor in (self.w1, self.w2, self.w3):
            init_score_factor(factor)
        nn.init.xavier_uniform_(self.v)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        """x: (b, c, n, t) -> (b, n, m), normalized over n (the source nodes)"""
        lhs = einsum(x, self.w1, "b c n t, t -> b n c")
        lhs = einsum(lhs, self.w2, "b n c, c t -> b n t")
        rhs = einsum(x, self.w3, "b c n t, c -> b n t")
        product = einsum(lhs, rhs, "b n t, b m t -> b n m") * self.score_scale
        scores = einsum(self.v, torch.sigmoid(product + self.bias), "n r, b r m -> b n m")
        return F.softmax(scores, dim=1)


class ChebConvWithAttention(nn.Module):
    """Chebyshev graph convolution whose polynomials are reweighted by the spatial attention,
    out_m = relu(sum_k sum_n (T_k ⊙ S)_{nm} x_n Theta_k), applied to every time step.

    The attention does not depend on the step, so T_k ⊙ S is formed once per sample and the
    graph product covers all steps in one contraction."""

    def __init__(self, in_channels: int, out_channels: int, num_supports: int):
        super().__init__()
        self.theta = nn.Parameter(torch.empty(num_supports, in_channels, out_channels))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        for theta_k in self.theta:  # one (in, out) matrix per polynomial
            nn.init.xavier_uniform_(theta_k)

    def forward(self, x: Tensor, attention: Tensor, supports: Tensor) -> Tensor:
        """x: (b, c, n, t), attention: (b, n, m), supports: (b, k, n, m) -> (b, d, m, t)"""
        attended = supports * rearrange(attention, "b n m -> b 1 n m")
        h = einsum(x, attended, "b c n t, b k n m -> b k c m t")
        return F.relu(einsum(h, self.theta, "b k c m t, k c d -> b d m t"))


class ASTGCNBlock(nn.Module):
    """Temporal attention, spatial attention, attention-weighted Chebyshev convolution and a
    temporal convolution, plus a residual 1x1 convolution; ReLU, then LayerNorm over channels."""

    def __init__(
        self,
        in_channels: int,
        cheb_channels: int,
        time_channels: int,
        time_kernel_size: int,
        time_stride: int,
        num_supports: int,
        num_nodes: int,
        seq_len: int,
    ):
        super().__init__()
        self.temporal_attention = TemporalAttention(in_channels, num_nodes, seq_len)
        self.spatial_attention = SpatialAttention(in_channels, num_nodes, seq_len)
        self.cheb_conv = ChebConvWithAttention(in_channels, cheb_channels, num_supports)
        self.time_conv = nn.Conv2d(
            cheb_channels,
            time_channels,
            kernel_size=(1, time_kernel_size),
            stride=(1, time_stride),
            padding=(0, time_kernel_size // 2),
        )
        self.residual_conv = nn.Conv2d(
            in_channels, time_channels, kernel_size=1, stride=(1, time_stride)
        )
        self.norm = nn.LayerNorm(time_channels)
        init_conv(self.time_conv)
        init_conv(self.residual_conv)

    def forward(self, x: Tensor, supports: Tensor) -> Tensor:
        """x: (b, c, n, t), supports: (b, k, n, n) -> (b, d, n, t'), t' = ceil(t / stride)"""
        temporal = self.temporal_attention(x)
        x_attended = einsum(x, temporal, "b c n t, b t s -> b c n s")
        # The spatial attention reads the time-attended input; the graph convolution reads x.
        spatial = self.spatial_attention(x_attended)
        h = self.time_conv(self.cheb_conv(x, spatial, supports))
        h = F.relu(self.residual_conv(x) + h)
        h = self.norm(rearrange(h, "b d n t -> b n t d"))
        return rearrange(h, "b n t d -> b d n t")


class ASTGCN(BaseModel):
    """ASTGCN (Guo et al., AAAI 2019), "Attention Based Spatial-Temporal Graph Convolutional
    Networks for Traffic Flow Forecasting", https://doi.org/10.1609/aaai.v33i01.3301922.

    Reference code: https://github.com/guoshnBJTU/ASTGCN-r-pytorch

    Single-component variant (the recent component; no hourly / daily / weekly split), as in the
    reference code's ASTGCN-r. The Chebyshev polynomials T_0 .. T_{K-1} arrive as `supports`, so
    the polynomial order is set by the graph transform.

    JEPA split: the encoder is the shared time encoding and the stack of ASTGCN blocks; its
    latent is the block output of every node, (b, n, t', d) with t' = ceil(his_len / stride),
    so the predictor mixes the steps of each node. The decoder is the output convolution, which
    maps the t' steps and d channels of each node to the horizon.

    The attention parameters are tied to the window length (b_e, V_e are t x t, W_1, W_2 have a
    time axis), so encode only accepts windows of his_len steps; input and target windows of
    JEPA pretraining both have that length.

    Initialization: the reference code draws every 1-D parameter from U(0, 1), which makes the
    attention scores huge and of one sign and saturates the sigmoid of the attentions from the
    second block on (zero gradient, from which they never recover). Here the score factors
    U_1, U_2, U_3, W_1, W_2, W_3 are drawn from N(0, 1/fan_in) (see `init_score_factor`), the
    attention and convolution biases start at zero, LayerNorm at its default, and every other
    matrix (V_e, V_s, Theta_k, convolution kernels) is Xavier-uniform as in the reference code.

    Score scaling: the bilinear attention scores are divided by the square root of the length
    they sum over (nodes for the temporal attention, steps for the spatial one), as dot-product
    attention divides by sqrt(d). The paper has no such scaling; without it the scores grow
    during training until the sigmoid saturates and the attention becomes a fixed gate.
    """

    def __init__(
        self,
        num_blocks: int,
        cheb_channels: int,
        time_channels: int,
        time_kernel_size: int,
        time_stride: int,
        num_supports: int,
        **common,
    ):
        super().__init__(**common)
        if time_kernel_size % 2 == 0:
            raise ValueError("an odd time kernel with padding kernel // 2 keeps the length")
        self.latent_len = (self.his_len - 1) // time_stride + 1  # steps after the first block
        block = dict(
            cheb_channels=cheb_channels,
            time_channels=time_channels,
            time_kernel_size=time_kernel_size,
            num_supports=num_supports,
            num_nodes=self.num_nodes,
        )
        self.blocks = nn.ModuleList(
            [ASTGCNBlock(self.in_channels, time_stride=time_stride, seq_len=self.his_len, **block)]
            + [
                ASTGCNBlock(time_channels, time_stride=1, seq_len=self.latent_len, **block)
                for _ in range(num_blocks - 1)
            ]
        )
        self.predictor = self.make_predictor(num_tokens=self.latent_len, dim=time_channels)
        # Reads the latent steps as input channels and the latent width as the kernel width,
        # i.e. a linear map from each node's (t', d) latent to its pred_len x out_channels values.
        self.output_conv = nn.Conv2d(
            self.latent_len, self.pred_len * self.out_channels, kernel_size=(1, time_channels)
        )
        init_conv(self.output_conv)

    def encode(self, x: Tensor, x_mark: Tensor, supports: Tensor | None = None) -> Tensor:
        """(b, his_len, n, c), (b, his_len, f), (b, k, n, n) -> latent (b, n, t', d)"""
        h = rearrange(self.add_time(x, x_mark), "b t n c -> b c n t")
        for block in self.blocks:
            h = block(h, supports)
        return rearrange(h, "b d n t -> b n t d")

    def predict(self, z: Tensor) -> Tensor:
        """(b, n, t', d) -> (b, n, t', d), mixing the t' steps of each node"""
        return self.predictor(z)

    def decode(
        self,
        z: Tensor,
        x: Tensor,
        x_mark: Tensor,
        y_mark: Tensor,
        supports: Tensor | None = None,
    ) -> Tensor:
        """latent (b, n, t', d) -> (b, pred_len, n, out_channels)"""
        out = self.output_conv(rearrange(z, "b n t d -> b t n d"))  # (b, p * c, n, 1)
        return rearrange(out, "b (p c) n 1 -> b p n c", c=self.out_channels)
