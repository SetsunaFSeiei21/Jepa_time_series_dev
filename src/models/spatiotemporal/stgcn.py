import math

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from torch import Tensor, nn

from src.models.base import BaseModel

# Every ST-Conv block holds two unpadded temporal convolutions.
TEMPORAL_CONVS_PER_BLOCK = 2


class ChannelAlign(nn.Module):
    """Matches a residual input to the output width: a 1x1 convolution when it shrinks, zero
    channels appended when it grows, the identity otherwise. The convolution only exists when
    it is used."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pad = max(out_channels - in_channels, 0)
        self.conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels > out_channels
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        """(b, c_in, t, n) -> (b, c_out, t, n)"""
        if self.conv is not None:
            return self.conv(x)
        return F.pad(x, (0, 0, 0, 0, 0, self.pad))


class TemporalConv(nn.Module):
    """Gated (GLU) temporal convolution over `kernel_size` steps without padding, with the
    aligned input added to the linear half: (P + x) * sigmoid(Q)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.align = ChannelAlign(in_channels, out_channels)
        self.conv = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=(kernel_size, 1))

    def forward(self, x: Tensor) -> Tensor:
        """(b, c_in, t, n) -> (b, c_out, t - kernel_size + 1, n)"""
        residual = self.align(x)[:, :, self.kernel_size - 1 :]
        p, q = self.conv(x).chunk(2, dim=1)
        return (p + residual) * torch.sigmoid(q)


class ChebGraphConv(nn.Module):
    """Chebyshev graph convolution sum_k T_k(L) X W_k over T_0 .. T_{order-1} of the scaled
    Laplacian L, applied to the aligned input, which is also added as residual."""

    def __init__(self, in_channels: int, out_channels: int, order: int):
        super().__init__()
        assert order >= 1, f"chebyshev_order must be positive, got {order}"
        self.order = order
        self.align = ChannelAlign(in_channels, out_channels)
        self.weight = nn.Parameter(torch.empty(order, out_channels, out_channels))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        """Reference-code scheme: kaiming uniform weight, bias uniform in +-1/sqrt(fan_in)."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor, laplacian: Tensor) -> Tensor:
        """x: (b, c_in, t, n), laplacian: (b, n, n) -> (b, c_out, t, n)"""
        x = self.align(x)
        h = rearrange(x, "b c t n -> b t n c")
        terms = [h]  # T_k(L) X via T_k = 2 L T_{k-1} - T_{k-2}
        for k in range(1, self.order):
            propagated = einsum(laplacian, terms[-1], "b n m, b t m c -> b t n c")
            terms.append(propagated if k == 1 else 2 * propagated - terms[-2])
        h = einsum(torch.stack(terms), self.weight, "k b t n c, k c d -> b t n d") + self.bias
        return rearrange(h, "b t n d -> b d t n") + x


class STConvBlock(nn.Module):
    """Temporal conv -> Chebyshev graph conv -> ReLU -> temporal conv -> LayerNorm over
    (nodes, channels) -> dropout (the "sandwich" structure of the paper)."""

    def __init__(
        self,
        in_channels: int,
        channels: list[int],
        num_nodes: int,
        temporal_kernel_size: int,
        chebyshev_order: int,
        dropout: float,
    ):
        super().__init__()
        temporal_in, spatial, temporal_out = channels
        self.temporal_conv_1 = TemporalConv(in_channels, temporal_in, temporal_kernel_size)
        self.graph_conv = ChebGraphConv(temporal_in, spatial, chebyshev_order)
        self.temporal_conv_2 = TemporalConv(spatial, temporal_out, temporal_kernel_size)
        self.norm = nn.LayerNorm([num_nodes, temporal_out])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, laplacian: Tensor) -> Tensor:
        """x: (b, c_in, t, n), laplacian: (b, n, n) -> (b, c_out, t - 2 (kernel_size - 1), n)"""
        h = self.temporal_conv_1(x)
        h = F.relu(self.graph_conv(h, laplacian))
        h = self.temporal_conv_2(h)
        h = rearrange(self.norm(rearrange(h, "b c t n -> b t n c")), "b t n c -> b c t n")
        return self.dropout(h)


class OutputBlock(nn.Module):
    """Temporal conv over all remaining steps, LayerNorm over (nodes, channels), then a
    two-layer MLP producing every prediction step at once."""

    def __init__(
        self,
        in_channels: int,
        channels: list[int],
        num_steps: int,
        num_nodes: int,
        out_features: int,
    ):
        super().__init__()
        conv_channels, hidden_channels = channels
        self.temporal_conv = TemporalConv(in_channels, conv_channels, num_steps)
        self.norm = nn.LayerNorm([num_nodes, conv_channels])
        self.fc_1 = nn.Linear(conv_channels, hidden_channels)
        self.fc_2 = nn.Linear(hidden_channels, out_features)

    def forward(self, x: Tensor) -> Tensor:
        """(b, c, num_steps, n) -> (b, 1, n, out_features)"""
        h = rearrange(self.temporal_conv(x), "b c t n -> b t n c")
        return self.fc_2(F.relu(self.fc_1(self.norm(h))))


class STGCN(BaseModel):
    """Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic
    Forecasting (Yu et al., IJCAI 2018), https://arxiv.org/abs/1709.04875.

    Chebyshev variant (STGCN(ChebGraphConv)) on the scaled Laplacian, which arrives as the
    single support. Every temporal convolution is unpadded, so each ST-Conv block shortens the
    window by 2 (temporal_kernel_size - 1) steps and the output block convolves over all
    remaining steps.

    - encode: shared time encoding appended to the input, then the ST-Conv blocks; the latent
      is (b, n, remaining_steps, d) with d the width of the last block.
    - predict: LatentPredictor mixing the remaining time steps of each node.
    - decode: the output block, which emits all pred_len steps of a node at once.

    Reference code: https://github.com/hazdzz/STGCN
    """

    def __init__(
        self,
        temporal_kernel_size: int,
        chebyshev_order: int,
        num_blocks: int,
        block_channels: list[int],
        head_channels: list[int],
        dropout: float,
        num_supports: int,
        **common,
    ):
        super().__init__(**common)
        assert num_supports == 1, f"STGCN takes the scaled Laplacian only, got {num_supports}"
        remaining_steps = self.his_len - TEMPORAL_CONVS_PER_BLOCK * num_blocks * (
            temporal_kernel_size - 1
        )
        assert remaining_steps >= 1, (
            f"his_len={self.his_len} is too short for {num_blocks} blocks with "
            f"temporal_kernel_size={temporal_kernel_size}"
        )

        widths = [self.in_channels] + [block_channels[-1]] * num_blocks
        self.blocks = nn.ModuleList(
            STConvBlock(
                widths[i],
                block_channels,
                self.num_nodes,
                temporal_kernel_size,
                chebyshev_order,
                dropout,
            )
            for i in range(num_blocks)
        )
        self.predictor = self.make_predictor(num_tokens=remaining_steps, dim=widths[-1])
        self.output = OutputBlock(
            widths[-1],
            head_channels,
            remaining_steps,
            self.num_nodes,
            self.pred_len * self.out_channels,
        )

    def encode(self, x: Tensor, x_mark: Tensor, supports: Tensor | None = None) -> Tensor:
        """x: (b, t, n, c), x_mark: (b, t, f), supports: (b, 1, n, n) -> (b, n, l, d)"""
        laplacian = supports[:, 0]  # (b, n, n)
        h = rearrange(self.add_time(x, x_mark), "b t n c -> b c t n")
        for block in self.blocks:
            h = block(h, laplacian)
        # remaining steps of each node as the tokens the predictor mixes
        return rearrange(h, "b d l n -> b n l d")

    def predict(self, z: Tensor) -> Tensor:
        """(b, n, l, d) -> (b, n, l, d)"""
        return self.predictor(z)

    def decode(
        self,
        z: Tensor,
        x: Tensor,
        x_mark: Tensor,
        y_mark: Tensor,
        supports: Tensor | None = None,
    ) -> Tensor:
        """z: (b, n, l, d) -> (b, pred_len, n, out_channels)"""
        h = self.output(rearrange(z, "b n l d -> b d l n"))  # (b, 1, n, p * c)
        return rearrange(h, "b 1 n (p c) -> b p n c", c=self.out_channels)
