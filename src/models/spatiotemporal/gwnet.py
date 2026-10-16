import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from src.models.base import BaseModel
from src.models.layers import DiffusionConv


class WaveNetLayer(nn.Module):
    """Gated dilated temporal convolution followed by graph diffusion, with residual and skip
    outputs. The last layer only feeds the skip path, so it has no graph convolution or residual
    branch (the original computes them and discards the result).

    The residual output is normalized with a LayerNorm over the channels of each (node, step),
    where the original uses BatchNorm2d: JEPA compares the online encoder (train mode) with an
    EMA target encoder (eval mode) on micro-batches of a few samples, so the normalization must
    not depend on batch statistics.
    """

    def __init__(
        self,
        residual_channels: int,
        dilation_channels: int,
        skip_channels: int,
        kernel_size: int,
        dilation: int,
        num_supports: int,
        diffusion_steps: int,
        dropout: float,
        residual_output: bool,
    ):
        super().__init__()
        conv = dict(kernel_size=(1, kernel_size), dilation=dilation)
        self.filter_conv = nn.Conv2d(residual_channels, dilation_channels, **conv)
        self.gate_conv = nn.Conv2d(residual_channels, dilation_channels, **conv)
        self.skip_conv = nn.Conv2d(dilation_channels, skip_channels, kernel_size=1)
        self.residual_output = residual_output
        if residual_output:
            self.graph_conv = DiffusionConv(
                dilation_channels, residual_channels, num_supports, diffusion_steps, dropout
            )
            self.norm = nn.LayerNorm(residual_channels)

    def forward(self, x: Tensor, supports: Tensor) -> tuple[Tensor | None, Tensor]:
        """x: (b, c_res, n, t), supports: (b, k, n, n)
        -> residual (b, c_res, n, t') or None, skip (b, c_skip, n, t')"""
        h = torch.tanh(self.filter_conv(x)) * torch.sigmoid(self.gate_conv(x))
        skip = self.skip_conv(h)
        if not self.residual_output:
            return None, skip
        h = self.graph_conv(rearrange(h, "b c n t -> b t n c"), supports)
        # The residual keeps its gradient: it is the path from every layer back to start_conv.
        residual = rearrange(x[..., -h.shape[1] :], "b c n t -> b t n c")
        h = self.norm(h + residual)
        return rearrange(h, "b t n c -> b c n t"), skip


class GWNET(BaseModel):
    """Graph WaveNet for Deep Spatial-Temporal Graph Modeling (Wu et al., IJCAI 2019),
    https://arxiv.org/abs/1906.00121. Reference code: https://github.com/nnzhan/Graph-WaveNet

    Stacked layers of gated dilated temporal convolutions and diffusion graph convolutions over
    the given supports plus a learned adaptive adjacency; the skip outputs of all layers are
    summed and an MLP head maps them to the forecast.

    encode:  window -> summed skip outputs, latent (b, n, t_out, skip_channels), where t_out is
             the number of steps left after the receptive field (1 when his_len <= receptive
             field, his_len - receptive_field + 1 otherwise)
    predict: LatentPredictor mixing the t_out steps of each node and the skip channels
    decode:  ReLU, an MLP over the t_out steps when t_out > 1, then the original output head
    """

    def __init__(
        self,
        residual_channels: int,
        dilation_channels: int,
        skip_channels: int,
        end_channels: int,
        kernel_size: int,
        blocks: int,
        layers_per_block: int,
        diffusion_steps: int,
        dropout: float,
        adaptive_adj: bool,
        adaptive_embedding_dim: int,
        time_mlp_hidden: list[int],
        gradient_checkpointing: bool,
        num_supports: int = 0,
        **common,
    ):
        super().__init__(**common)
        self.adaptive_adj = adaptive_adj
        self.gradient_checkpointing = gradient_checkpointing
        total_supports = num_supports + int(adaptive_adj)
        assert total_supports > 0, "GWNET needs a graph_transform or adaptive_adj"
        if adaptive_adj:
            self.source_embedding = nn.Parameter(
                torch.randn(self.num_nodes, adaptive_embedding_dim)
            )
            self.target_embedding = nn.Parameter(
                torch.randn(adaptive_embedding_dim, self.num_nodes)
            )

        self.start_conv = nn.Conv2d(self.in_channels, residual_channels, kernel_size=1)
        dilations = [2**i for _ in range(blocks) for i in range(layers_per_block)]
        self.layers = nn.ModuleList(
            WaveNetLayer(
                residual_channels,
                dilation_channels,
                skip_channels,
                kernel_size,
                dilation,
                total_supports,
                diffusion_steps,
                dropout,
                residual_output=i < len(dilations) - 1,
            )
            for i, dilation in enumerate(dilations)
        )
        self.receptive_field = 1 + (kernel_size - 1) * sum(dilations)
        # Shorter windows are left-padded to the receptive field, so at least one step is left.
        self.latent_steps = max(self.his_len, self.receptive_field) - self.receptive_field + 1
        self.predictor = self.make_predictor(num_tokens=self.latent_steps, dim=skip_channels)

        # Inputs longer than the receptive field leave several steps; an MLP over time reduces
        # them to the single step the output head expects.
        self.time_mlp = None
        if self.latent_steps > 1:
            dims = [self.latent_steps, *time_mlp_hidden]
            mlp = []
            for d_in, d_out in zip(dims[:-1], dims[1:], strict=True):
                mlp += [nn.Linear(d_in, d_out), nn.ReLU()]
            self.time_mlp = nn.Sequential(*mlp, nn.Linear(dims[-1], 1))

        # 1x1 convolutions of the original output head, written as linear layers on the
        # channels-last latent.
        self.end_conv_1 = nn.Linear(skip_channels, end_channels)
        self.end_conv_2 = nn.Linear(end_channels, self.pred_len * self.out_channels)

    def _supports(self, supports: Tensor | None, batch_size: int) -> Tensor:
        """Graph supports plus the learned adaptive adjacency: (b, k_total, n, n)."""
        all_supports = [] if supports is None else [supports]
        if self.adaptive_adj:
            adaptive = F.softmax(F.relu(self.source_embedding @ self.target_embedding), dim=1)
            all_supports.append(repeat(adaptive, "n m -> b 1 n m", b=batch_size))
        return torch.cat(all_supports, dim=1)

    def encode(self, x: Tensor, x_mark: Tensor, supports: Tensor | None = None) -> Tensor:
        """(b, t, n, c), (b, t, f), (b, k, n, n) -> (b, n, latent_steps, skip_channels)"""
        h = rearrange(self.add_time(x, x_mark), "b t n c -> b c n t")
        h = F.pad(h, (max(self.receptive_field - h.shape[-1], 0), 0))
        h = self.start_conv(h)  # (b, residual_channels, n, t)
        supports = self._supports(supports, x.shape[0])

        skip = None
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                # Recomputes the layer in the backward pass instead of storing its activations
                # (the RNG state is restored, so dropout masks and results are identical).
                h, s = checkpoint(layer, h, supports, use_reentrant=False)
            else:
                h, s = layer(h, supports)
            skip = s if skip is None else s + skip[..., -s.shape[-1] :]
        return rearrange(skip, "b d n t -> b n t d")

    def predict(self, z: Tensor) -> Tensor:
        """(b, n, latent_steps, skip_channels) -> same shape"""
        return self.predictor(z)

    def decode(
        self,
        z: Tensor,
        x: Tensor,
        x_mark: Tensor,
        y_mark: Tensor,
        supports: Tensor | None = None,
    ) -> Tensor:
        """(b, n, latent_steps, skip_channels) -> (b, pred_len, n, out_channels)"""
        # The ReLU of the original head acts on the skip sum, before the time MLP: after the
        # MLP's single-output layer it would zero the output (and every gradient) whenever that
        # layer starts with the same sign for all nodes and channels.
        h = F.relu(z)
        if self.time_mlp is None:
            h = rearrange(h, "b n 1 d -> b n d")
        else:
            h = rearrange(self.time_mlp(rearrange(h, "b n t d -> b n d t")), "b n d 1 -> b n d")
        h = self.end_conv_2(F.relu(self.end_conv_1(h)))  # (b, n, pred_len * out_channels)
        return rearrange(h, "b n (p c) -> b p n c", c=self.out_channels)
