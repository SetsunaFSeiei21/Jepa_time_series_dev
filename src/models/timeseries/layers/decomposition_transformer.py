"""Encoder-decoder with progressive series decomposition, shared by Autoformer and FEDformer.

Reference: https://github.com/thuml/Autoformer (layers/Autoformer_EncDec.py, models/Autoformer.py).
Shape letters: `v` variate (every node-channel series of the input), `p` predicted step.
"""

from collections.abc import Callable

import torch
from einops import rearrange, reduce, repeat
from torch import Tensor, nn

from src.models.layers import SeriesDecomposition

from .attention import ACTIVATIONS, ProjectedAttention
from .embedding import DataEmbedding


def label_length(his_len: int, label_ratio: float) -> int:
    """Number of input steps the decoder re-reads: the last `label_ratio` of the window."""
    label_len = int(his_len * label_ratio)
    assert 1 <= label_len <= his_len, f"label_ratio={label_ratio} gives label_len={label_len}"
    return label_len


class _Decomposition(nn.Module):
    """SeriesDecomposition of (b, t, d) sequences."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.decomposition = SeriesDecomposition(kernel_size)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """(b, t, d) -> seasonal, trend, both (b, t, d)"""
        seasonal, trend = self.decomposition(rearrange(x, "b t d -> b t 1 d"))
        return rearrange(seasonal, "b t 1 d -> b t d"), rearrange(trend, "b t 1 d -> b t d")


class SeasonalLayerNorm(nn.Module):
    """LayerNorm followed by removing the mean over time (for the seasonal part)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """(b, t, d) -> (b, t, d)"""
        x = self.layer_norm(x)
        return x - reduce(x, "b t d -> b 1 d", "mean")


class _FeedForward(nn.Module):
    """Position-wise two-layer MLP without biases, dropout after each layer."""

    def __init__(self, d_model: int, d_ff: int, dropout: float, activation: str):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = ACTIVATIONS[activation]

    def forward(self, x: Tensor) -> Tensor:
        """(b, t, d) -> (b, t, d)"""
        return self.dropout(self.linear2(self.dropout(self.activation(self.linear1(x)))))


class DecompositionEncoderLayer(nn.Module):
    """Self-attention and feed-forward, each followed by removing the trend."""

    def __init__(
        self,
        attention: ProjectedAttention,
        d_model: int,
        d_ff: int,
        moving_avg: int,
        dropout: float,
        activation: str,
    ):
        super().__init__()
        self.attention = attention
        self.feed_forward = _FeedForward(d_model, d_ff, dropout, activation)
        self.decomposition1 = _Decomposition(moving_avg)
        self.decomposition2 = _Decomposition(moving_avg)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """(b, t, d) -> (b, t, d)"""
        x, _ = self.decomposition1(x + self.dropout(self.attention(x)))
        x, _ = self.decomposition2(x + self.feed_forward(x))
        return x


class DecompositionDecoderLayer(nn.Module):
    """Self-attention, cross-attention and feed-forward on the seasonal part; the trends removed
    after each are summed and projected to the variates by a circular convolution."""

    def __init__(
        self,
        self_attention: ProjectedAttention,
        cross_attention: ProjectedAttention,
        d_model: int,
        num_variates: int,
        d_ff: int,
        moving_avg: int,
        trend_kernel_size: int,
        dropout: float,
        activation: str,
    ):
        super().__init__()
        assert trend_kernel_size % 2 == 1, f"trend_kernel_size must be odd, got {trend_kernel_size}"
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = _FeedForward(d_model, d_ff, dropout, activation)
        self.decomposition1 = _Decomposition(moving_avg)
        self.decomposition2 = _Decomposition(moving_avg)
        self.decomposition3 = _Decomposition(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.trend_projection = nn.Conv1d(
            d_model,
            num_variates,
            trend_kernel_size,
            padding=trend_kernel_size // 2,
            padding_mode="circular",
            bias=False,
        )

    def forward(self, x: Tensor, cross: Tensor) -> tuple[Tensor, Tensor]:
        """x (b, t, d), cross (b, s, d) -> seasonal (b, t, d), trend (b, t, v)"""
        x, trend1 = self.decomposition1(x + self.dropout(self.self_attention(x)))
        x, trend2 = self.decomposition2(x + self.dropout(self.cross_attention(x, cross)))
        x, trend3 = self.decomposition3(x + self.feed_forward(x))
        trend = rearrange(trend1 + trend2 + trend3, "b t d -> b d t")
        return x, rearrange(self.trend_projection(trend), "b v t -> b t v")


class DecompositionTransformer(nn.Module):
    """Autoformer's encoder-decoder, split into `encode` and `decode`.

    `encode` embeds the input window and refines it with the encoder layers. `decode` splits the
    input window into seasonal part and trend; the decoder starts from the seasonal part of the
    last `label_len` input steps followed by zeros and from their trend followed by the input
    mean, refines the seasonal part with attention over the encoded window (the `cross`
    sequence) and accumulates the trends removed in every layer.

    The attention mechanisms come from factories called once per layer; a factory may return
    the same module every time to share it between layers (FEDformer).
    """

    def __init__(
        self,
        num_variates: int,
        num_time_features: int,
        label_len: int,
        pred_len: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        e_layers: int,
        d_layers: int,
        moving_avg: int,
        embedding_kernel_size: int,
        trend_kernel_size: int,
        positional_embedding: bool,
        dropout: float,
        activation: str,
        encoder_attention: Callable[[], nn.Module],
        decoder_self_attention: Callable[[], nn.Module],
        decoder_cross_attention: Callable[[], nn.Module],
        max_len: int,
    ):
        super().__init__()
        self.label_len = label_len
        self.pred_len = pred_len
        self.decomposition = SeriesDecomposition(moving_avg)

        def embedding():
            return DataEmbedding(
                num_variates,
                num_time_features,
                d_model,
                embedding_kernel_size,
                max_len=max_len,
                positional=positional_embedding,
                dropout=dropout,
            )

        self.encoder_embedding = embedding()
        self.decoder_embedding = embedding()
        self.encoder_layers = nn.ModuleList(
            DecompositionEncoderLayer(
                ProjectedAttention(encoder_attention(), d_model, n_heads),
                d_model,
                d_ff,
                moving_avg,
                dropout,
                activation,
            )
            for _ in range(e_layers)
        )
        self.encoder_norm = SeasonalLayerNorm(d_model)
        self.decoder_layers = nn.ModuleList(
            DecompositionDecoderLayer(
                ProjectedAttention(decoder_self_attention(), d_model, n_heads),
                ProjectedAttention(decoder_cross_attention(), d_model, n_heads),
                d_model,
                num_variates,
                d_ff,
                moving_avg,
                trend_kernel_size,
                dropout,
                activation,
            )
            for _ in range(d_layers)
        )
        self.decoder_norm = SeasonalLayerNorm(d_model)
        self.projection = nn.Linear(d_model, num_variates)

    def encode(self, x: Tensor, x_mark: Tensor) -> Tensor:
        """x (b, t, v), x_mark (b, t, f) -> encoded window (b, t, d)"""
        h = self.encoder_embedding(x, x_mark)
        for layer in self.encoder_layers:
            h = layer(h)
        return self.encoder_norm(h)

    def decode(self, cross: Tensor, x: Tensor, x_mark: Tensor, y_mark: Tensor) -> Tensor:
        """cross (b, s, d), x (b, t, v), x_mark (b, t, f), y_mark (b, p, f) -> (b, p, v)"""
        seasonal, trend = self.decomposition(rearrange(x, "b t v -> b t 1 v"))
        seasonal = rearrange(seasonal[:, -self.label_len :], "b t 1 v -> b t v")
        trend = rearrange(trend[:, -self.label_len :], "b t 1 v -> b t v")
        mean = repeat(reduce(x, "b t v -> b v", "mean"), "b v -> b p v", p=self.pred_len)
        # decoder inputs over the last label_len input steps and the predicted steps
        seasonal = torch.cat([seasonal, torch.zeros_like(mean)], dim=1)  # (b, label_len + p, v)
        trend = torch.cat([trend, mean], dim=1)  # (b, label_len + p, v)
        marks = torch.cat([x_mark[:, -self.label_len :], y_mark], dim=1)

        h = self.decoder_embedding(seasonal, marks)  # (b, label_len + p, d)
        for layer in self.decoder_layers:
            h, residual_trend = layer(h, cross)
            trend = trend + residual_trend
        seasonal = self.projection(self.decoder_norm(h))
        return (trend + seasonal)[:, -self.pred_len :]
