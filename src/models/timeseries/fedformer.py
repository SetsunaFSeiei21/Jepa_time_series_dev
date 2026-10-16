from einops import rearrange
from torch import Tensor

from src.models.base import BaseModel

from .layers import (
    DecompositionTransformer,
    FourierBlock,
    FourierCrossAttention,
    MultiWaveletCross,
    MultiWaveletTransform,
    label_length,
)

VERSIONS = ("fourier", "wavelets")


class FEDformer(BaseModel):
    """FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting
    (Zhou et al., ICML 2022), https://arxiv.org/abs/2201.12740.
    Reference code: https://github.com/MAZiqing/FEDformer.

    Autoformer's decomposition encoder-decoder (see `DecompositionTransformer`) with frequency
    enhanced blocks instead of auto-correlation, on a Fourier (`version: fourier`) or Legendre
    multiwavelet (`version: wavelets`) basis. As in the reference, the encoder layers share one
    block, and so do the decoder layers (one self- and one cross-attention block). The Fourier
    modes each block keeps are buffers of the state dict, so a checkpoint restores them.

    encode:  (b, t, n, c) -> (b, t, d)   embedding and encoder layers over the window; one
                                         token per time step
    predict: mixes the t time-step tokens (and the d channels)
    decode:  the original decoder, initialized from the input window and the future time
             features, attending to the predicted latent in place of the encoder output

    custom_time_features: the decoder consumes the time features of the predicted steps
    (`y_mark`), which the shared encoding of the input steps cannot provide. They are embedded
    linearly per token (TSLib's `timeF`).
    """

    custom_time_features = True

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        e_layers: int,
        d_layers: int,
        moving_avg: int,
        label_ratio: float,
        embedding_kernel_size: int,
        trend_kernel_size: int,
        positional_embedding: bool,
        dropout: float,
        activation: str,
        version: str,
        modes: int,
        mode_select: str,
        cross_activation: str,
        wavelet_order: int,
        wavelet_modes: int,
        wavelet_channels: int,
        wavelet_cross_channels: int,
        wavelet_blocks: int,
        wavelet_skip_levels: int,
        wavelet_cross_skip_levels: int,
        **common,
    ):
        super().__init__(**common)
        assert self.in_channels == self.out_channels, "the trend is forecast in the input space"
        if version not in VERSIONS:
            raise ValueError(f"version must be one of {VERSIONS}, got {version!r}")
        label_len = label_length(self.his_len, label_ratio)
        decoder_len = label_len + self.pred_len  # the decoder runs over label + predicted steps
        if version == "fourier":
            encoder_block = FourierBlock(d_model, n_heads, self.his_len, modes, mode_select)
            decoder_block = FourierBlock(d_model, n_heads, decoder_len, modes, mode_select)
            cross_block = FourierCrossAttention(
                d_model, n_heads, decoder_len, self.his_len, modes, mode_select, cross_activation
            )
        else:

            def wavelet_transform():
                return MultiWaveletTransform(
                    d_model,
                    wavelet_order,
                    wavelet_modes,
                    wavelet_channels,
                    wavelet_blocks,
                    wavelet_skip_levels,
                )

            encoder_block, decoder_block = wavelet_transform(), wavelet_transform()
            cross_block = MultiWaveletCross(
                d_model,
                wavelet_order,
                wavelet_cross_channels,
                modes,
                wavelet_cross_skip_levels,
                cross_activation,
            )
        self.network = DecompositionTransformer(
            num_variates=self.num_nodes * self.in_channels,
            num_time_features=self.num_time_features,
            label_len=label_len,
            pred_len=self.pred_len,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            e_layers=e_layers,
            d_layers=d_layers,
            moving_avg=moving_avg,
            embedding_kernel_size=embedding_kernel_size,
            trend_kernel_size=trend_kernel_size,
            positional_embedding=positional_embedding,
            dropout=dropout,
            activation=activation,
            encoder_attention=lambda: encoder_block,
            decoder_self_attention=lambda: decoder_block,
            decoder_cross_attention=lambda: cross_block,
            max_len=max(self.his_len, decoder_len),
        )
        self.predictor = self.make_predictor(num_tokens=self.his_len, dim=d_model)

    def encode(self, x: Tensor, x_mark: Tensor, supports: Tensor | None = None) -> Tensor:
        """x (b, t, n, c), x_mark (b, t, f) -> (b, t, d)"""
        return self.network.encode(rearrange(x, "b t n c -> b t (n c)"), x_mark)

    def predict(self, z: Tensor) -> Tensor:
        """(b, t, d) -> (b, t, d)"""
        return self.predictor(z)

    def decode(
        self,
        z: Tensor,
        x: Tensor,
        x_mark: Tensor,
        y_mark: Tensor,
        supports: Tensor | None = None,
    ) -> Tensor:
        """z (b, t, d), x (b, t, n, c), x_mark (b, t, f), y_mark (b, p, f) -> (b, p, n, c)"""
        y = self.network.decode(z, rearrange(x, "b t n c -> b t (n c)"), x_mark, y_mark)
        return rearrange(y, "b p (n c) -> b p n c", n=self.num_nodes)
