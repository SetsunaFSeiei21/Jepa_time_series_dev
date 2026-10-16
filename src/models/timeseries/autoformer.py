from einops import rearrange
from torch import Tensor

from src.models.base import BaseModel

from .layers import AutoCorrelation, DecompositionTransformer, label_length


class Autoformer(BaseModel):
    """Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series
    Forecasting (Wu et al., NeurIPS 2021), https://arxiv.org/abs/2106.13008.
    Reference code: https://github.com/thuml/Autoformer.

    Every node-channel series is a variate; all variates of a step form one token. Encoder and
    decoder use auto-correlation instead of self-attention and progressively decompose the
    series into seasonal part and trend (see `DecompositionTransformer`).

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
        factor: float,
        moving_avg: int,
        label_ratio: float,
        embedding_kernel_size: int,
        trend_kernel_size: int,
        dropout: float,
        activation: str,
        **common,
    ):
        super().__init__(**common)
        assert self.in_channels == self.out_channels, "the trend is forecast in the input space"
        label_len = label_length(self.his_len, label_ratio)
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
            positional_embedding=False,
            dropout=dropout,
            activation=activation,
            encoder_attention=lambda: AutoCorrelation(factor),
            decoder_self_attention=lambda: AutoCorrelation(factor),
            decoder_cross_attention=lambda: AutoCorrelation(factor),
            max_len=max(self.his_len, label_len + self.pred_len),
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
