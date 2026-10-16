from einops import rearrange
from torch import Tensor, nn

from src.models.base import BaseModel

from .layers.transformer import TransformerEncoderLayer


class iTransformer(BaseModel):
    """iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
    (Liu et al., ICLR 2024), https://arxiv.org/abs/2310.06625.
    Reference code: https://github.com/thuml/iTransformer.

    Inverts the Transformer: every node is one token, embedded linearly from its whole input
    window; self-attention runs across nodes, the feed-forward network within each token, and
    a linear head maps each token to the node's forecast.

    Time features use the shared encoding, appended to every node's channels, so each node's
    token is embedded from its data series and the time-encoding series of the window.

    encode:  (b, t, n, c) -> (b, n, d)   one token per node after the encoder and its LayerNorm
    predict: mixes the n node tokens (and the d channels), as the encoder's attention does
    decode:  (b, n, d) -> (b, p, n, c_out) linear head per token
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        e_layers: int,
        dropout: float,
        activation: str,
        **common,
    ):
        super().__init__(**common)
        self.embedding = nn.Linear(self.his_len * self.in_channels, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, activation)
            for _ in range(e_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.predictor = self.make_predictor(num_tokens=self.num_nodes, dim=d_model)
        self.projection = nn.Linear(d_model, self.pred_len * self.out_channels)

    def encode(self, x: Tensor, x_mark: Tensor, supports: Tensor | None = None) -> Tensor:
        """(b, t, n, c), (b, t, f) -> (b, n, d)"""
        h = self.add_time(x, x_mark)  # (b, t, n, in_channels)
        # Inverted embedding: each node's whole window (all channels) becomes one token.
        h = self.dropout(self.embedding(rearrange(h, "b t n c -> b n (c t)")))  # (b, n, d)
        for layer in self.layers:
            h = layer(h)  # attention across nodes
        return self.norm(h)

    def predict(self, z: Tensor) -> Tensor:
        """(b, n, d) -> (b, n, d)"""
        return self.predictor(z)

    def decode(
        self,
        z: Tensor,
        x: Tensor,
        x_mark: Tensor,
        y_mark: Tensor,
        supports: Tensor | None = None,
    ) -> Tensor:
        """(b, n, d) -> (b, p, n, c_out)"""
        y = self.projection(z)  # (b, n, c_out * p)
        return rearrange(y, "b n (c p) -> b p n c", c=self.out_channels)
