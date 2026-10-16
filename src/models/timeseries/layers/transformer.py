from torch import Tensor, nn

from .attention import ACTIVATIONS, FullAttention, ProjectedAttention


class TransformerEncoderLayer(nn.Module):
    """Post-norm Transformer encoder layer: full self-attention and a feed-forward network,
    each with dropout, a residual connection and LayerNorm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, activation: str):
        super().__init__()
        self.attention = ProjectedAttention(FullAttention(dropout), d_model, n_heads)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = ACTIVATIONS[activation]

    def forward(self, x: Tensor) -> Tensor:
        """(b, t, d) -> (b, t, d)"""
        x = self.norm1(x + self.dropout(self.attention(x)))
        y = self.dropout(self.linear2(self.dropout(self.activation(self.linear1(x)))))
        return self.norm2(x + y)
