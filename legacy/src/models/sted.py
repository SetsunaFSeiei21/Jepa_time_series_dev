import torch
from einops import rearrange, repeat
from torch import nn

from src.base.model import BaseModel


class STED(BaseModel):
    def __init__(self, dim, depth, heads, mlp_dim, dropout, **kwargs):
        super(STED, self).__init__(**kwargs)
        self.enc_feat_embedding = nn.Linear(self.input_dim, dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=depth,
        )
        self.dec_feat_embedding = nn.Linear(self.input_dim - 1, dim)
        self.decoder_query = nn.Parameter(torch.randn(1, self.pred_len, dim))
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=depth,
        )
        self.proj = nn.Linear(dim, self.output_dim)

    def forward(self, input_seq, input_features, target_features, *args, **kwargs):
        # (b, t, n, f)
        assert target_features is not None, "target_features must be provided for STED"
        assert (
            input_seq.shape[2] == self.node_num
        ), f"input_seq shape mismatch: expected node_num {self.node_num}, got {input_seq.shape[2]}"
        input_seq = torch.concat(
            [input_seq, repeat(input_features, "b t f -> b t n f", n=self.node_num)],
            dim=-1,
        )
        input_seq = self.enc_feat_embedding(input_seq)
        input_seq = rearrange(input_seq, "b t n f -> (b n) t f")
        input_seq = self.encoder(input_seq)
        target_features = self.dec_feat_embedding(target_features)
        x_dec_query = self.decoder_query + target_features
        x_dec_query = repeat(x_dec_query, "b t f -> b t n f", n=self.node_num)
        x_dec_query = rearrange(x_dec_query, "b t n f -> (b n) t f")
        x_pred = self.decoder(x_dec_query, input_seq)
        x_pred = self.proj(x_pred)
        x_pred = rearrange(x_pred, "(b n) t f -> b t n f", n=self.node_num)
        return x_pred
