import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.Embed import DataEmbedding_inverted
from .layers.SelfAttention_Family import AttentionLayer, FullAttention
from .layers.Transformer_EncDec import Encoder, EncoderLayer

from src.base.model import BaseModel


class iTransformer(BaseModel):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(
        self,
        d_model,
        embed,
        dropout,
        activation,
        e_layers,
        n_heads,
        d_ff,
        factor,
        freq,
        moving_avg,
        **args
    ):
        super(iTransformer, self).__init__(**args)
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            self.his_len,
            d_model,
            embed,
            freq,
            dropout,
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        # Decoder
        self.projection = nn.Linear(d_model, self.pred_len, bias=True)

    def forward(self, input_seq, input_features, target_features, *args, **kwargs):
        # Normalization from Non-stationary Transformer
        # means = x_enc.mean(1, keepdim=True).detach()
        # x_enc = x_enc - means
        # stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x_enc /= stdev

        _, _, N, _ = input_seq.shape
        input_seq = input_seq.squeeze(-1)

        # Embedding
        enc_out = self.enc_embedding(input_seq, input_features)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        # dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        # dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out.unsqueeze(-1)
