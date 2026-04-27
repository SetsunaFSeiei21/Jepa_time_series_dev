import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from src.base.model import BaseModel

from .layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from .layers.Autoformer_EncDec import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    my_Layernorm,
    series_decomp,
)
from .layers.Embed import DataEmbedding, DataEmbedding_wo_pos


class Autoformer(BaseModel):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    Paper link: https://openreview.net/pdf?id=I55UqU-M11y
    """

    def __init__(
        self,
        d_model,
        embed,
        dropout,
        activation,
        e_layers,
        d_layers,
        n_heads,
        d_ff,
        factor,
        freq,
        moving_avg,
        **args
    ):
        super(Autoformer, self).__init__(**args)

        # Decomp
        kernel_size = moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(
            self.node_num,
            d_model,
            embed,
            freq,
            dropout,
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
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
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model),
        )
        # Decoder
        self.dec_embedding = DataEmbedding_wo_pos(
            self.node_num,
            d_model,
            embed,
            freq,
            dropout,
        )
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            True,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    self.node_num,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, self.node_num, bias=True),
        )

    def forward(self, input_seq, input_features, target_features, *args, **kwargs):
        # decomp init
        B, T, N, _ = input_seq.shape
        input_seq = input_seq.squeeze(-1)
        mean = torch.mean(input_seq, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([B, self.pred_len, N], device=input_seq.device)
        seasonal_init, trend_init = self.decomp(input_seq)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.his_len :, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.his_len :, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(input_seq, input_features)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # dec
        dec_out = self.dec_embedding(
            seasonal_init, torch.concat([input_features, target_features], dim=1)
        )
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init
        )
        # final
        dec_out = trend_part + seasonal_part
        dec_out = dec_out[:, -self.pred_len :, :].unsqueeze(-1)
        return dec_out
