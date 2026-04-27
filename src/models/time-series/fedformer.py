import torch
import torch.nn as nn
import torch.nn.functional as F

from src.base.model import BaseModel

from .layers.AutoCorrelation import AutoCorrelationLayer
from .layers.Autoformer_EncDec import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    my_Layernorm,
    series_decomp,
)
from .layers.Embed import DataEmbedding
from .layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from .layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform


class FEDformer(BaseModel):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    Paper link: https://proceedings.mlr.press/v162/zhou22g.html
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
        version="fourier",
        mode_select="random",
        modes=32,
        **args
    ):
        super(FEDformer, self).__init__(**args)
        """
        version: str, for FEDformer, there are two versions to choose, options: [Fourier, Wavelets].
        mode_select: str, for FEDformer, there are two mode selection method, options: [random, low].
        modes: int, modes to be selected.
        """

        self.version = version
        self.mode_select = mode_select
        self.modes = modes

        # Decomp
        self.decomp = series_decomp(moving_avg)
        self.enc_embedding = DataEmbedding(
            self.node_num,
            d_model,
            embed,
            freq,
            dropout,
        )
        self.dec_embedding = DataEmbedding(
            self.node_num,
            d_model,
            embed,
            freq,
            dropout,
        )

        if self.version == "Wavelets":
            encoder_self_att = MultiWaveletTransform(ich=d_model, L=1, base="legendre")
            decoder_self_att = MultiWaveletTransform(ich=d_model, L=1, base="legendre")
            decoder_cross_att = MultiWaveletCross(
                in_channels=d_model,
                out_channels=d_model,
                seq_len_q=self.his_len // 2 + self.pred_len,
                seq_len_kv=self.his_len,
                modes=self.modes,
                ich=d_model,
                base="legendre",
                activation="tanh",
            )
        else:
            encoder_self_att = FourierBlock(
                in_channels=d_model,
                out_channels=d_model,
                n_heads=n_heads,
                seq_len=self.his_len,
                modes=self.modes,
                mode_select_method=self.mode_select,
            )
            decoder_self_att = FourierBlock(
                in_channels=d_model,
                out_channels=d_model,
                n_heads=n_heads,
                seq_len=self.his_len // 2 + self.pred_len,
                modes=self.modes,
                mode_select_method=self.mode_select,
            )
            decoder_cross_att = FourierCrossAttention(
                in_channels=d_model,
                out_channels=d_model,
                seq_len_q=self.his_len // 2 + self.pred_len,
                seq_len_kv=self.his_len,
                modes=self.modes,
                mode_select_method=self.mode_select,
                num_heads=n_heads,
            )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,  # instead of multi-head attention in transformer
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
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(decoder_self_att, d_model, n_heads),
                    AutoCorrelationLayer(decoder_cross_att, d_model, n_heads),
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
        seasonal_init, trend_init = self.decomp(input_seq)  # x - moving_avg, moving_avg
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.his_len :, :], mean], dim=1)
        seasonal_init = F.pad(
            seasonal_init[:, -self.his_len :, :], (0, 0, 0, self.pred_len)
        )
        # enc
        enc_out = self.enc_embedding(input_seq, input_features)
        dec_out = self.dec_embedding(
            seasonal_init, torch.concat([input_features, target_features], dim=1)
        )
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # dec
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init
        )
        # final
        dec_out = trend_part + seasonal_part
        dec_out = dec_out[:, -self.pred_len :, :].unsqueeze(-1)
        return dec_out
