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
        
        # JEPA predictor:
        # Ex: (B, T, d_model)
        # Ey: (B, L, d_model)
        # Only temporal length needs to be mapped: T -> L.
        self.d_model = d_model
        self.jepa_time_predictor = nn.Linear(self.his_len, self.pred_len)

    # def forward(self, input_seq, input_features, target_features, *args, **kwargs):
    #     # decomp init
    #     B, T, N, _ = input_seq.shape
    #     input_seq = input_seq.squeeze(-1)
    #     mean = torch.mean(input_seq, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
    #     zeros = torch.zeros([B, self.pred_len, N], device=input_seq.device)
    #     seasonal_init, trend_init = self.decomp(input_seq)
    #     # decoder input
    #     trend_init = torch.cat([trend_init[:, -self.his_len :, :], mean], dim=1)
    #     seasonal_init = torch.cat([seasonal_init[:, -self.his_len :, :], zeros], dim=1)
    #     # enc
    #     enc_out = self.enc_embedding(input_seq, input_features)
    #     enc_out, attns = self.encoder(enc_out, attn_mask=None)
    #     # dec
    #     dec_out = self.dec_embedding(
    #         seasonal_init, torch.concat([input_features, target_features], dim=1)
    #     )
    #     seasonal_part, trend_part = self.decoder(
    #         dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init
    #     )
    #     # final
    #     dec_out = trend_part + seasonal_part
    #     dec_out = dec_out[:, -self.pred_len :, :].unsqueeze(-1)
    #     return dec_out
    def _to_btn(self, seq, name="seq"):
        """
        Convert supported sequence layouts to Autoformer's internal layout.

        Supported:
            input_seq / label:
                (B, T, N, 1)  current dataloader format
                (B, T, N)     squeezed format
                (B, N, T)     JEPA-style format

        Return:
            seq: (B, T, N)
        """
        if seq is None:
            raise ValueError(f"{name} must not be None.")

        if seq.dim() == 4:
            if seq.size(-1) != 1:
                raise ValueError(
                    f"Autoformer expects {name} last channel dim = 1, "
                    f"but got shape {tuple(seq.shape)}."
                )
            seq = seq.squeeze(-1)

        if seq.dim() != 3:
            raise ValueError(
                f"Autoformer expects {name} as (B,T,N,1), (B,T,N), or (B,N,T), "
                f"but got shape {tuple(seq.shape)}."
            )

        # Existing project convention: (B, T, N)
        if seq.size(-1) == self.node_num:
            return seq

        # JEPA-style convention: (B, N, T)
        if seq.size(1) == self.node_num:
            return seq.transpose(1, 2)

        raise ValueError(
            f"Cannot infer node dimension for {name}. "
            f"Expected one dimension to equal node_num={self.node_num}, "
            f"but got shape {tuple(seq.shape)}."
        )

    def encode(self, seq, seq_features=None, *args, **kwargs):
        """
        Encode input or label into embedding space.

        input_seq:
            (B, T, N, 1) or (B, T, N) or (B, N, T)

        label / target_seq:
            (B, L, N, 1) or (B, L, N) or (B, N, L)

        Return:
            input encoding Ex: (B, T, d_model)
            label encoding Ey: (B, L, d_model)
        """
        seq = self._to_btn(seq, name="seq")
        enc_out = self.enc_embedding(seq, seq_features)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        return enc_out

    def predict(self, Ex):
        """
        Predict label embedding from input embedding.

        Ex:
            (B, T, d_model)

        Ey_pred:
            (B, L, d_model)
        """
        if Ex.dim() != 3:
            raise ValueError(
                f"Autoformer JEPA predictor expects Ex with shape (B,T,d_model), "
                f"but got {tuple(Ex.shape)}."
            )

        if Ex.size(1) != self.his_len:
            raise ValueError(
                f"Autoformer JEPA predictor was initialized with his_len={self.his_len}, "
                f"but got Ex temporal length {Ex.size(1)}."
            )

        # Linear acts on last dimension.
        # Ex:      (B, T, d_model)
        # x:       (B, d_model, T)
        # mapped:  (B, d_model, L)
        # Ey_pred: (B, L, d_model)
        x = Ex.transpose(1, 2)
        x = self.jepa_time_predictor(x)
        Ey_pred = x.transpose(1, 2)
        return Ey_pred

    def _build_decoder_inputs(self, input_seq, input_features, target_features):
        """
        Build Autoformer decoder seasonal/trend inputs.

        input_seq:
            (B, T, N, 1) or (B, T, N) or (B, N, T)

        input_features:
            (B, T, F)

        target_features:
            (B, L, F)

        Return:
            seasonal_init:   (B, T + L, N)
            trend_init:      (B, T + L, N)
            decoder_features:(B, T + L, F)
        """
        input_seq = self._to_btn(input_seq, name="input_seq")
        B, T, N = input_seq.shape

        if T < self.his_len:
            raise ValueError(
                f"Autoformer decode expects input temporal length >= his_len={self.his_len}, "
                f"but got T={T}."
            )

        context_len = self.his_len

        mean = torch.mean(input_seq, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros(
            [B, self.pred_len, N],
            device=input_seq.device,
            dtype=input_seq.dtype,
        )

        seasonal_init, trend_init = self.decomp(input_seq)

        trend_init = torch.cat(
            [trend_init[:, -context_len:, :], mean],
            dim=1,
        )

        seasonal_init = torch.cat(
            [seasonal_init[:, -context_len:, :], zeros],
            dim=1,
        )

        if input_features is None and target_features is None:
            decoder_features = None
        elif input_features is not None and target_features is not None:
            decoder_features = torch.cat(
                [input_features[:, -context_len:, :], target_features],
                dim=1,
            )
        else:
            raise ValueError(
                "Autoformer decode requires both input_features and target_features, "
                "or neither of them."
            )

        return seasonal_init, trend_init, decoder_features

    def decode(self, Ey_pred, input_seq=None, input_features=None, target_features=None):
        """
        Decode predicted label embedding into forecasting value space.

        Ey_pred:
            (B, L, d_model)

        Return:
            y_pred: (B, L, N, 1)
        """
        if Ey_pred.dim() != 3:
            raise ValueError(
                f"Autoformer decode expects Ey_pred with shape (B,L,d_model), "
                f"but got {tuple(Ey_pred.shape)}."
            )

        seasonal_init, trend_init, decoder_features = self._build_decoder_inputs(
            input_seq=input_seq,
            input_features=input_features,
            target_features=target_features,
        )

        dec_out = self.dec_embedding(seasonal_init, decoder_features)

        seasonal_part, trend_part = self.decoder(
            dec_out,
            Ey_pred,
            x_mask=None,
            cross_mask=None,
            trend=trend_init,
        )

        y_pred = trend_part + seasonal_part
        y_pred = y_pred[:, -self.pred_len :, :].unsqueeze(-1)
        return y_pred

    def forward(
        self,
        input_seq,
        input_features=None,
        target_features=None,
        target_seq=None,
        label=None,
        mode="finetune",
        *args,
        **kwargs,
    ):
        """
        mode == "pretrain":
            Ex      = encode(input_seq) -> (B, T, d_model)
            Ey      = encode(label)     -> (B, L, d_model)
            Ey_pred = predict(Ex)       -> (B, L, d_model)
            return Ey, Ey_pred

        mode == "finetune":
            Ex      = encode(input_seq) -> (B, T, d_model)
            Ey_pred = predict(Ex)       -> (B, L, d_model)
            y_pred  = decode(Ey_pred)   -> (B, L, N, 1)
            return y_pred

        Default mode is finetune, so existing downstream training code can still
        call forward normally.

        Note:
            This JEPA finetune path is not identical to the original Autoformer
            forward. The original decoder cross-attends to enc_out, while this
            JEPA version decodes from Ey_pred.
        """
        if label is None:
            label = target_seq

        if label is None:
            label = kwargs.get("target_seq", None)

        if mode == "pretrain":
            if label is None:
                raise ValueError(
                    "Autoformer forward(mode='pretrain') requires label or target_seq."
                )

            Ex = self.encode(input_seq, input_features)
            Ey = self.encode(label, target_features)
            Ey_pred = self.predict(Ex)

            if Ey_pred.shape != Ey.shape:
                raise RuntimeError(
                    f"Autoformer JEPA shape mismatch: "
                    f"Ey_pred.shape={tuple(Ey_pred.shape)} vs Ey.shape={tuple(Ey.shape)}."
                )

            return Ey.unsqueeze(2), Ey_pred.unsqueeze(2)

        if mode == "finetune":
            Ex = self.encode(input_seq, input_features)
            Ey_pred = self.predict(Ex)
            y_pred = self.decode(
                Ey_pred,
                input_seq=input_seq,
                input_features=input_features,
                target_features=target_features,
            )
            return y_pred

        raise ValueError(f"Unsupported mode: {mode}")