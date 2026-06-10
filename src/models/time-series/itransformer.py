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
        
        # JEPA predictor baseline:
        # iTransformer uses inverted embedding:
        #   input_seq: (B, T, N, 1) -> (B, T, N)
        #   enc_embedding -> (B, N, d_model)
        #
        # If input_features are used, DataEmbedding_inverted treats them as
        # additional tokens:
        #   Ex: (B, N + F, d_model)
        #
        # Therefore, unlike Autoformer/FEDformer, iTransformer should not use
        # Linear(his_len -> pred_len) here. The explicit time axis has already
        # been projected into d_model.
        self.d_model = d_model
        self.jepa_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    # def forward(self, input_seq, input_features, target_features, *args, **kwargs):
    #     # Normalization from Non-stationary Transformer
    #     # means = x_enc.mean(1, keepdim=True).detach()
    #     # x_enc = x_enc - means
    #     # stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
    #     # x_enc /= stdev

    #     _, _, N, _ = input_seq.shape
    #     input_seq = input_seq.squeeze(-1)

    #     # Embedding
    #     enc_out = self.enc_embedding(input_seq, input_features)
    #     enc_out, attns = self.encoder(enc_out, attn_mask=None)

    #     dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
    #     # De-Normalization from Non-stationary Transformer
    #     # dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
    #     # dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
    #     return dec_out.unsqueeze(-1)

    def _to_btn(self, seq, name="seq"):
        """
        Convert supported sequence layouts to iTransformer's internal layout.

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
                    f"iTransformer expects {name} last channel dim = 1, "
                    f"but got shape {tuple(seq.shape)}."
                )
            seq = seq.squeeze(-1)

        if seq.dim() != 3:
            raise ValueError(
                f"iTransformer expects {name} as (B,T,N,1), (B,T,N), or (B,N,T), "
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
            without time features:
                Ex / Ey: (B, N, d_model)

            with time features:
                Ex / Ey: (B, N + F, d_model)

        Note:
            DataEmbedding_inverted is initialized with c_in=self.his_len.
            Therefore this v0 implementation assumes the encoded sequence
            temporal length equals self.his_len. Current experiments use
            his_len == pred_len, so encode(label) is expected to be safe.
        """
        seq = self._to_btn(seq, name="seq")

        if seq.size(1) != self.his_len:
            raise ValueError(
                f"iTransformer encode expects temporal length == his_len={self.his_len}, "
                f"but got {seq.size(1)}. "
                "This v0 JEPA refactor assumes his_len == pred_len for label encoding."
            )

        enc_out = self.enc_embedding(seq, seq_features)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        return enc_out

    def predict(self, Ex):
        """
        Predict label embedding from input embedding.

        Ex:
            without time features:
                (B, N, d_model)

            with time features:
                (B, N + F, d_model)

        Ey_pred:
            same shape as Ex.
        """
        if Ex.dim() != 3:
            raise ValueError(
                f"iTransformer JEPA predictor expects Ex with shape "
                f"(B,N,d_model) or (B,N+F,d_model), but got {tuple(Ex.shape)}."
            )

        if Ex.size(-1) != self.d_model:
            raise ValueError(
                f"iTransformer JEPA predictor expects hidden dim d_model={self.d_model}, "
                f"but got {Ex.size(-1)}."
            )

        Ey_pred = self.jepa_predictor(Ex)
        return Ey_pred

    def decode(self, Ey_pred, input_seq=None, *args, **kwargs):
        """
        Decode predicted label embedding into forecasting value space.

        Ey_pred:
            without time features:
                (B, N, d_model)

            with time features:
                (B, N + F, d_model)

        Return:
            y_pred: (B, L, N, 1)
        """
        if Ey_pred.dim() != 3:
            raise ValueError(
                f"iTransformer decode expects Ey_pred with shape "
                f"(B,N,d_model) or (B,N+F,d_model), but got {tuple(Ey_pred.shape)}."
            )

        if Ey_pred.size(-1) != self.d_model:
            raise ValueError(
                f"iTransformer decode expects hidden dim d_model={self.d_model}, "
                f"but got {Ey_pred.size(-1)}."
            )

        # Original iTransformer output head:
        # Ey_pred: (B, token_num, d_model)
        # projection: (B, token_num, pred_len)
        # permute: (B, pred_len, token_num)
        # slice: (B, pred_len, node_num)
        # output: (B, pred_len, node_num, 1)
        dec_out = self.projection(Ey_pred).permute(0, 2, 1)
        dec_out = dec_out[:, :, : self.node_num]
        return dec_out.unsqueeze(-1)

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
            Ex      = encode(input_seq)
            Ey      = encode(label)
            Ey_pred = predict(Ex)
            return Ey, Ey_pred

        mode == "finetune":
            Ex      = encode(input_seq)
            Ey_pred = predict(Ex)
            y_pred  = decode(Ey_pred)
            return y_pred

        Default mode is finetune, so existing downstream training code can still
        call forward normally.

        Note:
            This JEPA finetune path is not identical to the original iTransformer
            forward. The original projection uses enc_out directly, while this
            JEPA version projects from Ey_pred.
        """
        if label is None:
            label = target_seq

        if label is None:
            label = kwargs.get("target_seq", None)

        if mode == "pretrain":
            if label is None:
                raise ValueError(
                    "iTransformer forward(mode='pretrain') requires label or target_seq."
                )

            Ex = self.encode(input_seq, input_features)
            Ey = self.encode(label, target_features)
            Ey_pred = self.predict(Ex)

            if Ey_pred.shape != Ey.shape:
                raise RuntimeError(
                    f"iTransformer JEPA shape mismatch: "
                    f"Ey_pred.shape={tuple(Ey_pred.shape)} vs Ey.shape={tuple(Ey.shape)}."
                )

            return Ey, Ey_pred

        if mode == "finetune":
            Ex = self.encode(input_seq, input_features)
            Ey_pred = self.predict(Ex)
            y_pred = self.decode(Ey_pred, input_seq=input_seq)
            return y_pred

        raise ValueError(f"Unsupported mode: {mode}")