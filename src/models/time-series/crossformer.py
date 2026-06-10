# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange, repeat

# from .layers.Crossformer_EncDec import Encoder
# from .layers.Crossformer_EncDec import Decoder
# from .layers.Crossformer_Attn import FullAttention, AttentionLayer, TwoStageAttentionLayer
# from .layers.DSW_Embedding import DSW_embedding
# from math import ceil

# from src.base.model import BaseModel

# class Crossformer(BaseModel):
#     def __init__(
#         self,
#         seg_len,
#         win_size=4,
#         factor=10,
#         d_model=64,
#         d_ff=128,
#         n_heads=4,
#         e_layers=3,
#         dropout=0.0,
#         baseline=False,
#         device=None,
#         **args,
#     ):
#         super(Crossformer, self).__init__(**args)

#         self.data_dim = self.node_num
#         self.in_len = self.his_len
#         self.out_len = self.pred_len
#         self.seg_len = seg_len
#         self.merge_win = win_size

#         self.baseline = baseline
#         self.device = device

#         self.pad_in_len = ceil(1.0 * self.in_len / seg_len) * seg_len
#         self.pad_out_len = ceil(1.0 * self.out_len / seg_len) * seg_len
#         self.in_len_add = self.pad_in_len - self.in_len

#         self.enc_value_embedding = DSW_embedding(seg_len, d_model)
#         self.enc_pos_embedding = nn.Parameter(
#             torch.randn(1, self.data_dim, (self.pad_in_len // seg_len), d_model)
#         )
#         self.pre_norm = nn.LayerNorm(d_model)

#         self.encoder = Encoder(
#             e_layers,
#             win_size,
#             d_model,
#             n_heads,
#             d_ff,
#             block_depth=1,
#             dropout=dropout,
#             in_seg_num=(self.pad_in_len // seg_len),
#             factor=factor,
#         )

#         self.dec_pos_embedding = nn.Parameter(
#             torch.randn(1, self.data_dim, (self.pad_out_len // seg_len), d_model)
#         )
#         self.decoder = Decoder(
#             seg_len,
#             e_layers + 1,
#             d_model,
#             n_heads,
#             d_ff,
#             dropout,
#             out_seg_num=(self.pad_out_len // seg_len),
#             factor=factor,
#         )

#     def _to_btn(self, input_seq):
#         if input_seq.dim() == 4:
#             if input_seq.size(-1) != 1:
#                 raise ValueError(
#                     f"Crossformer expects input_seq last channel dim = 1, "
#                     f"but got shape {tuple(input_seq.shape)}."
#                 )
#             input_seq = input_seq.squeeze(-1)

#         if input_seq.dim() != 3:
#             raise ValueError(
#                 f"Crossformer expects input_seq as (B,T,N,1), (B,T,N), or (B,N,T), "
#                 f"but got shape {tuple(input_seq.shape)}."
#             )

#         if input_seq.size(-1) == self.node_num:
#             return input_seq

#         if input_seq.size(1) == self.node_num:
#             return input_seq.transpose(1, 2)

#         raise ValueError(
#             f"Cannot infer node dimension for Crossformer input_seq. "
#             f"Expected one dimension to equal node_num={self.node_num}, "
#             f"but got shape {tuple(input_seq.shape)}."
#         )

#     def forward(
#         self,
#         input_seq,
#         input_features=None,
#         target_features=None,
#         target_seq=None,
#         label=None,
#         mode="finetune",
#         *args,
#         **kwargs,
#     ):
#         if mode != "finetune":
#             raise NotImplementedError(
#                 "Crossformer JEPA pretrain is not implemented yet. "
#                 "Please refactor encode / predict / decode before using mode='pretrain'."
#             )

#         x_seq = self._to_btn(input_seq)

#         if self.baseline:
#             base = x_seq.mean(dim=1, keepdim=True)
#         else:
#             base = 0

#         batch_size = x_seq.shape[0]

#         if self.in_len_add != 0:
#             x_seq = torch.cat(
#                 (x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq),
#                 dim=1,
#             )

#         x_seq = self.enc_value_embedding(x_seq)
#         x_seq += self.enc_pos_embedding
#         x_seq = self.pre_norm(x_seq)

#         enc_out = self.encoder(x_seq)

#         dec_in = repeat(
#             self.dec_pos_embedding,
#             "b ts_d l d -> (repeat b) ts_d l d",
#             repeat=batch_size,
#         )
#         predict_y = self.decoder(dec_in, enc_out)

#         y = base + predict_y[:, : self.out_len, :]
#         return y.unsqueeze(-1)

import torch
import torch.nn as nn
from einops import repeat
from types import SimpleNamespace
from math import ceil

from .layers.Crossformer_EncDec import Encoder, Decoder, DecoderLayer, scale_block
from .layers.SelfAttention_Family import FullAttention, AttentionLayer, TwoStageAttentionLayer
from .layers.DSW_Embedding import DSW_embedding
from src.base.model import BaseModel


class Crossformer(BaseModel):
    def __init__(
        self,
        seg_len,
        win_size=4,
        factor=10,
        d_model=64,
        d_ff=128,
        n_heads=4,
        e_layers=3,
        dropout=0.0,
        baseline=False,
        device=None,
        **args,
    ):
        super(Crossformer, self).__init__(**args)

        self.data_dim = self.node_num
        self.in_len = self.his_len
        self.out_len = self.pred_len
        self.seg_len = seg_len
        self.merge_win = win_size
        self.factor = factor
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.e_layers = e_layers

        self.baseline = baseline
        self.device = device

        self.pad_in_len = ceil(1.0 * self.in_len / seg_len) * seg_len
        self.pad_out_len = ceil(1.0 * self.out_len / seg_len) * seg_len
        self.in_len_add = self.pad_in_len - self.in_len

        self.in_seg_num = self.pad_in_len // seg_len
        self.out_seg_num = self.pad_out_len // seg_len

        # Crossformer attention modules in SelfAttention_Family expect a config-like
        # object with factor and dropout attributes.
        self.attn_configs = SimpleNamespace(
            factor=factor,
            dropout=dropout,
        )

        # Encoder input embedding:
        # input_seq: (B, T, N)
        # after padding: (B, pad_in_len, N)
        # enc_value_embedding: (B, N, in_seg_num, d_model)
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)

        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, self.data_dim, self.in_seg_num, d_model)
        )
        self.pre_norm = nn.LayerNorm(d_model)

        # Encoder:
        # Encoder.forward returns a multi-scale representation list:
        # [
        #   (B, N, seg_num_0, d_model),
        #   (B, N, seg_num_1, d_model),
        #   ...
        # ]
        encoder_layers = []
        for layer_idx in range(e_layers):
            merge_win = 1 if layer_idx == 0 else win_size
            seg_num = ceil(self.in_seg_num / (win_size ** layer_idx))

            encoder_layers.append(
                scale_block(
                    self.attn_configs,
                    merge_win,
                    d_model,
                    n_heads,
                    d_ff,
                    depth=1,
                    dropout=dropout,
                    seg_num=seg_num,
                    factor=factor,
                )
            )

        self.encoder = Encoder(encoder_layers)

        # Decoder input:
        # dec_pos_embedding: (1, N, out_seg_num, d_model)
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, self.data_dim, self.out_seg_num, d_model)
        )

        decoder_layers = []
        for _ in range(e_layers + 1):
            decoder_layers.append(
                DecoderLayer(
                    self_attention=TwoStageAttentionLayer(
                        self.attn_configs,
                        self.out_seg_num,
                        factor,
                        d_model,
                        n_heads,
                        d_ff,
                        dropout,
                    ),
                    cross_attention=AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    seg_len=seg_len,
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                )
            )

        self.decoder = Decoder(decoder_layers)

        # JEPA predictor baseline:
        # Crossformer encoder output is a multi-scale list.
        # For each scale:
        #   Ex_i:      (B, N, seg_num_i, d_model)
        #   Ey_i:      (B, N, seg_num_i, d_model)
        #   Ey_pred_i: (B, N, seg_num_i, d_model)
        #
        # Current v0 assumes his_len == pred_len, so segment counts match.
        self.jepa_predictors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Linear(d_model, d_model),
                )
                for _ in range(e_layers + 1)
            ]
        )

    def _to_btn(self, seq, name="seq"):
        """
        Convert supported sequence layouts to Crossformer internal layout.

        Supported:
            input_seq / label:
                (B, T, N, 1)  current dataloader format
                (B, T, N)     Crossformer internal format
                (B, N, T)     JEPA-style format

        Return:
            seq: (B, T, N)
        """
        if seq is None:
            raise ValueError(f"{name} must not be None.")

        if seq.dim() == 4:
            if seq.size(-1) != 1:
                raise ValueError(
                    f"Crossformer expects {name} last channel dim = 1, "
                    f"but got shape {tuple(seq.shape)}."
                )
            seq = seq.squeeze(-1)

        if seq.dim() != 3:
            raise ValueError(
                f"Crossformer expects {name} as (B,T,N,1), (B,T,N), or (B,N,T), "
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

    def _pad_input(self, seq):
        """
        Pad Crossformer input sequence to pad_in_len.

        seq:
            (B, T, N)

        Return:
            padded_seq: (B, pad_in_len, N)
        """
        if seq.size(1) != self.in_len:
            raise ValueError(
                f"Crossformer JEPA v0 expects temporal length == his_len={self.in_len}, "
                f"but got {seq.size(1)}. "
                "Current v0 assumes his_len == pred_len for label encoding."
            )

        if self.in_len_add != 0:
            seq = torch.cat(
                (seq[:, :1, :].expand(-1, self.in_len_add, -1), seq),
                dim=1,
            )

        return seq

    def _build_decoder_input(self, batch_size):
        """
        Build decoder query from learnable decoder positional embedding.

        Return:
            dec_in: (B, N, out_seg_num, d_model)
        """
        return repeat(
            self.dec_pos_embedding,
            "b ts_d l d -> (repeat b) ts_d l d",
            repeat=batch_size,
        )

    def _pack_multiscale(self, enc_list):
        """
        Pack multi-scale Crossformer representations into one tensor for JEPA loss.

        enc_list:
            [
                (B, N, seg_num_0, d_model),
                (B, N, seg_num_1, d_model),
                ...
            ]

        Return:
            packed: (B, N, total_feature_dim)
        """
        if not isinstance(enc_list, (list, tuple)):
            raise ValueError(
                "Crossformer _pack_multiscale expects a list/tuple of tensors."
            )

        chunks = []
        for idx, x in enumerate(enc_list):
            if x.dim() != 4:
                raise ValueError(
                    f"Crossformer scale {idx} expects shape "
                    f"(B,N,seg_num,d_model), but got {tuple(x.shape)}."
                )
            chunks.append(x.flatten(start_dim=2))

        return torch.cat(chunks, dim=-1)

    def encode(self, seq, seq_features=None, *args, **kwargs):
        """
        Encode input or label into Crossformer multi-scale embedding space.

        input_seq:
            (B, T, N, 1) or (B, T, N) or (B, N, T)

        label / target_seq:
            (B, L, N, 1) or (B, L, N) or (B, N, L)

        Return:
            enc_out_list:
                [
                    (B, N, seg_num_0, d_model),
                    (B, N, seg_num_1, d_model),
                    ...
                ]

        Note:
            This v0 JEPA refactor assumes his_len == pred_len.
        """
        seq = self._to_btn(seq, name="seq")
        seq = self._pad_input(seq)

        # DSW embedding:
        # seq: (B, pad_in_len, N)
        # x:   (B, N, in_seg_num, d_model)
        x = self.enc_value_embedding(seq)
        x = x + self.enc_pos_embedding
        x = self.pre_norm(x)

        enc_out_list, _ = self.encoder(x)
        return enc_out_list

    def predict(self, Ex):
        """
        Predict label multi-scale representation from input multi-scale representation.

        Ex:
            [
                (B, N, seg_num_i, d_model)
            ]

        Ey_pred:
            same list structure and same shapes as Ex.
        """
        if not isinstance(Ex, (list, tuple)):
            raise ValueError(
                "Crossformer predict expects Ex as a multi-scale list/tuple."
            )

        if len(Ex) != len(self.jepa_predictors):
            raise ValueError(
                f"Crossformer predict expected {len(self.jepa_predictors)} scales, "
                f"but got {len(Ex)}."
            )

        Ey_pred = []
        for idx, (x, predictor) in enumerate(zip(Ex, self.jepa_predictors)):
            if x.dim() != 4:
                raise ValueError(
                    f"Crossformer scale {idx} expects shape "
                    f"(B,N,seg_num,d_model), but got {tuple(x.shape)}."
                )

            if x.size(-1) != self.d_model:
                raise ValueError(
                    f"Crossformer scale {idx} expects d_model={self.d_model}, "
                    f"but got {x.size(-1)}."
                )

            Ey_pred.append(predictor(x))

        return Ey_pred

    def decode(self, Ey_pred, input_seq=None, *args, **kwargs):
        """
        Decode predicted multi-scale representation into forecasting value space.

        Ey_pred:
            [
                (B, N, seg_num_i, d_model)
            ]

        input_seq:
            used only for optional baseline calculation.

        Return:
            y_pred: (B, L, N, 1)
        """
        if not isinstance(Ey_pred, (list, tuple)):
            raise ValueError(
                "Crossformer decode expects Ey_pred as a multi-scale list/tuple."
            )

        if input_seq is None:
            raise ValueError(
                "Crossformer decode requires input_seq for batch size and baseline."
            )

        input_seq = self._to_btn(input_seq, name="input_seq")
        batch_size = input_seq.shape[0]

        if self.baseline:
            base = input_seq.mean(dim=1, keepdim=True)
        else:
            base = 0

        dec_in = self._build_decoder_input(batch_size)
        predict_y = self.decoder(dec_in, Ey_pred)

        y = base + predict_y[:, : self.out_len, :]
        return y.unsqueeze(-1)

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
            Ex_list      = encode(input_seq)
            Ey_list      = encode(label)
            Ey_pred_list = predict(Ex_list)

            return:
                Ey:      (B, N, total_feature_dim)
                Ey_pred: (B, N, total_feature_dim)

        mode == "finetune":
            Ex_list      = encode(input_seq)
            Ey_pred_list = predict(Ex_list)
            y_pred       = decode(Ey_pred_list)

            return:
                y_pred: (B, L, N, 1)
        """
        if label is None:
            label = target_seq

        if label is None:
            label = kwargs.get("target_seq", None)

        if mode == "pretrain":
            if label is None:
                raise ValueError(
                    "Crossformer forward(mode='pretrain') requires label or target_seq."
                )

            Ex = self.encode(input_seq)
            Ey = self.encode(label)
            Ey_pred = self.predict(Ex)

            Ey = self._pack_multiscale(Ey)
            Ey_pred = self._pack_multiscale(Ey_pred)

            if Ey_pred.shape != Ey.shape:
                raise RuntimeError(
                    f"Crossformer JEPA shape mismatch: "
                    f"Ey_pred.shape={tuple(Ey_pred.shape)} vs Ey.shape={tuple(Ey.shape)}."
                )

            return Ey, Ey_pred

        if mode == "finetune":
            Ex = self.encode(input_seq)
            Ey_pred = self.predict(Ex)
            y_pred = self.decode(Ey_pred, input_seq=input_seq)
            return y_pred

        raise ValueError(f"Unsupported mode: {mode}")