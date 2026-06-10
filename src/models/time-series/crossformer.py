import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .layers.Crossformer_EncDec import Encoder
from .layers.Crossformer_EncDec import Decoder
from .layers.Crossformer_Attn import FullAttention, AttentionLayer, TwoStageAttentionLayer
from .layers.DSW_Embedding import DSW_embedding
from src.base.model import BaseModel

from math import ceil

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

        self.baseline = baseline
        self.device = device

        self.pad_in_len = ceil(1.0 * self.in_len / seg_len) * seg_len
        self.pad_out_len = ceil(1.0 * self.out_len / seg_len) * seg_len
        self.in_len_add = self.pad_in_len - self.in_len

        self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, self.data_dim, (self.pad_in_len // seg_len), d_model)
        )
        self.pre_norm = nn.LayerNorm(d_model)

        self.encoder = Encoder(
            e_layers,
            win_size,
            d_model,
            n_heads,
            d_ff,
            block_depth=1,
            dropout=dropout,
            in_seg_num=(self.pad_in_len // seg_len),
            factor=factor,
        )

        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, self.data_dim, (self.pad_out_len // seg_len), d_model)
        )
        self.decoder = Decoder(
            seg_len,
            e_layers + 1,
            d_model,
            n_heads,
            d_ff,
            dropout,
            out_seg_num=(self.pad_out_len // seg_len),
            factor=factor,
        )

    def _to_btn(self, input_seq):
        if input_seq.dim() == 4:
            if input_seq.size(-1) != 1:
                raise ValueError(
                    f"Crossformer expects input_seq last channel dim = 1, "
                    f"but got shape {tuple(input_seq.shape)}."
                )
            input_seq = input_seq.squeeze(-1)

        if input_seq.dim() != 3:
            raise ValueError(
                f"Crossformer expects input_seq as (B,T,N,1), (B,T,N), or (B,N,T), "
                f"but got shape {tuple(input_seq.shape)}."
            )

        if input_seq.size(-1) == self.node_num:
            return input_seq

        if input_seq.size(1) == self.node_num:
            return input_seq.transpose(1, 2)

        raise ValueError(
            f"Cannot infer node dimension for Crossformer input_seq. "
            f"Expected one dimension to equal node_num={self.node_num}, "
            f"but got shape {tuple(input_seq.shape)}."
        )

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
        if mode != "finetune":
            raise NotImplementedError(
                "Crossformer JEPA pretrain is not implemented yet. "
                "Please refactor encode / predict / decode before using mode='pretrain'."
            )

        x_seq = self._to_btn(input_seq)

        if self.baseline:
            base = x_seq.mean(dim=1, keepdim=True)
        else:
            base = 0

        batch_size = x_seq.shape[0]

        if self.in_len_add != 0:
            x_seq = torch.cat(
                (x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq),
                dim=1,
            )

        x_seq = self.enc_value_embedding(x_seq)
        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)

        enc_out = self.encoder(x_seq)

        dec_in = repeat(
            self.dec_pos_embedding,
            "b ts_d l d -> (repeat b) ts_d l d",
            repeat=batch_size,
        )
        predict_y = self.decoder(dec_in, enc_out)

        y = base + predict_y[:, : self.out_len, :]
        return y.unsqueeze(-1)