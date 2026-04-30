# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from src.base.model import BaseModel


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import repeat


# class STTN(BaseModel):
#     """
#     Reference code: https://github.com/xumingxingsjtu/STTN
#     """

#     def __init__(
#         self,
#         supports,
#         blocks,
#         mlp_expand,
#         hidden_channels,
#         end_channels,
#         dropout,
#         **args,
#     ):
#         super(STTN, self).__init__(**args)
#         self.t_modules = nn.ModuleList()
#         self.s_modules = nn.ModuleList()
#         self.supports = tuple(supports)
#         self.blocks = blocks
#         self.bn = nn.ModuleList()

#         self.start_conv = nn.Conv2d(
#             in_channels=self.input_dim, out_channels=hidden_channels, kernel_size=(1, 1)
#         )
#         self.blocks = blocks
#         for b in range(blocks):
#             self.t_modules.append(
#                 TemporalTransformer(
#                     dim=hidden_channels,
#                     depth=1,
#                     heads=4,
#                     mlp_dim=hidden_channels * mlp_expand,
#                     time_num=self.his_len,
#                     dropout=dropout,
#                     window_size=self.his_len,
#                 )
#             )

#             self.s_modules.append(
#                 SpatialTransformer(
#                     dim=hidden_channels,
#                     depth=1,
#                     heads=4,
#                     mlp_dim=hidden_channels * mlp_expand,
#                     node_num=self.node_num,
#                     dropout=dropout,
#                     stage=b,
#                 )
#             )

#             self.bn.append(nn.BatchNorm2d(hidden_channels))

#         self.end_conv_1 = nn.Conv2d(
#             in_channels=hidden_channels,
#             out_channels=end_channels,
#             kernel_size=(1, 1),
#             bias=True,
#         )

#         self.end_conv_2 = nn.Conv2d(
#             in_channels=end_channels,
#             out_channels=self.output_dim * self.pred_len,
#             kernel_size=(1, 1),
#             bias=True,
#         )

#     def forward(self, input_seq, input_features, *args, **kwargs):
#         # (b, t, n, f)
#         x = torch.concat(
#             [
#                 input_seq,
#                 repeat(input_features, "b t f -> b t n f", n=input_seq.shape[2]),
#             ],
#             dim=-1,
#         )
#         x = x.transpose(1, 3)
#         x = self.start_conv(x)
#         for i in range(self.blocks):
#             residual = x.detach()
#             x = self.s_modules[i](x, torch.stack(self.supports))
#             x = self.t_modules[i](x)
#             x = self.bn[i](x) + residual

#         x = x[..., -1:]
#         out = F.relu(self.end_conv_1(x))
#         out = self.end_conv_2(out)
#         return out


# class TemporalTransformer(nn.Module):
#     def __init__(self, dim, depth, heads, mlp_dim, time_num, dropout, window_size):
#         super().__init__()
#         self.pos_embedding = nn.Parameter(torch.randn(1, time_num, dim))
#         self.layers = nn.ModuleList([])
#         for i in range(depth):
#             self.layers.append(
#                 nn.ModuleList(
#                     [
#                         TemporalAttention(
#                             dim=dim,
#                             heads=heads,
#                             window_size=window_size,
#                             dropout=dropout,
#                             causal=True,
#                             stage=i,
#                         ),
#                         PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
#                     ]
#                 )
#             )

#     def forward(self, x):
#         b, c, n, t = x.shape
#         x = x.permute(0, 2, 3, 1).reshape(b * n, t, c)
#         x = x + self.pos_embedding
#         for attn, ff in self.layers:
#             x = attn(x) + x
#             x = ff(x) + x
#         x = x.reshape(b, n, t, c).permute(0, 3, 1, 2)
#         return x


# class TemporalAttention(nn.Module):
#     def __init__(
#         self,
#         dim,
#         heads=8,
#         window_size=1,
#         dropout=0.0,
#         causal=True,
#         stage=0,
#         qkv_bias=False,
#         qk_scale=None,
#     ):
#         super().__init__()
#         assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."

#         self.dim = dim
#         self.num_heads = heads
#         self.causal = causal
#         head_dim = dim // heads
#         self.scale = qk_scale or head_dim**-0.5
#         self.window_size = window_size
#         self.stage = stage

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

#         self.attn_drop = nn.Dropout(dropout)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(dropout)

#         self.mask = torch.tril(torch.ones(window_size, window_size))

#     def forward(self, x):
#         B_prev, T_prev, C_prev = x.shape
#         if self.window_size > 0:
#             x = x.reshape(-1, self.window_size, C_prev)
#         B, T, C = x.shape

#         qkv = (
#             self.qkv(x)
#             .reshape(B, -1, 3, self.num_heads, C // self.num_heads)
#             .permute(2, 0, 3, 1, 4)
#         )
#         q, k, v = qkv[0], qkv[1], qkv[2]

#         attn = (q @ k.transpose(-2, -1)) * self.scale

#         if self.causal:
#             mask = self.mask.to(x.device)
#             attn = attn.masked_fill_(mask == 0, float("-inf")).softmax(dim=-1)

#         attn = attn.softmax(dim=-1)

#         x = (attn @ v).transpose(1, 2).reshape(B, T, C)

#         x = self.proj(x)
#         x = self.proj_drop(x)

#         if self.window_size > 0:
#             x = x.reshape(B_prev, T_prev, C_prev)
#         return x


# class SpatialTransformer(nn.Module):
#     def __init__(self, dim, depth, heads, mlp_dim, node_num, dropout, stage=0):
#         super().__init__()
#         self.pos_embedding = nn.Parameter(torch.randn(1, node_num, dim))
#         self.layers = nn.ModuleList([])
#         for i in range(depth):
#             self.layers.append(
#                 nn.ModuleList(
#                     [
#                         SpatialAttention(
#                             dim, heads=heads, dropout=dropout, stage=stage
#                         ),
#                         PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
#                         GCN(dim, dim, dropout, support_len=2),
#                     ]
#                 )
#             )

#     def forward(self, x, adj):
#         b, c, n, t = x.shape
#         x = x.permute(0, 3, 2, 1).reshape(b * t, n, c)
#         x = x + self.pos_embedding
#         for attn, ff, gcn in self.layers:
#             residual = x.reshape(b, t, n, c)
#             x = attn(x, adj) + x
#             x = ff(x) + x

#             x = (
#                 gcn(residual.permute(0, 3, 2, 1), adj)
#                 .permute(0, 3, 2, 1)
#                 .reshape(b * t, n, c)
#                 + x
#             )
#         x = x.reshape(b, t, n, c).permute(0, 3, 2, 1)
#         return x


# class SpatialAttention(nn.Module):
#     def __init__(
#         self, dim, heads=8, dropout=0.0, stage=0, qkv_bias=False, qk_scale=None
#     ):
#         super().__init__()
#         assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."

#         self.dim = dim
#         self.num_heads = heads
#         head_dim = dim // heads
#         self.scale = qk_scale or head_dim**-0.5
#         self.stage = stage

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

#         self.attn_drop = nn.Dropout(dropout)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(dropout)

#     def forward(self, x, adj=None):
#         B, N, C = x.shape

#         qkv = (
#             self.qkv(x)
#             .reshape(B, -1, 3, self.num_heads, C // self.num_heads)
#             .permute(2, 0, 3, 1, 4)
#         )
#         q, k, v = qkv[0], qkv[1], qkv[2]

#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)

#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)

#         return x


# class PreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.fn = fn

#     def forward(self, x, **kwargs):
#         return self.fn(self.norm(x), **kwargs)


# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout=0.0):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout),
#         )

#     def forward(self, x):
#         return self.net(x)


# class nconv(nn.Module):
#     def __init__(self):
#         super(nconv, self).__init__()

#     def forward(self, x, A):
#         x = torch.einsum("ncvl,vw->ncwl", (x, A))
#         return x.contiguous()


# class linear(nn.Module):
#     def __init__(self, c_in, c_out):
#         super(linear, self).__init__()
#         self.mlp = torch.nn.Conv2d(
#             c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True
#         )

#     def forward(self, x):
#         return self.mlp(x)


# class GCN(nn.Module):
#     def __init__(self, c_in, c_out, dropout, support_len=1, order=2):
#         super(GCN, self).__init__()
#         self.nconv = nconv()
#         c_in = (order * support_len + 1) * c_in
#         self.mlp = linear(c_in, c_out)
#         self.dropout = dropout
#         self.order = order

#     def forward(self, x, support):
#         out = [x]
#         for a in support:
#             x1 = self.nconv(x, a)
#             out.append(x1)
#             for k in range(2, self.order + 1):
#                 x2 = self.nconv(x1, a)
#                 out.append(x2)
#                 x1 = x2

#         h = torch.cat(out, dim=1)
#         h = self.mlp(h)
#         h = F.dropout(h, self.dropout, training=self.training)
#         return h

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional

from src.base.model import BaseModel


class STTN(BaseModel):
    """
    Reference code: https://github.com/xumingxingsjtu/STTN

    JEPA refactor notes:
    - encode: input/target sequence + optional time features -> STTN hidden representation.
    - predict: lightweight trainable predictor in hidden space. It maps encoded input
      representation Ex to predicted target representation Ey_pred.
    - decode: original output head end_conv_1 + end_conv_2, applied to the predicted
      hidden representation.

    Important assumption:
    - The original STTN TemporalTransformer uses a fixed positional embedding and
      fixed attention window based on self.his_len. Therefore, JEPA pretraining is
      reliable when target sequence length equals self.his_len. In the current
      short/long settings, his_len == pred_len, so input_seq and target_seq can be
      encoded by the same backbone.
    """

    def __init__(
        self,
        supports,
        blocks,
        mlp_expand,
        hidden_channels,
        end_channels,
        dropout,
        **args,
    ):
        super(STTN, self).__init__(**args)
        self.t_modules = nn.ModuleList()
        self.s_modules = nn.ModuleList()
        self.supports = tuple(supports)
        self.blocks = blocks
        self.bn = nn.ModuleList()
        self.hidden_channels = hidden_channels

        self.start_conv = nn.Conv2d(
            in_channels=self.input_dim,
            out_channels=hidden_channels,
            kernel_size=(1, 1),
        )

        for b in range(blocks):
            self.t_modules.append(
                TemporalTransformer(
                    dim=hidden_channels,
                    depth=1,
                    heads=4,
                    mlp_dim=hidden_channels * mlp_expand,
                    time_num=self.his_len,
                    dropout=dropout,
                    window_size=self.his_len,
                )
            )

            self.s_modules.append(
                SpatialTransformer(
                    dim=hidden_channels,
                    depth=1,
                    heads=4,
                    mlp_dim=hidden_channels * mlp_expand,
                    node_num=self.node_num,
                    dropout=dropout,
                    stage=b,
                )
            )

            self.bn.append(nn.BatchNorm2d(hidden_channels))

        self.end_conv_1 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=end_channels,
            kernel_size=(1, 1),
            bias=True,
        )

        self.end_conv_2 = nn.Conv2d(
            in_channels=end_channels,
            out_channels=self.output_dim * self.pred_len,
            kernel_size=(1, 1),
            bias=True,
        )

        # JEPA predictor in hidden space.
        # encode(...) returns Ex/Ey with shape (B, encoded_t, N, hidden_channels).
        # Under the current setting encoded_t == his_len == pred_len.
        self.jepa_time_predictor = nn.Linear(self.his_len, self.his_len)
        self.jepa_dim_predictor = nn.Linear(hidden_channels, hidden_channels)
        self._init_identity_linear(self.jepa_time_predictor)
        self._init_identity_linear(self.jepa_dim_predictor)

    @staticmethod
    def _init_identity_linear(layer: nn.Linear):
        """Initialize a square Linear layer as identity when possible."""
        if layer.in_features == layer.out_features:
            nn.init.eye_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def _build_sttn_input(
        self,
        seq: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build STTN input tensor.

        Args:
            seq: shape (B, T, N, C_seq), usually C_seq = 1.
            features: optional temporal/external features. Supported shapes:
                - (B, T, F): repeated to every node;
                - (B, T, N, F): used directly;
                - (B, T, 1, F): expanded to every node.

        Returns:
            x: shape (B, input_dim, N, T), matching the original STTN input
               after concat + transpose.
        """
        if seq.dim() != 4:
            raise ValueError(
                f"STTN seq must have shape (B, T, N, C), but received {seq.shape}."
            )

        if features is None:
            x = seq
        else:
            if features.dim() == 3:
                features = repeat(features, "b t f -> b t n f", n=seq.shape[2])
            elif features.dim() == 4:
                if features.shape[2] == 1 and seq.shape[2] != 1:
                    features = features.expand(-1, -1, seq.shape[2], -1)
                elif features.shape[2] != seq.shape[2]:
                    raise ValueError(
                        f"STTN feature node dimension mismatch: "
                        f"features.shape={features.shape}, seq.shape={seq.shape}."
                    )
            else:
                raise ValueError(
                    f"STTN features must have shape (B, T, F) or (B, T, N, F), "
                    f"but received {features.shape}."
                )
            x = torch.concat([seq, features], dim=-1)

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"STTN expected input feature dim {self.input_dim}, "
                f"but received {x.shape[-1]}. "
                "Please provide matching input_features/target_features."
            )

        return x.transpose(1, 3)  # (B, F, N, T)

    def _check_encode_length(self, seq: torch.Tensor, stage_name: str):
        """
        STTN uses fixed temporal positional embeddings and attention windows
        initialized with self.his_len. Therefore, the encoded sequence length
        must match self.his_len for this refactored shared encoder.
        """
        if seq.shape[1] != self.his_len:
            raise ValueError(
                f"STTN JEPA {stage_name} requires sequence length == self.his_len "
                f"because TemporalTransformer has fixed positional embeddings and "
                f"window_size. Got seq_len={seq.shape[1]}, self.his_len={self.his_len}. "
                "If his_len != pred_len is required, please implement a separate "
                "target encoder or a dynamic temporal embedding/window mechanism."
            )

    def encode(
        self,
        seq: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        stage_name: str = "encode",
    ) -> torch.Tensor:
        """
        Encode input_seq or target_seq into STTN hidden space.

        Shapes:
            seq: (B, T, N, C_seq), usually C_seq = 1.
            features: optional (B, T, F), (B, T, N, F), or (B, T, 1, F).
            internal x: (B, hidden_channels, N, T).
            returned embedding: (B, T, N, hidden_channels).
        """
        self._check_encode_length(seq, stage_name)

        x = self._build_sttn_input(seq, features)
        x = self.start_conv(x)  # (B, hidden_channels, N, T)

        supports = torch.stack(self.supports)

        for i in range(self.blocks):
            # Keep original residual behavior for compatibility.
            residual = x.detach()
            x = self.s_modules[i](x, supports)
            x = self.t_modules[i](x)
            x = self.bn[i](x) + residual

        # Convert to common JEPA representation layout:
        # (B, hidden_channels, N, T) -> (B, T, N, hidden_channels)
        return rearrange(x, "b d n t -> b t n d")

    def predict(self, Ex: torch.Tensor) -> torch.Tensor:
        """
        Predict target hidden representation from input hidden representation.

        Args:
            Ex: shape (B, T, N, hidden_channels).

        Returns:
            Ey_pred: shape (B, T, N, hidden_channels), matching Ey exactly.
        """
        if Ex.dim() != 4:
            raise ValueError(
                f"STTN predict expects Ex with shape (B, T, N, D), "
                f"but received {Ex.shape}."
            )

        if Ex.shape[1] != self.his_len:
            raise ValueError(
                f"STTN predictor expects time dimension {self.his_len}, "
                f"but received {Ex.shape[1]}."
            )

        x = rearrange(Ex, "b t n d -> b n d t")
        x = self.jepa_time_predictor(x)
        x = rearrange(x, "b n d t -> b t n d")
        x = self.jepa_dim_predictor(x)
        return x

    def decode(self, Ey_pred: torch.Tensor) -> torch.Tensor:
        """
        Decode hidden representation to value-space forecasting output.

        Args:
            Ey_pred: shape (B, T, N, hidden_channels).

        Returns:
            y_pred: shape (B, pred_len, N, output_dim), matching original STTN
            forward output when output_dim == 1.
        """
        if Ey_pred.dim() != 4:
            raise ValueError(
                f"STTN decode expects Ey_pred with shape (B, T, N, D), "
                f"but received {Ey_pred.shape}."
            )

        x = rearrange(Ey_pred, "b t n d -> b d n t")

        # Original STTN only uses the final hidden time step for value-space output.
        x = x[..., -1:]
        out = F.relu(self.end_conv_1(x))
        out = self.end_conv_2(out)  # (B, output_dim * pred_len, N, 1)

        if self.output_dim == 1:
            return out

        # General fallback for output_dim > 1:
        # (B, output_dim * pred_len, N, 1) -> (B, pred_len, N, output_dim)
        return rearrange(
            out.squeeze(-1),
            "b (t c) n -> b t n c",
            t=self.pred_len,
            c=self.output_dim,
        )

    def forward(
        self,
        input_seq: torch.Tensor,
        input_features: Optional[torch.Tensor] = None,
        target_seq: Optional[torch.Tensor] = None,
        target_features: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        mode: str = "finetune",
        *args,
        **kwargs,
    ):
        """
        Forward modes:
            - mode="pretrain": return (Ey, Ey_pred) for JEPA hidden-space loss.
            - mode="finetune": return y_pred for forecasting loss.

        Shapes:
            input_seq: (B, T, N, C_seq), usually C_seq = 1.
            target_seq/label: (B, L, N, C_seq), usually C_seq = 1.
            Ex: (B, T, N, hidden_channels).
            Ey: (B, L, N, hidden_channels), currently requires L == T == self.his_len.
            Ey_pred: (B, T, N, hidden_channels), checked to match Ey.
            y_pred: (B, pred_len, N, output_dim), matching original output when output_dim == 1.
        """
        if mode == "pretrain":
            label_seq = label if label is not None else target_seq
            if label_seq is None:
                raise ValueError("Labels need to be provided during the pre-training process.")

            label_features = target_features
            if label_features is None and input_features is not None:
                if input_features.shape[1] == label_seq.shape[1]:
                    label_features = input_features
                else:
                    raise ValueError(
                        "target_features must be provided for STTN pretraining when "
                        "label length differs from input_features length."
                    )

            Ex = self.encode(input_seq, input_features, stage_name="input encoding")
            Ey = self.encode(label_seq, label_features, stage_name="target encoding")
            Ey_pred = self.predict(Ex)

            if Ey_pred.shape != Ey.shape:
                raise ValueError(
                    f"JEPA hidden shape mismatch: Ey_pred.shape={Ey_pred.shape}, "
                    f"Ey.shape={Ey.shape}. STTN currently expects encoded input and "
                    f"target lengths to match, typically his_len == pred_len."
                )

            return Ey, Ey_pred

        elif mode == "finetune":
            Ex = self.encode(input_seq, input_features, stage_name="finetune input encoding")
            Ey_pred = self.predict(Ex)
            y_pred = self.decode(Ey_pred)
            return y_pred

        else:
            raise ValueError(f"Unsupported mode: {mode}")


class TemporalTransformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, time_num, dropout, window_size):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, time_num, dim))
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        TemporalAttention(
                            dim=dim,
                            heads=heads,
                            window_size=window_size,
                            dropout=dropout,
                            causal=True,
                            stage=i,
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        b, c, n, t = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b * n, t, c)
        x = x + self.pos_embedding
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        x = x.reshape(b, n, t, c).permute(0, 3, 1, 2)
        return x


class TemporalAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        window_size=1,
        dropout=0.0,
        causal=True,
        stage=0,
        qkv_bias=False,
        qk_scale=None,
    ):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."

        self.dim = dim
        self.num_heads = heads
        self.causal = causal
        head_dim = dim // heads
        self.scale = qk_scale or head_dim**-0.5
        self.window_size = window_size
        self.stage = stage

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        self.mask = torch.tril(torch.ones(window_size, window_size))

    def forward(self, x):
        B_prev, T_prev, C_prev = x.shape
        if self.window_size > 0:
            x = x.reshape(-1, self.window_size, C_prev)
        B, T, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, -1, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.causal:
            mask = self.mask.to(x.device)
            attn = attn.masked_fill_(mask == 0, float("-inf")).softmax(dim=-1)

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, T, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        if self.window_size > 0:
            x = x.reshape(B_prev, T_prev, C_prev)
        return x


class SpatialTransformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, node_num, dropout, stage=0):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, node_num, dim))
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        SpatialAttention(dim, heads=heads, dropout=dropout, stage=stage),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                        GCN(dim, dim, dropout, support_len=2),
                    ]
                )
            )

    def forward(self, x, adj):
        b, c, n, t = x.shape
        x = x.permute(0, 3, 2, 1).reshape(b * t, n, c)
        x = x + self.pos_embedding
        for attn, ff, gcn in self.layers:
            residual = x.reshape(b, t, n, c)
            x = attn(x, adj) + x
            x = ff(x) + x

            x = (
                gcn(residual.permute(0, 3, 2, 1), adj)
                .permute(0, 3, 2, 1)
                .reshape(b * t, n, c)
                + x
            )
        x = x.reshape(b, t, n, c).permute(0, 3, 2, 1)
        return x


class SpatialAttention(nn.Module):
    def __init__(
        self, dim, heads=8, dropout=0.0, stage=0, qkv_bias=False, qk_scale=None
    ):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."

        self.dim = dim
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim**-0.5
        self.stage = stage

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, adj=None):
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, -1, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum("ncvl,vw->ncwl", (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(
            c_in,
            c_out,
            kernel_size=(1, 1),
            padding=(0, 0),
            stride=(1, 1),
            bias=True,
        )

    def forward(self, x):
        return self.mlp(x)


class GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=1, order=2):
        super(GCN, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h
