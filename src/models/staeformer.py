# import torch.nn as nn
# import torch
# from src.base.model import BaseModel
# from einops import repeat


# class STAEformer(BaseModel):
#     def __init__(
#         self,
#         input_embedding_dim,
#         temporal_embedding_dim,
#         spatial_embedding_dim,
#         adaptive_embedding_dim,
#         feed_forward_dim,
#         num_heads,
#         num_layers,
#         use_mixed_proj,
#         dropout,
#         **args
#     ):
#         super(STAEformer, self).__init__(**args)
#         # self.tod_embedding_dim = config.tod_embedding_dim
#         # self.dow_embedding_dim = config.dow_embedding_dim
#         self.input_embedding_dim = input_embedding_dim
#         self.temporal_embedding_dim = temporal_embedding_dim
#         self.spatial_embedding_dim = spatial_embedding_dim
#         self.adaptive_embedding_dim = adaptive_embedding_dim
#         self.feed_forward_dim = feed_forward_dim
#         self.dropout = dropout
#         self.model_dim = (
#             self.input_embedding_dim
#             # + self.tod_embedding_dim
#             # + self.dow_embedding_dim
#             + self.temporal_embedding_dim
#             + self.spatial_embedding_dim
#             + self.adaptive_embedding_dim
#         )
#         self.num_heads = num_heads
#         self.num_layers = num_layers
#         self.use_mixed_proj = use_mixed_proj

#         self.input_proj = nn.Linear(1, self.input_embedding_dim)
#         # if self.tod_embedding_dim > 0:
#         #     self.tod_embedding = nn.Embedding(self.steps_per_day, self.tod_embedding_dim)
#         # if self.dow_embedding_dim > 0:
#         #     self.dow_embedding = nn.Embedding(7, self.dow_embedding_dim)
#         if self.temporal_embedding_dim > 0:
#             self.t_embed = nn.Linear(self.input_dim - 1, self.temporal_embedding_dim)
#             nn.init.xavier_uniform_(self.t_embed.weight)
#             nn.init.zeros_(self.t_embed.bias)
#         if self.spatial_embedding_dim > 0:
#             self.node_emb = nn.Parameter(
#                 torch.empty(self.node_num, self.spatial_embedding_dim)
#             )
#             nn.init.xavier_uniform_(self.node_emb)
#         if self.adaptive_embedding_dim > 0:
#             self.adaptive_embedding = nn.init.xavier_uniform_(
#                 nn.Parameter(
#                     torch.empty(
#                         self.his_len, self.node_num, self.adaptive_embedding_dim
#                     )
#                 )
#             )

#         if self.use_mixed_proj:
#             self.output_proj = nn.Linear(
#                 self.his_len * self.model_dim, self.pred_len * self.output_dim
#             )
#         else:
#             self.temporal_proj = nn.Linear(self.his_len, self.pred_len)
#             self.output_proj = nn.Linear(self.model_dim, self.output_dim)

#         self.attn_layers_t = nn.ModuleList(
#             [
#                 SelfAttentionLayer(
#                     self.model_dim, self.feed_forward_dim, self.num_heads, self.dropout
#                 )
#                 for _ in range(self.num_layers)
#             ]
#         )

#         self.attn_layers_s = nn.ModuleList(
#             [
#                 SelfAttentionLayer(
#                     self.model_dim, self.feed_forward_dim, self.num_heads, self.dropout
#                 )
#                 for _ in range(self.num_layers)
#             ]
#         )

#     def forward(self, input_seq, input_features, *args, **kwargs):
#         # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
#         batch_size = input_seq.shape[0]

#         # if self.tod_embedding_dim > 0:
#         #     tod = x[..., 1]
#         # if self.dow_embedding_dim > 0:
#         #     dow = x[..., 2]
#         # x = x[..., : self.input_dim]
#         # (batch_size, in_steps, num_nodes, input_dim)
#         x = self.input_proj(
#             input_seq
#         )  # (batch_size, in_steps, num_nodes, input_embedding_dim)
#         features = [x]
#         # if self.tod_embedding_dim > 0:
#         #     tod_emb = self.tod_embedding(
#         #         (tod * self.steps_per_day).long()
#         #     )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
#         #     features.append(tod_emb)
#         # if self.dow_embedding_dim > 0:
#         #     dow_emb = self.dow_embedding(
#         #         dow.long()
#         #     )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
#         if self.temporal_embedding_dim > 0:
#             assert (
#                 input_features is not None
#             ), "x_enc_feat must be provided when using temporal embedding"
#             t_emb = self.t_embed(input_features)
#             t_emb = repeat(t_emb, "b t f -> b t n f", n=self.node_num)
#             features.append(t_emb)
#         if self.spatial_embedding_dim > 0:
#             spatial_emb = self.node_emb.expand(
#                 batch_size, self.his_len, *self.node_emb.shape
#             )
#             features.append(spatial_emb)
#         if self.adaptive_embedding_dim > 0:
#             adp_emb = self.adaptive_embedding.expand(
#                 size=(batch_size, *self.adaptive_embedding.shape)
#             )
#             features.append(adp_emb)
#         x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)

#         for attn in self.attn_layers_t:
#             x = attn(x, dim=1)
#         for attn in self.attn_layers_s:
#             x = attn(x, dim=2)
#         # (batch_size, in_steps, num_nodes, model_dim)

#         if self.use_mixed_proj:
#             out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
#             out = out.reshape(batch_size, self.node_num, self.his_len * self.model_dim)
#             out = self.output_proj(out).view(
#                 batch_size, self.node_num, self.pred_len, self.output_dim
#             )
#             out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
#         else:
#             out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
#             out = self.temporal_proj(
#                 out
#             )  # (batch_size, model_dim, num_nodes, out_steps)
#             out = self.output_proj(
#                 out.transpose(1, 3)
#             )  # (batch_size, out_steps, num_nodes, output_dim)

#         return out


# class AttentionLayer(nn.Module):
#     """Perform attention across the -2 dim (the -1 dim is `model_dim`).

#     Make sure the tensor is permuted to correct shape before attention.

#     E.g.
#     - Input shape (batch_size, in_steps, num_nodes, model_dim).
#     - Then the attention will be performed across the nodes.

#     Also, it supports different src and tgt length.

#     But must `src length == K length == V length`.

#     """

#     def __init__(self, model_dim, num_heads=8, mask=False):
#         super().__init__()

#         self.model_dim = model_dim
#         self.num_heads = num_heads
#         self.mask = mask

#         self.head_dim = model_dim // num_heads

#         self.FC_Q = nn.Linear(model_dim, model_dim)
#         self.FC_K = nn.Linear(model_dim, model_dim)
#         self.FC_V = nn.Linear(model_dim, model_dim)

#         self.out_proj = nn.Linear(model_dim, model_dim)

#     def forward(self, query, key, value):
#         # Q    (batch_size, ..., tgt_length, model_dim)
#         # K, V (batch_size, ..., src_length, model_dim)
#         batch_size = query.shape[0]
#         tgt_length = query.shape[-2]
#         src_length = key.shape[-2]

#         query = self.FC_Q(query)
#         key = self.FC_K(key)
#         value = self.FC_V(value)

#         # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
#         query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
#         key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
#         value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

#         key = key.transpose(
#             -1, -2
#         )  # (num_heads * batch_size, ..., head_dim, src_length)

#         attn_score = (
#             query @ key
#         ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

#         if self.mask:
#             mask = torch.ones(
#                 tgt_length, src_length, dtype=torch.bool, device=query.device
#             ).tril()  # lower triangular part of the matrix
#             attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

#         attn_score = torch.softmax(attn_score, dim=-1)
#         out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
#         out = torch.cat(
#             torch.split(out, batch_size, dim=0), dim=-1
#         )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

#         out = self.out_proj(out)

#         return out


# class SelfAttentionLayer(nn.Module):
#     def __init__(
#         self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
#     ):
#         super().__init__()

#         self.attn = AttentionLayer(model_dim, num_heads, mask)
#         self.feed_forward = nn.Sequential(
#             nn.Linear(model_dim, feed_forward_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(feed_forward_dim, model_dim),
#         )
#         self.ln1 = nn.LayerNorm(model_dim)
#         self.ln2 = nn.LayerNorm(model_dim)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)

#     def forward(self, x, dim=-2):
#         x = x.transpose(dim, -2)
#         # x: (batch_size, ..., length, model_dim)
#         residual = x
#         out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
#         out = self.dropout1(out)
#         out = self.ln1(residual + out)

#         residual = out
#         out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
#         out = self.dropout2(out)
#         out = self.ln2(residual + out)

#         out = out.transpose(dim, -2)
#         return out

from typing import Optional

import torch
import torch.nn as nn
from einops import repeat, rearrange

from src.base.model import BaseModel


class STAEformer(BaseModel):
    """
    JEPA-compatible STAEformer.

    Refactor summary:
    - encode: value projection + temporal/spatial/adaptive embeddings + temporal/spatial attention layers.
    - predict: lightweight hidden-space predictor that maps Ex -> Ey_pred.
    - decode: original output projection path that maps hidden representation to value-space forecast.

    Notes:
    - Current project settings use his_len == pred_len, so both input_seq and label can be encoded
      by the same STAEformer backbone.
    - If adaptive embeddings are enabled, the encoded sequence length must match self.his_len,
      because self.adaptive_embedding is parameterized with shape (his_len, node_num, adaptive_dim).
    """

    def __init__(
        self,
        input_embedding_dim,
        temporal_embedding_dim,
        spatial_embedding_dim,
        adaptive_embedding_dim,
        feed_forward_dim,
        num_heads,
        num_layers,
        use_mixed_proj,
        dropout,
        **args,
    ):
        super(STAEformer, self).__init__(**args)

        self.input_embedding_dim = input_embedding_dim
        self.temporal_embedding_dim = temporal_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.feed_forward_dim = feed_forward_dim
        self.dropout = dropout
        self.model_dim = (
            self.input_embedding_dim
            + self.temporal_embedding_dim
            + self.spatial_embedding_dim
            + self.adaptive_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj

        # Value embedding for input_seq / label.
        self.input_proj = nn.Linear(1, self.input_embedding_dim)

        # Temporal feature embedding. input_dim includes value channel, so temporal feature dim is input_dim - 1.
        if self.temporal_embedding_dim > 0:
            self.t_embed = nn.Linear(self.input_dim - 1, self.temporal_embedding_dim)
            nn.init.xavier_uniform_(self.t_embed.weight)
            nn.init.zeros_(self.t_embed.bias)

        # Node embedding.
        if self.spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.node_num, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)

        # Adaptive spatio-temporal embedding. This is tied to self.his_len.
        if self.adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.Parameter(
                torch.empty(self.his_len, self.node_num, self.adaptive_embedding_dim)
            )
            nn.init.xavier_uniform_(self.adaptive_embedding)

        # Original decoder / output projection.
        if self.use_mixed_proj:
            self.output_proj = nn.Linear(
                self.his_len * self.model_dim,
                self.pred_len * self.output_dim,
            )
        else:
            self.temporal_proj = nn.Linear(self.his_len, self.pred_len)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(
                    self.model_dim,
                    self.feed_forward_dim,
                    self.num_heads,
                    self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(
                    self.model_dim,
                    self.feed_forward_dim,
                    self.num_heads,
                    self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

        # JEPA hidden-space predictor.
        # encode(...) returns (B, his_len, N, model_dim) under the current setting.
        self.jepa_time_predictor = nn.Linear(self.his_len, self.his_len)
        self.jepa_dim_predictor = nn.Linear(self.model_dim, self.model_dim)
        self._init_identity_linear(self.jepa_time_predictor)
        self._init_identity_linear(self.jepa_dim_predictor)

    @staticmethod
    def _init_identity_linear(layer: nn.Linear):
        """Initialize a square Linear layer as identity when possible."""
        if layer.in_features == layer.out_features:
            nn.init.eye_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def _build_feature_embeddings(
        self,
        seq: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build STAEformer token embeddings.

        Args:
            seq: (B, T, N, 1), input_seq or label/target_seq.
            features: optional temporal features with shape (B, T, F) or (B, T, N, F).

        Returns:
            x: (B, T, N, model_dim), before attention layers.
        """
        if seq.dim() != 4:
            raise ValueError(
                f"STAEformer expects seq with shape (B, T, N, C), but received {seq.shape}."
            )
        if seq.shape[-1] != 1:
            raise ValueError(
                f"STAEformer input_proj expects value channel dim 1, but received {seq.shape[-1]}."
            )

        batch_size, seq_len, node_num, _ = seq.shape
        if node_num != self.node_num:
            raise ValueError(
                f"Node number mismatch: expected {self.node_num}, but received {node_num}."
            )

        value_emb = self.input_proj(seq)  # (B, T, N, input_embedding_dim)
        output_features = [value_emb]

        if self.temporal_embedding_dim > 0:
            if features is None:
                raise ValueError(
                    "input_features/target_features must be provided when temporal_embedding_dim > 0."
                )
            if features.dim() == 3:
                if features.shape[1] != seq_len:
                    raise ValueError(
                        f"Temporal feature length mismatch: features.shape[1]={features.shape[1]}, seq_len={seq_len}."
                    )
                t_emb = self.t_embed(features)
                t_emb = repeat(t_emb, "b t f -> b t n f", n=self.node_num)
            elif features.dim() == 4:
                if features.shape[1] != seq_len or features.shape[2] != self.node_num:
                    raise ValueError(
                        f"Temporal feature shape mismatch: expected (B, {seq_len}, {self.node_num}, F), "
                        f"but received {features.shape}."
                    )
                t_emb = self.t_embed(features)
            else:
                raise ValueError(
                    f"features must have shape (B, T, F) or (B, T, N, F), but received {features.shape}."
                )
            output_features.append(t_emb)

        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(batch_size, seq_len, *self.node_emb.shape)
            output_features.append(spatial_emb)

        if self.adaptive_embedding_dim > 0:
            if seq_len != self.his_len:
                raise ValueError(
                    "STAEformer adaptive_embedding is tied to self.his_len. "
                    f"Expected seq_len={self.his_len}, but received seq_len={seq_len}. "
                    "For his_len != pred_len settings, define a separate target adaptive embedding "
                    "or disable adaptive_embedding_dim."
                )
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            output_features.append(adp_emb)

        x = torch.cat(output_features, dim=-1)  # (B, T, N, model_dim)
        return x

    def encode(
        self,
        seq: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode input_seq or label into hidden representation.

        Shapes:
            seq: (B, T, N, 1)
            features: (B, T, F) or (B, T, N, F)
            return: (B, T, N, model_dim)
        """
        x = self._build_feature_embeddings(seq, features)

        for attn in self.attn_layers_t:
            x = attn(x, dim=1)
        for attn in self.attn_layers_s:
            x = attn(x, dim=2)

        # Ex/Ey: (B, T, N, model_dim)
        return x

    def predict(self, Ex: torch.Tensor) -> torch.Tensor:
        """
        Predict target hidden representation from input hidden representation.

        Shapes:
            Ex: (B, his_len, N, model_dim)
            Ey_pred: (B, his_len, N, model_dim)
        """
        if Ex.dim() != 4:
            raise ValueError(f"Ex must have shape (B, T, N, D), but received {Ex.shape}.")
        if Ex.shape[1] != self.his_len:
            raise ValueError(
                f"JEPA time predictor expects T={self.his_len}, but received T={Ex.shape[1]}."
            )
        if Ex.shape[-1] != self.model_dim:
            raise ValueError(
                f"JEPA dim predictor expects D={self.model_dim}, but received D={Ex.shape[-1]}."
            )

        x = rearrange(Ex, "b t n d -> b n d t")
        x = self.jepa_time_predictor(x)
        x = rearrange(x, "b n d t -> b t n d")
        x = self.jepa_dim_predictor(x)
        return x

    def decode(self, Ey_pred: torch.Tensor) -> torch.Tensor:
        """
        Decode hidden representation to value-space forecast.

        Shapes:
            Ey_pred: (B, his_len, N, model_dim)
            y_pred: (B, pred_len, N, output_dim)
        """
        if Ey_pred.dim() != 4:
            raise ValueError(
                f"Ey_pred must have shape (B, T, N, D), but received {Ey_pred.shape}."
            )
        if Ey_pred.shape[1] != self.his_len:
            raise ValueError(
                f"STAEformer decoder expects T={self.his_len}, but received T={Ey_pred.shape[1]}."
            )
        if Ey_pred.shape[2] != self.node_num:
            raise ValueError(
                f"STAEformer decoder expects N={self.node_num}, but received N={Ey_pred.shape[2]}."
            )
        if Ey_pred.shape[-1] != self.model_dim:
            raise ValueError(
                f"STAEformer decoder expects D={self.model_dim}, but received D={Ey_pred.shape[-1]}."
            )

        batch_size = Ey_pred.shape[0]

        if self.use_mixed_proj:
            out = Ey_pred.transpose(1, 2)  # (B, N, his_len, model_dim)
            out = out.reshape(batch_size, self.node_num, self.his_len * self.model_dim)
            out = self.output_proj(out).view(
                batch_size,
                self.node_num,
                self.pred_len,
                self.output_dim,
            )
            out = out.transpose(1, 2)  # (B, pred_len, N, output_dim)
        else:
            out = Ey_pred.transpose(1, 3)  # (B, model_dim, N, his_len)
            out = self.temporal_proj(out)  # (B, model_dim, N, pred_len)
            out = self.output_proj(out.transpose(1, 3))  # (B, pred_len, N, output_dim)

        return out

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
            - mode="finetune": return y_pred for downstream forecasting loss.

        Shapes:
            input_seq: (B, his_len, N, 1)
            label/target_seq: (B, his_len, N, 1) under current his_len == pred_len setting
            Ex: (B, his_len, N, model_dim)
            Ey: (B, his_len, N, model_dim)
            Ey_pred: (B, his_len, N, model_dim)
            y_pred: (B, pred_len, N, output_dim)
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
                        "target_features must be provided for STAEformer pretraining when "
                        "label length differs from input_features length."
                    )

            Ex = self.encode(input_seq, input_features)
            Ey = self.encode(label_seq, label_features)
            Ey_pred = self.predict(Ex)

            if Ey_pred.shape != Ey.shape:
                raise ValueError(
                    f"JEPA hidden shape mismatch: Ey_pred.shape={Ey_pred.shape}, Ey.shape={Ey.shape}."
                )
            return Ey, Ey_pred

        elif mode == "finetune":
            Ex = self.encode(input_seq, input_features)
            Ey_pred = self.predict(Ex)
            y_pred = self.decode(Ey_pred)
            return y_pred

        else:
            raise ValueError(f"Unsupported mode: {mode}")


class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is model_dim)."""

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(-1, -2)
        attn_score = (query @ key) / self.head_dim**0.5

        if self.mask:
            mask = torch.ones(
                tgt_length,
                src_length,
                dtype=torch.bool,
                device=query.device,
            ).tril()
            attn_score.masked_fill_(~mask, -torch.inf)

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)
        out = self.out_proj(out)
        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        model_dim,
        feed_forward_dim=2048,
        num_heads=8,
        dropout=0,
        mask=False,
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        residual = x
        out = self.attn(x, x, x)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out
