# from typing import Optional, Sequence

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange, repeat
# from torch.nn import Linear, ReLU, Sequential

# from src.base.model import BaseModel


# class GWNET(BaseModel):
#     """
#     JEPA-compatible Graph WaveNet backbone.

#     Reference code: https://github.com/nnzhan/Graph-WaveNet

#     Refactor notes:
#     - encode: input/target sequence + optional time features -> WaveNet skip hidden representation.
#     - predict: lightweight trainable hidden-space predictor. It maps encoded input
#       hidden states to the encoded target hidden-state shape.
#     - decode: original Graph WaveNet output head, i.e. optional temporal MLP projection
#       followed by end_conv_1 and end_conv_2.

#     Expected data shapes:
#     - input_seq:      (B, T, N, C_seq), usually C_seq = 1.
#     - target_seq:     (B, L, N, C_seq), usually C_seq = 1.
#     - input_features: (B, T, F) or (B, T, N, F), optional.
#     - target_features:(B, L, F) or (B, L, N, F), optional.

#     Hidden representation shape returned by encode:
#     - Ex/Ey:          (B, t_enc, N, skip_channels).

#     Forward modes:
#     - mode="pretrain": return (Ey, Ey_pred), where Ey_pred.shape == Ey.shape.
#     - mode="finetune": return y_pred with the original forecasting output shape.
#     """

#     def __init__(
#         self,
#         supports,
#         adp_adj,
#         dropout,
#         residual_channels,
#         dilation_channels,
#         skip_channels,
#         end_channels,
#         kernel_size,
#         blocks,
#         layers,
#         **args,
#     ):
#         super(GWNET, self).__init__(**args)
#         self.supports = supports
#         self.supports_len = len(supports)
#         self.adp_adj = adp_adj

#         self.dropout = dropout
#         self.blocks = blocks
#         self.layers = layers
#         self.kernel_size = kernel_size
#         self.skip_channels = skip_channels

#         self.filter_convs = nn.ModuleList()
#         self.gate_convs = nn.ModuleList()
#         self.residual_convs = nn.ModuleList()
#         self.skip_convs = nn.ModuleList()
#         self.bn = nn.ModuleList()
#         self.gconv = nn.ModuleList()

#         self.start_conv = nn.Conv2d(
#             in_channels=self.input_dim,
#             out_channels=residual_channels,
#             kernel_size=(1, 1),
#         )
#         receptive_field = 1

#         if self.adp_adj:
#             self.nodevec1 = nn.Parameter(
#                 torch.randn(self.node_num, 10), requires_grad=True
#             )
#             self.nodevec2 = nn.Parameter(
#                 torch.randn(10, self.node_num), requires_grad=True
#             )
#             self.supports_len += 1

#         for _ in range(self.blocks):
#             additional_scope = self.kernel_size - 1
#             new_dilation = 1
#             for _ in range(self.layers):
#                 # Dilated temporal convolutions.
#                 self.filter_convs.append(
#                     nn.Conv2d(
#                         in_channels=residual_channels,
#                         out_channels=dilation_channels,
#                         kernel_size=(1, self.kernel_size),
#                         dilation=new_dilation,
#                         stride=1,
#                     )
#                 )

#                 self.gate_convs.append(
#                     nn.Conv2d(
#                         in_channels=residual_channels,
#                         out_channels=dilation_channels,
#                         kernel_size=(1, self.kernel_size),
#                         dilation=new_dilation,
#                         stride=1,
#                     )
#                 )

#                 # Kept for compatibility with the original module definition.
#                 # The original forward path does not explicitly call residual_convs.
#                 self.residual_convs.append(
#                     nn.Conv1d(
#                         in_channels=dilation_channels,
#                         out_channels=residual_channels,
#                         kernel_size=(1, 1),
#                     )
#                 )

#                 # 1x1 convolution for skip connection.
#                 self.skip_convs.append(
#                     nn.Conv2d(
#                         in_channels=dilation_channels,
#                         out_channels=skip_channels,
#                         kernel_size=(1, 1),
#                     )
#                 )
#                 self.bn.append(nn.BatchNorm2d(residual_channels))
#                 new_dilation *= 2
#                 receptive_field += additional_scope
#                 additional_scope *= 2
#                 self.gconv.append(
#                     GCN(
#                         dilation_channels,
#                         residual_channels,
#                         dropout,
#                         support_len=self.supports_len,
#                     )
#                 )

#         self.end_conv_1 = nn.Conv2d(
#             in_channels=skip_channels,
#             out_channels=end_channels,
#             kernel_size=(1, 1),
#             bias=True,
#         )

#         self.end_conv_2 = nn.Conv2d(
#             in_channels=end_channels,
#             out_channels=self.pred_len,
#             kernel_size=(1, 1),
#             bias=True,
#         )

#         self.receptive_field = receptive_field

#         # This follows the original implementation. For the current repository
#         # configs, it matches the encoded temporal length for long forecasting;
#         # for short forecasting it is <= 0 and the MLP projection is skipped.
#         self.mlp_input_dim = (
#             self.his_len - (self.kernel_size - 1) * (1 + self.layers) * blocks
#         )
#         if self.mlp_input_dim > 0:
#             self.mlp_projection = Sequential(
#                 Linear(self.mlp_input_dim, 64),
#                 ReLU(),
#                 Linear(64, 128),
#                 ReLU(),
#                 Linear(128, 64),
#                 ReLU(),
#                 Linear(64, self.output_dim),
#             )

#         # Encoded time length after the dilated WaveNet stack. The stack pads
#         # sequences shorter than the receptive field, so the encoded length is
#         # max(seq_len, receptive_field) - receptive_field + 1.
#         self.source_encoded_time_dim = self._encoded_time_dim(self.his_len)
#         self.target_encoded_time_dim = self._encoded_time_dim(self.pred_len)
#         if self.source_encoded_time_dim <= 0 or self.target_encoded_time_dim <= 0:
#             raise ValueError(
#                 "GWNET_JEPA cannot infer positive encoded time dimensions: "
#                 f"source={self.source_encoded_time_dim}, "
#                 f"target={self.target_encoded_time_dim}."
#             )

#         # JEPA hidden-space predictor. It is registered as self.xxx so optimizer
#         # can update it. Identity initialization is used when dimensions match.
#         self.jepa_time_predictor = nn.Linear(
#             self.source_encoded_time_dim,
#             self.target_encoded_time_dim,
#         )
#         self._init_identity_linear(self.jepa_time_predictor)

#         self.jepa_dim_predictor = nn.Linear(skip_channels, skip_channels)
#         self._init_identity_linear(self.jepa_dim_predictor)

#     def _encoded_time_dim(self, seq_len: int) -> int:
#         padded_len = max(seq_len, self.receptive_field)
#         return padded_len - self.receptive_field + 1

#     @staticmethod
#     def _init_identity_linear(layer: nn.Linear):
#         """Initialize a square Linear layer as identity when possible."""
#         if layer.in_features == layer.out_features:
#             nn.init.eye_(layer.weight)
#             if layer.bias is not None:
#                 nn.init.zeros_(layer.bias)

#     def _build_gwnet_input(
#         self,
#         seq: torch.Tensor,
#         features: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         """
#         Build Graph WaveNet input.

#         Args:
#             seq:      (B, T, N, C_seq), usually C_seq = 1.
#             features: optional temporal/external features:
#                 - (B, T, F): repeated to every node;
#                 - (B, T, N, F): used directly.

#         Returns:
#             inputs: (B, C_in, N, T), where C_in == self.input_dim.
#         """
#         if features is None:
#             x = seq
#         else:
#             if features.dim() == 3:
#                 features = repeat(features, "b t f -> b t n f", n=seq.shape[2])
#             elif features.dim() == 4:
#                 if features.shape[2] == 1 and seq.shape[2] != 1:
#                     features = features.expand(-1, -1, seq.shape[2], -1)
#             else:
#                 raise ValueError(
#                     "GWNET features must have shape (B, T, F) or (B, T, N, F), "
#                     f"but received {tuple(features.shape)}."
#                 )
#             x = torch.concat([seq, features], dim=-1)

#         if x.shape[-1] != self.input_dim:
#             raise ValueError(
#                 f"GWNET expected input feature dim {self.input_dim}, "
#                 f"but received {x.shape[-1]}. Please provide matching features."
#             )

#         return rearrange(x, "b t n f -> b f n t")

#     def _get_supports(self):
#         """Return static supports plus optional adaptive adjacency."""
#         if self.adp_adj:
#             adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
#             new_supports = list(self.supports)
#             new_supports.append(adp)
#             return new_supports
#         return list(self.supports)

#     def encode(
#         self,
#         seq: torch.Tensor,
#         features: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         """
#         Encode input_seq or target_seq into a Graph WaveNet hidden representation.

#         Shapes:
#             seq:        (B, T_or_L, N, C_seq), usually C_seq = 1.
#             features:   optional (B, T_or_L, F) or (B, T_or_L, N, F).
#             internal x: (B, residual_channels, N, padded_T).
#             skip:       (B, skip_channels, N, t_enc).
#             return:     (B, t_enc, N, skip_channels).
#         """
#         inputs = self._build_gwnet_input(seq, features)
#         seq_len = inputs.shape[-1]

#         if seq_len < self.receptive_field:
#             x = nn.functional.pad(inputs, (self.receptive_field - seq_len, 0, 0, 0))
#         else:
#             x = inputs

#         x = self.start_conv(x)
#         skip = None
#         new_supports = self._get_supports()

#         # WaveNet layers. This keeps the original Graph WaveNet computation and
#         # returns the accumulated skip representation before the final output head.
#         for i in range(self.blocks * self.layers):
#             # Preserve the original implementation behavior, which detaches the
#             # residual before filter/gate convolutions.
#             residual = x.detach()

#             filter_out = self.filter_convs[i](residual)
#             filter_out = torch.tanh(filter_out)
#             gate = self.gate_convs[i](residual)
#             gate = torch.sigmoid(gate)
#             x = filter_out * gate

#             s = self.skip_convs[i](x)
#             if skip is not None:
#                 skip = skip[:, :, :, -s.size(3):]
#                 skip = s + skip
#             else:
#                 skip = s

#             x = self.gconv[i](x, new_supports)
#             x = x + residual[:, :, :, -x.size(3):]
#             x = self.bn[i](x)

#         if skip is None:
#             raise RuntimeError("GWNET encode failed to produce skip representation.")

#         # Return a uniform JEPA hidden format: (B, t_enc, N, D).
#         return rearrange(skip, "b d n t -> b t n d")

#     def predict(self, Ex: torch.Tensor) -> torch.Tensor:
#         """
#         Predict target hidden representation from input hidden representation.

#         Shapes:
#             Ex:      (B, t_src, N, D), D = skip_channels.
#             Ey_pred: (B, t_tgt, N, D), matching Ey shape.
#         """
#         if Ex.shape[1] != self.jepa_time_predictor.in_features:
#             raise ValueError(
#                 f"GWNET_JEPA expected encoded source time dim "
#                 f"{self.jepa_time_predictor.in_features}, but received {Ex.shape[1]}."
#             )
#         if Ex.shape[-1] != self.jepa_dim_predictor.in_features:
#             raise ValueError(
#                 f"GWNET_JEPA expected hidden dim {self.jepa_dim_predictor.in_features}, "
#                 f"but received {Ex.shape[-1]}."
#             )

#         x = rearrange(Ex, "b t n d -> b n d t")
#         x = self.jepa_time_predictor(x)
#         x = rearrange(x, "b n d t -> b t n d")
#         x = self.jepa_dim_predictor(x)
#         return x

#     def decode(self, Ey_pred: torch.Tensor) -> torch.Tensor:
#         """
#         Decode hidden representation into value-space forecasting output.

#         Shapes:
#             Ey_pred: (B, t_tgt, N, skip_channels).
#             skip:    (B, skip_channels, N, t_tgt).
#             y_pred:  (B, pred_len, N, output_dim), normally output_dim = 1.
#         """
#         skip = rearrange(Ey_pred, "b t n d -> b d n t")

#         if self.mlp_input_dim > 0:
#             if skip.shape[-1] != self.mlp_input_dim:
#                 raise ValueError(
#                     "GWNET decoder temporal dimension mismatch: "
#                     f"mlp_projection expects {self.mlp_input_dim}, "
#                     f"but received {skip.shape[-1]}. This usually means his_len/pred_len "
#                     "or receptive-field settings need manual confirmation."
#                 )
#             skip = self.mlp_projection(skip)

#         x = F.relu(skip)
#         x = F.relu(self.end_conv_1(x))
#         x = self.end_conv_2(x)
#         return x

#     def forward(
#         self,
#         input_seq: torch.Tensor,
#         input_features: Optional[torch.Tensor] = None,
#         target_seq: Optional[torch.Tensor] = None,
#         target_features: Optional[torch.Tensor] = None,
#         label: Optional[torch.Tensor] = None,
#         mode: str = "finetune",
#         *args,
#         **kwargs,
#     ):
#         """
#         Forward modes:
#             - mode="pretrain": return (Ey, Ey_pred) for JEPA hidden-space loss.
#             - mode="finetune": return y_pred for forecasting loss.

#         Shapes:
#             input_seq: (B, T, N, C_seq), usually C_seq = 1.
#             label/target_seq: (B, L, N, C_seq), usually C_seq = 1.
#             Ex: (B, t_src, N, skip_channels).
#             Ey: (B, t_tgt, N, skip_channels).
#             Ey_pred: (B, t_tgt, N, skip_channels).
#             y_pred: (B, pred_len, N, output_dim).
#         """
#         if mode == "pretrain":
#             label_seq = label if label is not None else target_seq
#             if label_seq is None:
#                 raise ValueError("Labels need to be provided during the pre-training process.")

#             label_features = target_features
#             if label_features is None and input_features is not None:
#                 if input_features.shape[1] == label_seq.shape[1]:
#                     label_features = input_features
#                 else:
#                     raise ValueError(
#                         "target_features must be provided for GWNET pretraining when "
#                         "label length differs from input_features length."
#                     )

#             Ex = self.encode(input_seq, input_features)
#             Ey = self.encode(label_seq, label_features)
#             Ey_pred = self.predict(Ex)

#             if Ey_pred.shape != Ey.shape:
#                 raise ValueError(
#                     f"JEPA hidden shape mismatch: Ey_pred.shape={Ey_pred.shape}, "
#                     f"Ey.shape={Ey.shape}."
#                 )
#             return Ey, Ey_pred

#         if mode == "finetune":
#             Ex = self.encode(input_seq, input_features)
#             Ey_pred = self.predict(Ex)
#             y_pred = self.decode(Ey_pred)
#             return y_pred

#         raise ValueError(f"Unsupported mode: {mode}")


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
#     def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
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
#             for _ in range(2, self.order + 1):
#                 x2 = self.nconv(x1, a)
#                 out.append(x2)
#                 x1 = x2

#         h = torch.cat(out, dim=1)
#         h = self.mlp(h)
#         h = F.dropout(h, self.dropout, training=self.training)
#         return h


# # Optional alias for users who prefer a distinct class name in config files.
# GWNET_JEPA = GWNET

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn import Linear, ReLU, Sequential

from src.base.model import BaseModel


class GWNET(BaseModel):
    """
    JEPA-compatible Graph WaveNet.

    Reference:
        https://github.com/nnzhan/Graph-WaveNet

    Structure
    ---------
    encode:
        input
        -> start_conv
        -> dilated gated temporal convolutions
        -> graph convolutions
        -> skip aggregation
        -> hidden representation

    predict:
        Ex
        -> temporal predictor
        -> hidden-dimension predictor
        -> Ey_pred

    decode:
        Ey_pred
        -> optional temporal projection
        -> Graph WaveNet output head
        -> forecasting output

    Expected input shapes
    ---------------------
    input_seq:
        (B, T, N, C_seq)

    target_seq:
        (B, L, N, C_seq)

    input_features:
        (B, T, F)
        or
        (B, T, N, F)

    target_features:
        (B, L, F)
        or
        (B, L, N, F)

    Hidden representation
    ---------------------
    Ex / Ey:
        (B, T_enc, N, skip_channels)

    Forward modes
    -------------
    mode="pretrain":
        return Ey, Ey_pred

    mode="finetune":
        return y_pred
    """

    def __init__(
        self,
        supports,
        adp_adj,
        dropout,
        residual_channels,
        dilation_channels,
        skip_channels,
        end_channels,
        kernel_size,
        blocks,
        layers,
        **args,
    ):
        super(GWNET, self).__init__(**args)

        # ---------------------------------------------------------
        # Basic configuration
        # ---------------------------------------------------------
        self.supports = list(supports)
        self.supports_len = len(self.supports)

        self.adp_adj = adp_adj
        self.dropout = dropout

        self.blocks = blocks
        self.layers = layers
        self.kernel_size = kernel_size

        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.end_channels = end_channels

        # ---------------------------------------------------------
        # WaveNet modules
        # ---------------------------------------------------------
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        # ---------------------------------------------------------
        # Input projection
        # ---------------------------------------------------------
        self.start_conv = nn.Conv2d(
            in_channels=self.input_dim,
            out_channels=residual_channels,
            kernel_size=(1, 1),
        )

        # ---------------------------------------------------------
        # Adaptive adjacency
        # ---------------------------------------------------------
        if self.adp_adj:
            self.nodevec1 = nn.Parameter(
                torch.randn(
                    self.node_num,
                    10,
                )
            )

            self.nodevec2 = nn.Parameter(
                torch.randn(
                    10,
                    self.node_num,
                )
            )

            self.supports_len += 1

        # ---------------------------------------------------------
        # Build dilated WaveNet stack
        # ---------------------------------------------------------
        receptive_field = 1

        for _ in range(self.blocks):

            additional_scope = self.kernel_size - 1
            dilation = 1

            for _ in range(self.layers):

                # -----------------------------
                # Temporal filter branch
                # -----------------------------
                self.filter_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, self.kernel_size),
                        dilation=(1, dilation),
                        stride=(1, 1),
                    )
                )

                # -----------------------------
                # Temporal gate branch
                # -----------------------------
                self.gate_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, self.kernel_size),
                        dilation=(1, dilation),
                        stride=(1, 1),
                    )
                )

                # -----------------------------
                # Skip branch
                # -----------------------------
                self.skip_convs.append(
                    nn.Conv2d(
                        in_channels=dilation_channels,
                        out_channels=skip_channels,
                        kernel_size=(1, 1),
                    )
                )

                # -----------------------------
                # Graph convolution
                # -----------------------------
                self.gconv.append(
                    GCN(
                        c_in=dilation_channels,
                        c_out=residual_channels,
                        dropout=dropout,
                        support_len=self.supports_len,
                    )
                )

                # -----------------------------
                # Normalization
                # -----------------------------
                self.bn.append(
                    nn.BatchNorm2d(
                        residual_channels
                    )
                )

                # Update receptive field
                receptive_field += additional_scope

                dilation *= 2
                additional_scope *= 2

        self.receptive_field = receptive_field

        # ---------------------------------------------------------
        # Encoded temporal dimensions
        # ---------------------------------------------------------
        #
        # After the complete dilated convolution stack:
        #
        # T_enc =
        #   max(T, receptive_field)
        #   - receptive_field
        #   + 1
        #
        self.source_encoded_time_dim = self._encoded_time_dim(
            self.his_len
        )

        self.target_encoded_time_dim = self._encoded_time_dim(
            self.pred_len
        )

        if self.source_encoded_time_dim <= 0:
            raise ValueError(
                "Invalid GWNET source encoded time dimension: "
                f"{self.source_encoded_time_dim}"
            )

        if self.target_encoded_time_dim <= 0:
            raise ValueError(
                "Invalid GWNET target encoded time dimension: "
                f"{self.target_encoded_time_dim}"
            )

        # ---------------------------------------------------------
        # JEPA predictor
        # ---------------------------------------------------------
        #
        # Time predictor:
        #
        # (B, N, D, T_src)
        #       ->
        # (B, N, D, T_tgt)
        #
        self.jepa_time_predictor = nn.Linear(
            self.source_encoded_time_dim,
            self.target_encoded_time_dim,
        )

        self._init_identity_linear(
            self.jepa_time_predictor
        )

        # Hidden/channel predictor
        self.jepa_dim_predictor = nn.Linear(
            skip_channels,
            skip_channels,
        )

        self._init_identity_linear(
            self.jepa_dim_predictor
        )

        # ---------------------------------------------------------
        # Temporal projection before Graph WaveNet output head
        # ---------------------------------------------------------
        #
        # Standard Graph WaveNet short setting:
        #
        #     target_encoded_time_dim = 1
        #
        # so no projection is needed.
        #
        # For long forecasting, e.g.:
        #
        #     96 -> 84 encoded steps
        #
        # we project:
        #
        #     84 -> 1
        #
        # before using the original Conv2d output head.
        #
        if self.target_encoded_time_dim > 1:

            self.temporal_projection = Sequential(
                Linear(
                    self.target_encoded_time_dim,
                    64,
                ),
                ReLU(),

                Linear(
                    64,
                    128,
                ),
                ReLU(),

                Linear(
                    128,
                    64,
                ),
                ReLU(),

                Linear(
                    64,
                    1,
                ),
            )

        else:
            self.temporal_projection = None

        # ---------------------------------------------------------
        # Graph WaveNet output head
        # ---------------------------------------------------------
        self.end_conv_1 = nn.Conv2d(
            in_channels=skip_channels,
            out_channels=end_channels,
            kernel_size=(1, 1),
            bias=True,
        )

        # General form:
        #
        # channel =
        #     pred_len * output_dim
        #
        self.end_conv_2 = nn.Conv2d(
            in_channels=end_channels,
            out_channels=self.pred_len * self.output_dim,
            kernel_size=(1, 1),
            bias=True,
        )

    # ============================================================
    # Utility
    # ============================================================

    def _encoded_time_dim(
        self,
        seq_len: int,
    ) -> int:
        """
        Compute temporal length after the complete dilated WaveNet
        convolution stack.

        Input shorter than receptive field is left-padded first.

        Therefore:

            padded_len = max(seq_len, receptive_field)

            T_enc =
                padded_len
                - receptive_field
                + 1
        """

        padded_len = max(
            seq_len,
            self.receptive_field,
        )

        encoded_len = (
            padded_len
            - self.receptive_field
            + 1
        )

        return encoded_len

    @staticmethod
    def _init_identity_linear(
        layer: nn.Linear,
    ):
        """
        Initialize a square Linear layer as identity.
        """

        if (
            layer.in_features
            == layer.out_features
        ):
            nn.init.eye_(
                layer.weight
            )

            if layer.bias is not None:
                nn.init.zeros_(
                    layer.bias
                )

    # ============================================================
    # Input construction
    # ============================================================

    def _build_gwnet_input(
        self,
        seq: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Convert common project input format into Graph WaveNet format.

        seq:
            (B, T, N, C_seq)

        features:
            (B, T, F)
            or
            (B, T, N, F)

        return:
            (B, C_in, N, T)
        """

        if seq.dim() != 4:
            raise ValueError(
                "GWNET expects seq with shape "
                "(B, T, N, C), "
                f"but received {tuple(seq.shape)}."
            )

        if features is None:

            x = seq

        else:

            if features.dim() == 3:

                # (B, T, F)
                # ->
                # (B, T, N, F)
                features = repeat(
                    features,
                    "b t f -> b t n f",
                    n=seq.shape[2],
                )

            elif features.dim() == 4:

                if (
                    features.shape[2] == 1
                    and seq.shape[2] != 1
                ):

                    features = features.expand(
                        -1,
                        -1,
                        seq.shape[2],
                        -1,
                    )

                elif (
                    features.shape[2]
                    != seq.shape[2]
                ):
                    raise ValueError(
                        "GWNET node dimension mismatch: "
                        f"seq={tuple(seq.shape)}, "
                        f"features={tuple(features.shape)}"
                    )

            else:
                raise ValueError(
                    "GWNET features must have shape "
                    "(B, T, F) or (B, T, N, F), "
                    f"but received {tuple(features.shape)}."
                )

            if (
                features.shape[0]
                != seq.shape[0]
                or features.shape[1]
                != seq.shape[1]
            ):
                raise ValueError(
                    "GWNET feature temporal/batch dimensions "
                    "do not match seq: "
                    f"seq={tuple(seq.shape)}, "
                    f"features={tuple(features.shape)}"
                )

            x = torch.concat(
                [
                    seq,
                    features,
                ],
                dim=-1,
            )

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                "GWNET input feature dimension mismatch: "
                f"expected {self.input_dim}, "
                f"received {x.shape[-1]}."
            )

        # (B, T, N, F)
        # ->
        # (B, F, N, T)
        x = rearrange(
            x,
            "b t n f -> b f n t",
        )

        return x

    # ============================================================
    # Graph support
    # ============================================================

    def _get_supports(self):
        """
        Return static graph supports together with the optional
        adaptive adjacency matrix.
        """

        supports = list(
            self.supports
        )

        if self.adp_adj:

            adaptive_adj = F.softmax(
                F.relu(
                    torch.mm(
                        self.nodevec1,
                        self.nodevec2,
                    )
                ),
                dim=1,
            )

            supports.append(
                adaptive_adj
            )

        return supports

    # ============================================================
    # Encoder
    # ============================================================

    def encode(
        self,
        seq: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode sequence into Graph WaveNet hidden representation.

        Input:
            seq:
                (B, T, N, C)

        Output:
            hidden:
                (B, T_enc, N, skip_channels)
        """

        inputs = self._build_gwnet_input(
            seq,
            features,
        )

        seq_len = inputs.shape[-1]

        # --------------------------------------------------------
        # Left padding
        # --------------------------------------------------------
        if seq_len < self.receptive_field:

            x = F.pad(
                inputs,
                (
                    self.receptive_field
                    - seq_len,
                    0,
                    0,
                    0,
                ),
            )

        else:
            x = inputs

        # --------------------------------------------------------
        # Input projection
        # --------------------------------------------------------
        x = self.start_conv(
            x
        )

        skip = None

        supports = self._get_supports()

        # --------------------------------------------------------
        # WaveNet blocks
        # --------------------------------------------------------
        for i in range(
            self.blocks
            * self.layers
        ):

            # ====================================================
            # IMPORTANT FIX
            # ====================================================
            #
            # WRONG:
            #
            #     residual = x.detach()
            #
            # This breaks end-to-end gradient propagation through
            # Graph WaveNet.
            #
            # Correct Graph WaveNet behavior:
            #
            residual = x

            # ----------------------------------------------------
            # Dilated gated temporal convolution
            # ----------------------------------------------------
            filter_out = self.filter_convs[i](
                residual
            )

            filter_out = torch.tanh(
                filter_out
            )

            gate_out = self.gate_convs[i](
                residual
            )

            gate_out = torch.sigmoid(
                gate_out
            )

            x = (
                filter_out
                * gate_out
            )

            # ----------------------------------------------------
            # Skip connection
            # ----------------------------------------------------
            skip_i = self.skip_convs[i](
                x
            )

            if skip is None:

                skip = skip_i

            else:

                # Different dilation levels have different
                # temporal lengths.
                skip = skip[
                    :,
                    :,
                    :,
                    -skip_i.size(3):,
                ]

                skip = (
                    skip
                    + skip_i
                )

            # ----------------------------------------------------
            # Graph convolution
            # ----------------------------------------------------
            x = self.gconv[i](
                x,
                supports,
            )

            # ----------------------------------------------------
            # Residual connection
            # ----------------------------------------------------
            residual = residual[
                :,
                :,
                :,
                -x.size(3):,
            ]

            x = (
                x
                + residual
            )

            # ----------------------------------------------------
            # Batch normalization
            # ----------------------------------------------------
            x = self.bn[i](
                x
            )

        if skip is None:
            raise RuntimeError(
                "GWNET encoder failed to produce "
                "a skip representation."
            )

        # (B, D, N, T)
        # ->
        # (B, T, N, D)
        hidden = rearrange(
            skip,
            "b d n t -> b t n d",
        )

        expected_time_dim = (
            self._encoded_time_dim(
                seq.shape[1]
            )
        )

        if hidden.shape[1] != expected_time_dim:
            raise RuntimeError(
                "GWNET encoded temporal dimension mismatch: "
                f"expected {expected_time_dim}, "
                f"received {hidden.shape[1]}."
            )

        return hidden

    # ============================================================
    # JEPA predictor
    # ============================================================

    def predict(
        self,
        Ex: torch.Tensor,
    ) -> torch.Tensor:
        """
        Map source hidden representation into target hidden space.

        Ex:
            (B, T_src, N, D)

        Ey_pred:
            (B, T_tgt, N, D)
        """

        if Ex.dim() != 4:
            raise ValueError(
                "GWNET predictor expects Ex with shape "
                "(B, T, N, D), "
                f"received {tuple(Ex.shape)}."
            )

        if (
            Ex.shape[1]
            != self.jepa_time_predictor.in_features
        ):
            raise ValueError(
                "GWNET source encoded temporal dimension mismatch: "
                f"expected "
                f"{self.jepa_time_predictor.in_features}, "
                f"received {Ex.shape[1]}."
            )

        if (
            Ex.shape[-1]
            != self.skip_channels
        ):
            raise ValueError(
                "GWNET hidden dimension mismatch: "
                f"expected {self.skip_channels}, "
                f"received {Ex.shape[-1]}."
            )

        # ---------------------------------------------
        # Temporal prediction
        # ---------------------------------------------
        #
        # (B, T, N, D)
        # ->
        # (B, N, D, T)
        #
        x = rearrange(
            Ex,
            "b t n d -> b n d t",
        )

        x = self.jepa_time_predictor(
            x
        )

        # ->
        # (B, T_target, N, D)
        x = rearrange(
            x,
            "b n d t -> b t n d",
        )

        # ---------------------------------------------
        # Hidden-space prediction
        # ---------------------------------------------
        x = self.jepa_dim_predictor(
            x
        )

        return x

    # ============================================================
    # Decoder
    # ============================================================

    def decode(
        self,
        Ey_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode JEPA hidden representation into forecasting values.

        Ey_pred:
            (B, T_target, N, skip_channels)

        return:
            (B, pred_len, N, output_dim)
        """

        if Ey_pred.dim() != 4:
            raise ValueError(
                "GWNET decoder expects Ey_pred with shape "
                "(B, T, N, D), "
                f"received {tuple(Ey_pred.shape)}."
            )

        if (
            Ey_pred.shape[1]
            != self.target_encoded_time_dim
        ):
            raise ValueError(
                "GWNET target encoded temporal dimension mismatch: "
                f"expected {self.target_encoded_time_dim}, "
                f"received {Ey_pred.shape[1]}."
            )

        if (
            Ey_pred.shape[-1]
            != self.skip_channels
        ):
            raise ValueError(
                "GWNET decoder hidden dimension mismatch: "
                f"expected {self.skip_channels}, "
                f"received {Ey_pred.shape[-1]}."
            )

        # (B, T, N, D)
        # ->
        # (B, D, N, T)
        x = rearrange(
            Ey_pred,
            "b t n d -> b d n t",
        )

        # --------------------------------------------------------
        # Long-sequence temporal projection
        # --------------------------------------------------------
        if self.temporal_projection is not None:

            if (
                x.shape[-1]
                != self.target_encoded_time_dim
            ):
                raise ValueError(
                    "GWNET temporal projection input mismatch: "
                    f"expected {self.target_encoded_time_dim}, "
                    f"received {x.shape[-1]}."
                )

            # Linear works on the last dimension:
            #
            # (B, D, N, T_enc)
            # ->
            # (B, D, N, 1)
            x = self.temporal_projection(
                x
            )

        # Standard short GWNet should already have T == 1.
        if x.shape[-1] != 1:
            raise RuntimeError(
                "GWNET output head expects temporal dimension 1 "
                "after temporal projection, "
                f"but received {x.shape[-1]}."
            )

        # --------------------------------------------------------
        # Original Graph WaveNet output head
        # --------------------------------------------------------
        x = F.relu(
            x
        )

        x = self.end_conv_1(
            x
        )

        x = F.relu(
            x
        )

        x = self.end_conv_2(
            x
        )

        # Current shape:
        #
        # (B, pred_len * output_dim, N, 1)
        #
        x = x.squeeze(
            -1
        )

        # --------------------------------------------------------
        # Restore common project output layout
        # --------------------------------------------------------
        #
        # (B, pred_len * output_dim, N)
        #
        # ->
        #
        # (B, pred_len, N, output_dim)
        #
        x = rearrange(
            x,
            "b (t c) n -> b t n c",
            t=self.pred_len,
            c=self.output_dim,
        )

        return x

    # ============================================================
    # Forward
    # ============================================================

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
        mode="pretrain"
        ----------------
        Online path:

            input_seq
                -> encode
                -> Ex
                -> predict
                -> Ey_pred

        The ordinary model-side target representation is also returned
        here for compatibility with callers that directly use
        model(..., mode="pretrain").

        The current JEPAPretrainEngine may instead construct Ey through
        its EMA target model.

        mode="finetune"
        ----------------
            input_seq
                -> encode
                -> predict
                -> decode
                -> y_pred
        """

        # ========================================================
        # JEPA pretraining
        # ========================================================
        if mode == "pretrain":

            label_seq = (
                label
                if label is not None
                else target_seq
            )

            if label_seq is None:
                raise ValueError(
                    "target_seq or label must be provided "
                    "during GWNET JEPA pretraining."
                )

            label_features = target_features

            if (
                label_features is None
                and input_features is not None
            ):

                if (
                    input_features.shape[1]
                    == label_seq.shape[1]
                ):
                    label_features = input_features

                else:
                    raise ValueError(
                        "target_features must be provided "
                        "when target sequence length differs "
                        "from input_features length."
                    )

            Ex = self.encode(
                input_seq,
                input_features,
            )

            Ey = self.encode(
                label_seq,
                label_features,
            )

            Ey_pred = self.predict(
                Ex
            )

            if (
                Ey_pred.shape
                != Ey.shape
            ):
                raise ValueError(
                    "GWNET JEPA hidden shape mismatch: "
                    f"Ey_pred={tuple(Ey_pred.shape)}, "
                    f"Ey={tuple(Ey.shape)}."
                )

            return (
                Ey,
                Ey_pred,
            )

        # ========================================================
        # Downstream forecasting
        # ========================================================
        elif mode == "finetune":

            Ex = self.encode(
                input_seq,
                input_features,
            )

            Ey_pred = self.predict(
                Ex
            )

            y_pred = self.decode(
                Ey_pred
            )

            return y_pred

        else:
            raise ValueError(
                f"Unsupported GWNET mode: {mode}"
            )


# ================================================================
# Graph convolution components
# ================================================================


class nconv(nn.Module):
    """
    Neighborhood convolution.

    x:
        (B, C, N, T)

    A:
        (N, N)
    """

    def __init__(self):
        super(nconv, self).__init__()

    def forward(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
    ) -> torch.Tensor:

        x = torch.einsum(
            "ncvl,vw->ncwl",
            x,
            A,
        )

        return x.contiguous()


class linear(nn.Module):
    """
    1x1 convolution used after concatenated diffusion features.
    """

    def __init__(
        self,
        c_in,
        c_out,
    ):
        super(linear, self).__init__()

        self.mlp = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=(1, 1),
            padding=(0, 0),
            stride=(1, 1),
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.mlp(
            x
        )


class GCN(nn.Module):
    """
    Diffusion graph convolution used by Graph WaveNet.

    For every graph support A:

        X
        A X
        A^2 X
        ...
        A^K X

    are concatenated along the channel dimension and projected
    through a 1x1 convolution.
    """

    def __init__(
        self,
        c_in,
        c_out,
        dropout,
        support_len=3,
        order=2,
    ):
        super(GCN, self).__init__()

        self.nconv = nconv()

        self.order = order
        self.dropout = dropout

        expanded_channels = (
            order
            * support_len
            + 1
        ) * c_in

        self.mlp = linear(
            expanded_channels,
            c_out,
        )

    def forward(
        self,
        x: torch.Tensor,
        support,
    ) -> torch.Tensor:

        out = [
            x
        ]

        for A in support:

            x1 = self.nconv(
                x,
                A,
            )

            out.append(
                x1
            )

            for _ in range(
                2,
                self.order + 1,
            ):

                x2 = self.nconv(
                    x1,
                    A,
                )

                out.append(
                    x2
                )

                x1 = x2

        h = torch.cat(
            out,
            dim=1,
        )

        h = self.mlp(
            h
        )

        h = F.dropout(
            h,
            self.dropout,
            training=self.training,
        )

        return h