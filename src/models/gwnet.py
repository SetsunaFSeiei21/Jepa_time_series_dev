# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import repeat
# from torch.nn import Linear, ReLU, Sequential

# from src.base.model import BaseModel


# class GWNET(BaseModel):
#     """
#     Reference code: https://github.com/nnzhan/Graph-WaveNet
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
#         **args
#     ):
#         super(GWNET, self).__init__(**args)
#         self.supports = supports
#         self.supports_len = len(supports)
#         self.adp_adj = adp_adj
#         # print("check supports length", len(supports), self.supports_len)

#         self.dropout = dropout
#         self.blocks = blocks
#         self.layers = layers
#         self.kernel_size = kernel_size

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

#         for b in range(self.blocks):
#             additional_scope = self.kernel_size - 1
#             new_dilation = 1
#             for i in range(self.layers):
#                 # dilated convolutions
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

#                 # 1x1 convolution for residual connection
#                 self.residual_convs.append(
#                     nn.Conv1d(
#                         in_channels=dilation_channels,
#                         out_channels=residual_channels,
#                         kernel_size=(1, 1),
#                     )
#                 )

#                 # 1x1 convolution for skip connection
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

#     def forward(self, input_seq, input_features, *args, **kwargs):
#         # (b, t, n, f)
#         x = torch.concat(
#             [
#                 input_seq,
#                 repeat(input_features, "b t f -> b t n f", n=input_seq.shape[2]),
#             ],
#             dim=-1,
#         )
#         inputs = x.permute(0, 3, 2, 1)

#         if self.his_len < self.receptive_field:
#             x = nn.functional.pad(
#                 inputs, (self.receptive_field - self.his_len, 0, 0, 0)
#             )
#         else:
#             x = inputs

#         x = self.start_conv(x)
#         skip = 0

#         # print(f'x after start_conv:{x.shape}')

#         # calculate the current adaptive adj matrix once per iteration
#         new_supports = None
#         if self.adp_adj:
#             adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
#             new_supports = self.supports.copy()
#             new_supports.append(adp)
#         else:
#             new_supports = self.supports.copy()

#         # WaveNet layers
#         for i in range(self.blocks * self.layers):

#             #            |----------------------------------------|     *residual*
#             #            |                                        |
#             #            |    |-- conv -- tanh --|                |
#             # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
#             #                 |-- conv -- sigm --|     |
#             #                                         1x1
#             #                                          |
#             # ---------------------------------------> + ------------->	*skip*

#             # (dilation, init_dilation) = self.dilations[i]

#             # residual = dilation_func(x, dilation, init_dilation, i)

#             residual = x.detach()

#             # print(f'x_residual:{x.shape}')

#             # dilated convolution
#             filter = self.filter_convs[i](residual)
#             # print(f'filter:{filter.shape}')

#             filter = torch.tanh(filter)
#             gate = self.gate_convs[i](residual)
#             gate = torch.sigmoid(gate)
#             # print(gate.shape)
#             x = filter * gate

#             s = x
#             s = self.skip_convs[i](s)
#             try:
#                 skip = skip[:, :, :, -s.size(3) :]
#             except:
#                 skip = 0
#             skip = s + skip

#             # print(f'skip_x:{x.shape}') # [64, 32, 1085, 12]

#             x = self.gconv[i](x, new_supports)

#             # print(f'GCN_x:{x.shape}')

#             x = x + residual[:, :, :, -x.size(3) :]

#             x = self.bn[i](x)

#         # (b, 256, 1085, 12) --> (b, 256, 1085)
#         if self.mlp_input_dim > 0:
#             skip = self.mlp_projection(skip)

#         x = F.relu(skip)
#         x = F.relu(self.end_conv_1(x))

#         x = self.end_conv_2(x)

#         return x


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
#             for k in range(2, self.order + 1):
#                 x2 = self.nconv(x1, a)
#                 out.append(x2)
#                 x1 = x2

#         h = torch.cat(out, dim=1)
#         h = self.mlp(h)
#         h = F.dropout(h, self.dropout, training=self.training)
#         return h

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn import Linear, ReLU, Sequential

from src.base.model import BaseModel


class GWNET(BaseModel):
    """
    JEPA-compatible Graph WaveNet backbone.

    Reference code: https://github.com/nnzhan/Graph-WaveNet

    Refactor notes:
    - encode: input/target sequence + optional time features -> WaveNet skip hidden representation.
    - predict: lightweight trainable hidden-space predictor. It maps encoded input
      hidden states to the encoded target hidden-state shape.
    - decode: original Graph WaveNet output head, i.e. optional temporal MLP projection
      followed by end_conv_1 and end_conv_2.

    Expected data shapes:
    - input_seq:      (B, T, N, C_seq), usually C_seq = 1.
    - target_seq:     (B, L, N, C_seq), usually C_seq = 1.
    - input_features: (B, T, F) or (B, T, N, F), optional.
    - target_features:(B, L, F) or (B, L, N, F), optional.

    Hidden representation shape returned by encode:
    - Ex/Ey:          (B, t_enc, N, skip_channels).

    Forward modes:
    - mode="pretrain": return (Ey, Ey_pred), where Ey_pred.shape == Ey.shape.
    - mode="finetune": return y_pred with the original forecasting output shape.
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
        self.supports = supports
        self.supports_len = len(supports)
        self.adp_adj = adp_adj

        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.kernel_size = kernel_size
        self.skip_channels = skip_channels

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(
            in_channels=self.input_dim,
            out_channels=residual_channels,
            kernel_size=(1, 1),
        )
        receptive_field = 1

        if self.adp_adj:
            self.nodevec1 = nn.Parameter(
                torch.randn(self.node_num, 10), requires_grad=True
            )
            self.nodevec2 = nn.Parameter(
                torch.randn(10, self.node_num), requires_grad=True
            )
            self.supports_len += 1

        for _ in range(self.blocks):
            additional_scope = self.kernel_size - 1
            new_dilation = 1
            for _ in range(self.layers):
                # Dilated temporal convolutions.
                self.filter_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, self.kernel_size),
                        dilation=new_dilation,
                        stride=1,
                    )
                )

                self.gate_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, self.kernel_size),
                        dilation=new_dilation,
                        stride=1,
                    )
                )

                # Kept for compatibility with the original module definition.
                # The original forward path does not explicitly call residual_convs.
                self.residual_convs.append(
                    nn.Conv1d(
                        in_channels=dilation_channels,
                        out_channels=residual_channels,
                        kernel_size=(1, 1),
                    )
                )

                # 1x1 convolution for skip connection.
                self.skip_convs.append(
                    nn.Conv2d(
                        in_channels=dilation_channels,
                        out_channels=skip_channels,
                        kernel_size=(1, 1),
                    )
                )
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                self.gconv.append(
                    GCN(
                        dilation_channels,
                        residual_channels,
                        dropout,
                        support_len=self.supports_len,
                    )
                )

        self.end_conv_1 = nn.Conv2d(
            in_channels=skip_channels,
            out_channels=end_channels,
            kernel_size=(1, 1),
            bias=True,
        )

        self.end_conv_2 = nn.Conv2d(
            in_channels=end_channels,
            out_channels=self.pred_len,
            kernel_size=(1, 1),
            bias=True,
        )

        self.receptive_field = receptive_field

        # This follows the original implementation. For the current repository
        # configs, it matches the encoded temporal length for long forecasting;
        # for short forecasting it is <= 0 and the MLP projection is skipped.
        self.mlp_input_dim = (
            self.his_len - (self.kernel_size - 1) * (1 + self.layers) * blocks
        )
        if self.mlp_input_dim > 0:
            self.mlp_projection = Sequential(
                Linear(self.mlp_input_dim, 64),
                ReLU(),
                Linear(64, 128),
                ReLU(),
                Linear(128, 64),
                ReLU(),
                Linear(64, self.output_dim),
            )

        # Encoded time length after the dilated WaveNet stack. The stack pads
        # sequences shorter than the receptive field, so the encoded length is
        # max(seq_len, receptive_field) - receptive_field + 1.
        self.source_encoded_time_dim = self._encoded_time_dim(self.his_len)
        self.target_encoded_time_dim = self._encoded_time_dim(self.pred_len)
        if self.source_encoded_time_dim <= 0 or self.target_encoded_time_dim <= 0:
            raise ValueError(
                "GWNET_JEPA cannot infer positive encoded time dimensions: "
                f"source={self.source_encoded_time_dim}, "
                f"target={self.target_encoded_time_dim}."
            )

        # JEPA hidden-space predictor. It is registered as self.xxx so optimizer
        # can update it. Identity initialization is used when dimensions match.
        self.jepa_time_predictor = nn.Linear(
            self.source_encoded_time_dim,
            self.target_encoded_time_dim,
        )
        self._init_identity_linear(self.jepa_time_predictor)

        self.jepa_dim_predictor = nn.Linear(skip_channels, skip_channels)
        self._init_identity_linear(self.jepa_dim_predictor)

    def _encoded_time_dim(self, seq_len: int) -> int:
        padded_len = max(seq_len, self.receptive_field)
        return padded_len - self.receptive_field + 1

    @staticmethod
    def _init_identity_linear(layer: nn.Linear):
        """Initialize a square Linear layer as identity when possible."""
        if layer.in_features == layer.out_features:
            nn.init.eye_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def _build_gwnet_input(
        self,
        seq: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build Graph WaveNet input.

        Args:
            seq:      (B, T, N, C_seq), usually C_seq = 1.
            features: optional temporal/external features:
                - (B, T, F): repeated to every node;
                - (B, T, N, F): used directly.

        Returns:
            inputs: (B, C_in, N, T), where C_in == self.input_dim.
        """
        if features is None:
            x = seq
        else:
            if features.dim() == 3:
                features = repeat(features, "b t f -> b t n f", n=seq.shape[2])
            elif features.dim() == 4:
                if features.shape[2] == 1 and seq.shape[2] != 1:
                    features = features.expand(-1, -1, seq.shape[2], -1)
            else:
                raise ValueError(
                    "GWNET features must have shape (B, T, F) or (B, T, N, F), "
                    f"but received {tuple(features.shape)}."
                )
            x = torch.concat([seq, features], dim=-1)

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"GWNET expected input feature dim {self.input_dim}, "
                f"but received {x.shape[-1]}. Please provide matching features."
            )

        return rearrange(x, "b t n f -> b f n t")

    def _get_supports(self):
        """Return static supports plus optional adaptive adjacency."""
        if self.adp_adj:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = list(self.supports)
            new_supports.append(adp)
            return new_supports
        return list(self.supports)

    def encode(
        self,
        seq: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode input_seq or target_seq into a Graph WaveNet hidden representation.

        Shapes:
            seq:        (B, T_or_L, N, C_seq), usually C_seq = 1.
            features:   optional (B, T_or_L, F) or (B, T_or_L, N, F).
            internal x: (B, residual_channels, N, padded_T).
            skip:       (B, skip_channels, N, t_enc).
            return:     (B, t_enc, N, skip_channels).
        """
        inputs = self._build_gwnet_input(seq, features)
        seq_len = inputs.shape[-1]

        if seq_len < self.receptive_field:
            x = nn.functional.pad(inputs, (self.receptive_field - seq_len, 0, 0, 0))
        else:
            x = inputs

        x = self.start_conv(x)
        skip = None
        new_supports = self._get_supports()

        # WaveNet layers. This keeps the original Graph WaveNet computation and
        # returns the accumulated skip representation before the final output head.
        for i in range(self.blocks * self.layers):
            # Preserve the original implementation behavior, which detaches the
            # residual before filter/gate convolutions.
            residual = x.detach()

            filter_out = self.filter_convs[i](residual)
            filter_out = torch.tanh(filter_out)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter_out * gate

            s = self.skip_convs[i](x)
            if skip is not None:
                skip = skip[:, :, :, -s.size(3):]
                skip = s + skip
            else:
                skip = s

            x = self.gconv[i](x, new_supports)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        if skip is None:
            raise RuntimeError("GWNET encode failed to produce skip representation.")

        # Return a uniform JEPA hidden format: (B, t_enc, N, D).
        return rearrange(skip, "b d n t -> b t n d")

    def predict(self, Ex: torch.Tensor) -> torch.Tensor:
        """
        Predict target hidden representation from input hidden representation.

        Shapes:
            Ex:      (B, t_src, N, D), D = skip_channels.
            Ey_pred: (B, t_tgt, N, D), matching Ey shape.
        """
        if Ex.shape[1] != self.jepa_time_predictor.in_features:
            raise ValueError(
                f"GWNET_JEPA expected encoded source time dim "
                f"{self.jepa_time_predictor.in_features}, but received {Ex.shape[1]}."
            )
        if Ex.shape[-1] != self.jepa_dim_predictor.in_features:
            raise ValueError(
                f"GWNET_JEPA expected hidden dim {self.jepa_dim_predictor.in_features}, "
                f"but received {Ex.shape[-1]}."
            )

        x = rearrange(Ex, "b t n d -> b n d t")
        x = self.jepa_time_predictor(x)
        x = rearrange(x, "b n d t -> b t n d")
        x = self.jepa_dim_predictor(x)
        return x

    def decode(self, Ey_pred: torch.Tensor) -> torch.Tensor:
        """
        Decode hidden representation into value-space forecasting output.

        Shapes:
            Ey_pred: (B, t_tgt, N, skip_channels).
            skip:    (B, skip_channels, N, t_tgt).
            y_pred:  (B, pred_len, N, output_dim), normally output_dim = 1.
        """
        skip = rearrange(Ey_pred, "b t n d -> b d n t")

        if self.mlp_input_dim > 0:
            if skip.shape[-1] != self.mlp_input_dim:
                raise ValueError(
                    "GWNET decoder temporal dimension mismatch: "
                    f"mlp_projection expects {self.mlp_input_dim}, "
                    f"but received {skip.shape[-1]}. This usually means his_len/pred_len "
                    "or receptive-field settings need manual confirmation."
                )
            skip = self.mlp_projection(skip)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x

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
            label/target_seq: (B, L, N, C_seq), usually C_seq = 1.
            Ex: (B, t_src, N, skip_channels).
            Ey: (B, t_tgt, N, skip_channels).
            Ey_pred: (B, t_tgt, N, skip_channels).
            y_pred: (B, pred_len, N, output_dim).
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
                        "target_features must be provided for GWNET pretraining when "
                        "label length differs from input_features length."
                    )

            Ex = self.encode(input_seq, input_features)
            Ey = self.encode(label_seq, label_features)
            Ey_pred = self.predict(Ex)

            if Ey_pred.shape != Ey.shape:
                raise ValueError(
                    f"JEPA hidden shape mismatch: Ey_pred.shape={Ey_pred.shape}, "
                    f"Ey.shape={Ey.shape}."
                )
            return Ey, Ey_pred

        if mode == "finetune":
            Ex = self.encode(input_seq, input_features)
            Ey_pred = self.predict(Ex)
            y_pred = self.decode(Ey_pred)
            return y_pred

        raise ValueError(f"Unsupported mode: {mode}")


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
            c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True
        )

    def forward(self, x):
        return self.mlp(x)


class GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
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
            for _ in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


# # Optional alias for users who prefer a distinct class name in config files.
# GWNET_JEPA = GWNET
