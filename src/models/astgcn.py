import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional

from src.base.model import BaseModel


class ASTGCN(BaseModel):
    """
    ASTGCN backbone refactored for JEPA-style pretraining and downstream finetuning.

    Reference code: https://github.com/guoshnBJTU/ASTGCN-r-pytorch

    Refactor notes:
    - encode: input/target sequence + optional temporal features -> ASTGCN hidden representation.
    - predict: lightweight trainable predictor in hidden space. It preserves the hidden shape
      and is initialized as identity when dimensions match.
    - decode: original final_conv forecasting head -> value-space prediction.

    Important limitation:
    ASTGCN's temporal/spatial attention layers are initialized with fixed seq_len parameters.
    Therefore, this implementation only supports JEPA pretraining when the sequence passed to
    encode has length self.his_len. In the current short/long settings, his_len == pred_len,
    so input_seq and target_seq can both be encoded by the same backbone. If his_len != pred_len,
    a separate target encoder or dynamically parameterized attention layers are required.
    """

    def __init__(
        self,
        cheb_poly,
        order,
        nb_block,
        nb_chev_filter,
        nb_time_filter,
        time_stride,
        **args,
    ):
        super(ASTGCN, self).__init__(**args)

        self.order = order
        self.nb_block = nb_block
        self.nb_chev_filter = nb_chev_filter
        self.nb_time_filter = nb_time_filter
        self.time_stride = time_stride

        self.BlockList = nn.ModuleList(
            [
                ASTGCN_block(
                    self.input_dim,
                    order,
                    nb_chev_filter,
                    nb_time_filter,
                    time_stride,
                    cheb_poly,
                    self.node_num,
                    self.his_len,
                )
            ]
        )
        self.BlockList.extend(
            [
                ASTGCN_block(
                    nb_time_filter,
                    order,
                    nb_chev_filter,
                    nb_time_filter,
                    1,
                    cheb_poly,
                    self.node_num,
                    self.his_len // time_stride,
                )
                for _ in range(nb_block - 1)
            ]
        )

        # Keep the original ASTGCN temporal length convention.
        # Original final_conv uses int(self.his_len / time_stride) as in_channels.
        self.encoder_time_len = int(self.his_len / time_stride)
        self.encoder_out_dim = nb_time_filter

        if self.encoder_time_len <= 0:
            raise ValueError(
                f"Invalid ASTGCN encoder_time_len={self.encoder_time_len}. "
                f"Please check his_len={self.his_len} and time_stride={time_stride}."
            )

        # Original value-space forecasting head.
        # Input:  (B, encoder_time_len, N, nb_time_filter)
        # Output: (B, pred_len, N, 1)
        self.final_conv = nn.Conv2d(
            self.encoder_time_len,
            self.pred_len,
            kernel_size=(1, nb_time_filter),
        )

        self.initialize()

        # JEPA hidden-space predictor.
        # Current supported setting: encode(input_seq) and encode(target_seq) share shape
        # (B, encoder_time_len, N, encoder_out_dim), because his_len == pred_len.
        self.jepa_time_predictor = nn.Linear(self.encoder_time_len, self.encoder_time_len)
        self.jepa_dim_predictor = nn.Linear(self.encoder_out_dim, self.encoder_out_dim)
        self._init_identity_linear(self.jepa_time_predictor)
        self._init_identity_linear(self.jepa_dim_predictor)

    def initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.uniform_(p)

    @staticmethod
    def _init_identity_linear(layer: nn.Linear):
        """Initialize a square Linear layer as identity when possible."""
        if layer.in_features == layer.out_features:
            nn.init.eye_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def _build_astgcn_input(
        self,
        seq: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build ASTGCN input tensor.

        Args:
            seq:      (B, T, N, C_seq), usually C_seq = 1.
            features: optional temporal/external features. Supported shapes:
                      - (B, T, F), repeated to every node;
                      - (B, T, N, F), used directly;
                      - (B, T, 1, F), expanded to every node.

        Returns:
            x: (B, N, input_dim, T), matching the original ASTGCN block input.
        """
        if seq.dim() != 4:
            raise ValueError(
                f"ASTGCN_JEPA expects seq with shape (B, T, N, C), but received {seq.shape}."
            )

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
                    f"ASTGCN_JEPA features must have shape (B, T, F) or (B, T, N, F), "
                    f"but received {features.shape}."
                )

            if features.shape[:3] != seq.shape[:3]:
                raise ValueError(
                    f"Feature shape {features.shape} is incompatible with seq shape {seq.shape}."
                )

            x = torch.concat([seq, features], dim=-1)

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"ASTGCN_JEPA expected input feature dim {self.input_dim}, "
                f"but received {x.shape[-1]}. Please provide matching features."
            )

        return rearrange(x, "b t n f -> b n f t")

    def encode(
        self,
        seq: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode input_seq or target_seq into ASTGCN hidden space.

        Shapes:
            seq:      (B, T, N, C_seq), usually C_seq = 1.
            features: optional (B, T, F) or (B, T, N, F).
            block input: (B, N, input_dim, T).
            block output before return: (B, N, encoder_out_dim, encoder_time_len).
            return Ex/Ey: (B, encoder_time_len, N, encoder_out_dim).

        ASTGCN attention parameters are tied to self.his_len, so T must equal self.his_len.
        """
        if seq.shape[1] != self.his_len:
            raise ValueError(
                "ASTGCN_JEPA.encode currently requires seq.shape[1] == self.his_len "
                f"because Temporal/Spatial Attention layers use fixed seq_len parameters. "
                f"Received seq length {seq.shape[1]}, but self.his_len={self.his_len}. "
                "For his_len != pred_len, use a separate target encoder or redesign the "
                "attention layers to support dynamic sequence lengths."
            )

        x = self._build_astgcn_input(seq, features)  # (B, N, F, T)

        for block in self.BlockList:
            x = block(x)  # (B, N, D, t)

        return rearrange(x, "b n d t -> b t n d")

    def predict(self, Ex: torch.Tensor) -> torch.Tensor:
        """
        Predict target hidden representation from input hidden representation.

        Shapes:
            Ex:      (B, encoder_time_len, N, encoder_out_dim)
            Ey_pred: (B, encoder_time_len, N, encoder_out_dim)
        """
        if Ex.dim() != 4:
            raise ValueError(f"ASTGCN_JEPA.predict expects 4D Ex, but received {Ex.shape}.")
        if Ex.shape[1] != self.encoder_time_len or Ex.shape[-1] != self.encoder_out_dim:
            raise ValueError(
                f"ASTGCN_JEPA.predict expected Ex shape (B, {self.encoder_time_len}, N, "
                f"{self.encoder_out_dim}), but received {Ex.shape}."
            )

        x = rearrange(Ex, "b t n d -> b n d t")
        x = self.jepa_time_predictor(x)
        x = rearrange(x, "b n d t -> b t n d")
        x = self.jepa_dim_predictor(x)
        return x

    def decode(self, Ey_pred: torch.Tensor) -> torch.Tensor:
        """
        Decode hidden representation to value-space forecasting output.

        Shapes:
            Ey_pred: (B, encoder_time_len, N, encoder_out_dim)
            y_pred:  (B, pred_len, N, output_dim), normally output_dim = 1.
        """
        if Ey_pred.dim() != 4:
            raise ValueError(
                f"ASTGCN_JEPA.decode expects 4D Ey_pred, but received {Ey_pred.shape}."
            )
        if Ey_pred.shape[1] != self.encoder_time_len or Ey_pred.shape[-1] != self.encoder_out_dim:
            raise ValueError(
                f"ASTGCN_JEPA.decode expected Ey_pred shape (B, {self.encoder_time_len}, N, "
                f"{self.encoder_out_dim}), but received {Ey_pred.shape}."
            )

        output = self.final_conv(Ey_pred)[:, :, :, -1]  # (B, pred_len, N)
        return output.unsqueeze(-1)  # (B, pred_len, N, 1)

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
            input_seq:  (B, T, N, C_seq), usually (B, his_len, N, 1).
            label/target_seq: (B, L, N, C_seq), currently must satisfy L == his_len.
            Ex:        (B, encoder_time_len, N, encoder_out_dim).
            Ey:        (B, encoder_time_len, N, encoder_out_dim).
            Ey_pred:   (B, encoder_time_len, N, encoder_out_dim).
            y_pred:    (B, pred_len, N, output_dim).
        """
        if mode == "pretrain":
            label_seq = label if label is not None else target_seq
            if label_seq is None:
                raise ValueError("Labels need to be provided during the pre-training process.")

            # Prefer target_features for target_seq/label encoding. If unavailable,
            # reuse input_features only when the temporal length matches.
            label_features = target_features
            if label_features is None and input_features is not None:
                if input_features.shape[1] == label_seq.shape[1]:
                    label_features = input_features
                else:
                    raise ValueError(
                        "target_features must be provided for ASTGCN_JEPA pretraining when "
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


class ASTGCN_block(nn.Module):
    def __init__(
        self,
        in_channels,
        order,
        nb_chev_filter,
        nb_time_filter,
        time_strides,
        cheb_polynomials,
        node_num,
        seq_len,
    ):
        super(ASTGCN_block, self).__init__()
        self.TAt = Temporal_Attention_layer(in_channels, node_num, seq_len)
        self.SAt = Spatial_Attention_layer(in_channels, node_num, seq_len)
        self.cheb_conv_SAt = cheb_conv_withSAt(
            order, cheb_polynomials, in_channels, nb_chev_filter
        )
        self.time_conv = nn.Conv2d(
            nb_chev_filter,
            nb_time_filter,
            kernel_size=(1, 3),
            stride=(1, time_strides),
            padding=(0, 1),
        )
        self.residual_conv = nn.Conv2d(
            in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides)
        )
        self.ln = nn.LayerNorm(nb_time_filter)

    def forward(self, x):
        bs, node_num, feature_num, seq_len = x.shape

        temporal_At = self.TAt(x)
        x_TAt = torch.matmul(x.reshape(bs, -1, seq_len), temporal_At).reshape(
            bs, node_num, feature_num, seq_len
        )

        spatial_At = self.SAt(x_TAt)
        spatial_gcn = self.cheb_conv_SAt(x, spatial_At)
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))

        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))
        x_residual = self.ln(
            F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)
        ).permute(0, 2, 3, 1)
        return x_residual


class Temporal_Attention_layer(nn.Module):
    def __init__(self, in_channels, node_num, seq_len):
        super(Temporal_Attention_layer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(node_num))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, node_num))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels))
        self.be = nn.Parameter(torch.FloatTensor(1, seq_len, seq_len))
        self.Ve = nn.Parameter(torch.FloatTensor(seq_len, seq_len))

    def forward(self, x):
        _, node_num, feature_num, seq_len = x.shape
        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        rhs = torch.matmul(self.U3, x)
        product = torch.matmul(lhs, rhs)

        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))
        E_normalized = F.softmax(E, dim=1)
        return E_normalized


class Spatial_Attention_layer(nn.Module):
    def __init__(self, in_channels, node_num, seq_len):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(seq_len))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, seq_len))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels))
        self.bs = nn.Parameter(torch.FloatTensor(1, node_num, node_num))
        self.Vs = nn.Parameter(torch.FloatTensor(node_num, node_num))

    def forward(self, x):
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)
        product = torch.matmul(lhs, rhs)

        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))
        S_normalized = F.softmax(S, dim=1)
        return S_normalized


class cheb_conv_withSAt(nn.Module):
    def __init__(self, order, cheb_polynomials, in_channels, out_channels):
        super(cheb_conv_withSAt, self).__init__()
        self.order = order
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.ParameterList(
            [
                nn.Parameter(torch.FloatTensor(in_channels, out_channels))
                for _ in range(order)
            ]
        )

    def forward(self, x, spatial_attention):
        bs, node_num, in_channels, seq_len = x.shape
        outputs = []
        for time_step in range(seq_len):
            graph_signal = x[:, :, :, time_step]

            output = torch.zeros(bs, node_num, self.out_channels).to(x.device)
            for k in range(self.order):
                T_k = self.cheb_polynomials[k]
                T_k_with_at = T_k.mul(spatial_attention)
                theta_k = self.Theta[k]
                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)
                output = output + rhs.matmul(theta_k)
            outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1))
