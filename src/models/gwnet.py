import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from torch.nn import Linear, ReLU, Sequential

from src.base.model import BaseModel


class GWNET(BaseModel):
    """
    Reference code: https://github.com/nnzhan/Graph-WaveNet
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
        **args
    ):
        super(GWNET, self).__init__(**args)
        self.supports = supports
        self.supports_len = len(supports)
        self.adp_adj = adp_adj
        # print("check supports length", len(supports), self.supports_len)

        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.kernel_size = kernel_size

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

        for b in range(self.blocks):
            additional_scope = self.kernel_size - 1
            new_dilation = 1
            for i in range(self.layers):
                # dilated convolutions
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

                # 1x1 convolution for residual connection
                self.residual_convs.append(
                    nn.Conv1d(
                        in_channels=dilation_channels,
                        out_channels=residual_channels,
                        kernel_size=(1, 1),
                    )
                )

                # 1x1 convolution for skip connection
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

    def forward(self, input_seq, input_features, *args, **kwargs):
        # (b, t, n, f)
        x = torch.concat(
            [
                input_seq,
                repeat(input_features, "b t f -> b t n f", n=input_seq.shape[2]),
            ],
            dim=-1,
        )
        inputs = x.permute(0, 3, 2, 1)

        if self.his_len < self.receptive_field:
            x = nn.functional.pad(
                inputs, (self.receptive_field - self.his_len, 0, 0, 0)
            )
        else:
            x = inputs

        x = self.start_conv(x)
        skip = 0

        # print(f'x after start_conv:{x.shape}')

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.adp_adj:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports.copy()
            new_supports.append(adp)
        else:
            new_supports = self.supports.copy()

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            # (dilation, init_dilation) = self.dilations[i]

            # residual = dilation_func(x, dilation, init_dilation, i)

            residual = x.detach()

            # print(f'x_residual:{x.shape}')

            # dilated convolution
            filter = self.filter_convs[i](residual)
            # print(f'filter:{filter.shape}')

            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            # print(gate.shape)
            x = filter * gate

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3) :]
            except:
                skip = 0
            skip = s + skip

            # print(f'skip_x:{x.shape}') # [64, 32, 1085, 12]

            x = self.gconv[i](x, new_supports)

            # print(f'GCN_x:{x.shape}')

            x = x + residual[:, :, :, -x.size(3) :]

            x = self.bn[i](x)

        # (b, 256, 1085, 12) --> (b, 256, 1085)
        if self.mlp_input_dim > 0:
            skip = self.mlp_projection(skip)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))

        x = self.end_conv_2(x)

        return x


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
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h
