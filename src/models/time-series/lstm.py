import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from src.base.model import BaseModel


class LSTM(BaseModel):
    def __init__(self, init_dim, hid_dim, end_dim, layer, dropout, **args):
        super(LSTM, self).__init__(**args)
        self.start_conv = nn.Conv2d(
            in_channels=self.input_dim, out_channels=init_dim, kernel_size=(1, 1)
        )

        self.lstm = nn.LSTM(
            input_size=init_dim,
            hidden_size=hid_dim,
            num_layers=layer,
            batch_first=True,
            dropout=dropout,
        )

        self.end_linear1 = nn.Linear(hid_dim, end_dim)
        self.end_linear2 = nn.Linear(end_dim, self.pred_len * self.output_dim)

    def forward(self, input_seq, input_features, *args, **kwargs):
        # (b, t, n, f)
        x = torch.concat(
            [
                input_seq,
                repeat(input_features, "b t f -> b t n f", n=input_seq.shape[2]),
            ],
            dim=-1,
        )
        x = x.transpose(1, 3)
        b, f, n, t = x.shape

        x = x.transpose(1, 2).reshape(b * n, f, 1, t)
        x = self.start_conv(x).squeeze().transpose(1, 2)

        out, _ = self.lstm(x)
        x = out[:, -1, :]

        x = F.relu(self.end_linear1(x))
        x = self.end_linear2(x)
        x = x.reshape(b, n, self.pred_len, self.output_dim).transpose(1, 2)
        return x
