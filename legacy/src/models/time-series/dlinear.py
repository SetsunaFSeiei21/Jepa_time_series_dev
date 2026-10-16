import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.Autoformer_EncDec import series_decomp

from src.base.model import BaseModel


class DLinear(BaseModel):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, individual, moving_avg, **args):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(DLinear, self).__init__(**args)

        # Series decomposition block from Autoformer
        self.decompsition = series_decomp(moving_avg)
        self.individual = individual
        self.channels = self.node_num

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.his_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.his_len, self.pred_len))

                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.his_len) * torch.ones([self.pred_len, self.his_len])
                )
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.his_len) * torch.ones([self.pred_len, self.his_len])
                )
        else:
            self.Linear_Seasonal = nn.Linear(self.his_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.his_len, self.pred_len)

            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.his_len) * torch.ones([self.pred_len, self.his_len])
            )
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.his_len) * torch.ones([self.pred_len, self.his_len])
            )

    def forward(self, input_seq, *args, **kwargs):
        input_seq = input_seq.squeeze(-1)
        seasonal_init, trend_init = self.decompsition(input_seq)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(
            0, 2, 1
        )
        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                dtype=seasonal_init.dtype,
            ).to(seasonal_init.device)
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.pred_len],
                dtype=trend_init.dtype,
            ).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :]
                )
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        input_seq = seasonal_output + trend_output
        return input_seq.permute(0, 2, 1).unsqueeze(-1)
