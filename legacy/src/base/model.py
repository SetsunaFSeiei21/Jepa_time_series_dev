import torch.nn as nn
from abc import abstractmethod


class BaseModel(nn.Module):
    def __init__(self, node_num, input_dim, output_dim, his_len, pred_len, **args):
        super(BaseModel, self).__init__()
        self.node_num = node_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.his_len = his_len
        self.pred_len = pred_len

    @abstractmethod
    def forward(self):
        raise NotImplementedError

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])
