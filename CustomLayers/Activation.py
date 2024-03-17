import torch
import torch.nn as nn


class LogSoftmax(nn.Module):
    def __init__(self, dim=1):
        super(LogSoftmax, self).__init__()
        self.dim = dim

    def forward(self, input_tensor):
        return input_tensor - torch.logsumexp(input_tensor, dim=self.dim, keepdim=True)


class RectifiedLinearUnit(nn.Module):
    def __init__(self):
        super(RectifiedLinearUnit, self).__init__()

    def forward(self, input_tensor):
        return input_tensor * (input_tensor > 0).float()


