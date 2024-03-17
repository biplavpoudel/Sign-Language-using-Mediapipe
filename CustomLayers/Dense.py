import math
import torch
import torch.nn as nn


class Dense(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.init_parameters()

    def init_parameters(self):
        # initializing weight from kaiming-distro as we use relu (xavier is more generalized)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # initializing bias
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            # fan_in = self.in_features will be valid I guess
            # bounds of uniform distro
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        output = self.linear(input, self.weight, self.bias)
        torch.cuda.empty_cache()
        return output

    def linear(self, input, weight, bias):
        input = input.to(weight.device)
        output = input.matmul(self.weight.t())
        if bias is not None:
            output += bias
        return output
