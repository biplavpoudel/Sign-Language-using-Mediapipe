import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        # we reshape input tensor x such that batch size is preserved
        # while remaining dimensions are flattened to single dimensions
        return x.view(x.size(0), -1)
        # -1 is a placeholder used by PyTorch to infer the dimension size based on the total number of elements
