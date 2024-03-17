import torch
import torch.nn as nn


class Dropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(Dropout, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        output = self.dropout(input, self.p, training=self.training, inplace=self.inplace)
        torch.cuda.empty_cache()
        return output

    def dropout(self, input, prob=0.5, training=True, inplace=False):
        if training:
            if inplace:
                # dropout in-place i.e. tensor is modified directly
                mask = (torch.rand(input.shape, device=input.device) >= prob).to(input.device)
                # mask is a boolean mask
                input *= mask / (1-prob)
                # multiplying by 1 / (1 - prob) "scales" the remaining elements by
                # the reciprocal of the probability of keeping an element,
                # thus, compensating for the dropped elements.
                # e.g.
                # Input tensor: tensor([1., 2., 3., 4., 5.])
                # Dropout mask: tensor([0., 0., 1., 1., 0.]) which is the result of comparison of random values
                # between 0 and 1 (generated with the same shape as input tensor) and probability
                # Scaled output: tensor([0., 0., 6., 8., 0.])
                # here elements 3 and 4 are scaled by 2 as to compensate for dropping elements 1,2,5
                return input
            else:
                # dropout out-of-place i.e. new tensor is created
                mask = (torch.rand(input.shape, device=input.device) >= prob).float().to(input.device)
                return input * mask/(1-prob)
        else:
            return input  # no dropout

