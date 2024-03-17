import torch
import torch.nn as nn
from CustomLayers.Activation import LogSoftmax


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.log_softmax = LogSoftmax()
        self.negative_log_likelihood = NegativeLogLikelihood()

    def forward(self, output, target):
        log_p = self.log_softmax(output)
        loss = self.negative_log_likelihood(log_p, target)
        return loss


class NegativeLogLikelihood(nn.Module):
    def __init__(self):
        super(NegativeLogLikelihood, self).__init__()

    def forward(self, input_tensor, target_tensor):
        return -input_tensor[range(input_tensor.shape[0]), target_tensor].mean()
