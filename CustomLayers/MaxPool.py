# MaxPool2d as pool dimension is two-dimensional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxPool2d(nn.Module):
    def __init__(self, pool_size, stride=None, padding=0):
        super(MaxPool2d, self).__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding
        # padding by 1 increases output_height/width by 2

    def forward(self, input):
        # return self.max_pool2d(input, pool_size=self.pool_size, stride=self.stride, padding=self.padding)
        # out of CUDA memory issues due to nested loops
        return F.max_pool2d(input, kernel_size=self.pool_size, stride=self.stride, padding=self.padding)

    def max_pool2d(self, input, pool_size, stride, padding):
        batch_size, channels, height, width = input.size()

        output_height = (height + 2 * padding - pool_size) // stride + 1
        output_width = (width + 2 * padding - pool_size) // stride + 1
        output = input.new(batch_size, channels, output_height, output_width)

        for b in range(batch_size):
            for c in range(channels):
                for h in range(output_height):
                    for w in range(output_width):
                        patch = input[b, c, h*stride: h*stride + pool_size, w*stride: w*stride + pool_size]
                        # remember slicing: is exclusive and so is range (0,n-1)
                        output[b, c, h, w] = torch.max(patch)
                        # max value of a pool (i.e.2x2) is the output
        return output




