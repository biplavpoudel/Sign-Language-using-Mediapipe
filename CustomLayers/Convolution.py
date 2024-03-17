import torch
import torch.nn as nn
import torch.nn.functional as F


class Convolution(nn.Module):
    def __init__(self, filters, kernel_size, input_shape, stride=1):
        super(Convolution, self).__init__()
        input_depth, input_height, input_width = input_shape
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.output_shape = (filters, input_height - kernel_size + 1, input_width - kernel_size + 1)
        # for valid convolution, output size is: (image_dimension - kernel_dimension +1)
        # for full convolution, output size is: (image_dimension + kernel_dimension -1)
        self.kernel_shape = (filters, input_depth, kernel_size, kernel_size)
        self.kernels = nn.Parameter(torch.randn(*self.kernel_shape))
        # This is equivalent to: self.kernels = np.random.randn((64,3,3,3))
        # self.biases = nn.Parameter(torch.randn(*self.output_shape))
        self.biases = nn.Parameter(torch.randn(filters))

    def forward(self, input):
        input = input.to("cuda")
        output = F.conv2d(input, self.kernels, bias=self.biases, stride=self.stride, padding=0)
        torch.cuda.empty_cache()
        return output
