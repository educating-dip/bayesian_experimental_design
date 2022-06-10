import torch
import torch
import torch.nn as nn

class Conv2dGPprior(nn.Module):

    def __init__(
        self,
        weights_sampler,
        in_channels,
        out_channels,
        kernel_size,
        bias,
        stride,
        padding,
        ):
        super(Conv2dGPprior, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size[0]
        self.bias = bias
        self.stride = stride[0]
        self.padding = padding[0]
        self.weights_sampler = weights_sampler

    def forward(self, x):
        N = self.in_channels * self.out_channels
        weights = \
            self.weights_sampler.sample(shape=[N]).reshape(self.out_channels,
                self.in_channels, self.kernel_size, self.kernel_size)
        return torch.nn.functional.conv2d(x, weights, self.bias,
                stride=self.stride, padding=self.padding)
