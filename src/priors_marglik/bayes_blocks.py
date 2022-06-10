#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import torch.linalg as linalg
from copy import deepcopy
from .bayes_layer import Conv2dGPprior
from .priors import GPprior, RadialBasisFuncCov

def get_GPprior(store_device, lengthscale_init, variance_init=1):
    dist_func = lambda x: linalg.norm(x, ord=2)  # setting GP cov dist func
    cov_kwards = {
        'kernel_size': 3,
        'lengthscale_init': lengthscale_init,
        'variance_init': variance_init, # self._get_average_var_group_filter(),
        'dist_func': dist_func,
        'store_device': store_device,
        }
    cov_func = \
        RadialBasisFuncCov(**cov_kwards).to(store_device)
    GPp = GPprior(cov_func, store_device)
    return GPp

class BayesianiseBlock(nn.Module):

    def __init__(
        self,
        block,
        store_device,
        lengthscale_init,
        variance_init=1,
        ):
        super(BayesianiseBlock, self).__init__()

        self.block = deepcopy(block)
        self.store_device = store_device
        self.norm_layers = self._list_norm_layers()
        self.GPp = get_GPprior(
                store_device=self.store_device,
                lengthscale_init=lengthscale_init,
                variance_init=variance_init)

        for (i, layer) in enumerate(self.block.conv):
            if isinstance(layer, torch.nn.Conv2d):
                conv_kwards = {
                    'in_channels': layer.in_channels,
                    'out_channels': layer.out_channels,
                    'kernel_size': layer.kernel_size,
                    'bias': layer.bias,
                    'stride': layer.stride,
                    'padding': layer.padding,
                    }
                self.block.conv[i] = Conv2dGPprior(self.GPp,
                        **conv_kwards).to(self.store_device)

    def forward(self, x):

        return self.block(x)

    def _list_norm_layers(self):
        """ compute list of names of all GroupNorm (or BatchNorm2d) layers in the model """

        norm_layers = []
        for (name, module) in self.block.named_modules():
            name = name.replace('module.', '')
            if isinstance(module, torch.nn.GroupNorm) \
                or isinstance(module, torch.nn.BatchNorm2d):
                norm_layers.append(name + '.weight')
                norm_layers.append(name + '.bias')
        return norm_layers

    def _get_average_var_group_filter(self):

        var = []
        for (name, param) in self.block.named_parameters():
            name = name.replace('module.', '')
            if 'weight' in name and name not in self.norm_layers \
                and 'skip_conv' not in name:
                param_ = param.view(-1,
                                    *param.shape[2:]).flatten(start_dim=1)
                var.append(param_.var(dim=1).mean(dim=0).item())
        return np.mean(var)


class BayesianiseBlockUp(BayesianiseBlock):

    def __init__(
        self,
        block,
        store_device,
        lengthscale_init,
        variance_init=1,
        ):
        super(BayesianiseBlockUp, self).__init__(block, store_device,
                lengthscale_init, variance_init=variance_init)

    def forward(self, x1, x2):
        x1 = self.block.up(x1)
        x2 = self.block.skip_conv(x2)
        if not self.block.skip:
            x2 = x2 * 0
        x = self.block.concat(x1, x2)
        x = self.block.conv(x)
        return x
