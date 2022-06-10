import torch
import torch.nn as nn
import numpy as np
from .bayes_blocks import get_GPprior, BayesianiseBlock, BayesianiseBlockUp
from .bayes_layer import Conv2dGPprior
from .priors import NormalPrior
from collections.abc import Iterable
from copy import deepcopy
from itertools import chain
from deep_image_prior import tv_loss

def _cov_mat_mul(x, cov_mat):
    N = x.shape[0]
    x = x.view(-1, cov_mat.shape[0], cov_mat.shape[-1])
    x = x.permute(1, 0, 2)
    out = x @ cov_mat
    out = out.permute(0, 2, 1).reshape([cov_mat.shape[0]
            * cov_mat.shape[-1], N]).t()
    return out

class BlocksGPpriors(nn.Module):

    def __init__(self, model, bayesianize_model, store_device, lengthscale_init, variance_init, lin_weights=None):
        super(BlocksGPpriors, self).__init__()

        self.model = model
        self.bayesianize_model = bayesianize_model
        self.store_device = store_device
        self.lengthscale_init = lengthscale_init
        self.variance_init = variance_init
        self.gp_priors = bayesianize_model.gp_priors
        self.normal_priors = bayesianize_model.normal_priors
        self.lin_weights = lin_weights

    @property
    def gp_log_lengthscales(self):
        return [gp_prior.cov.log_lengthscale for gp_prior in self.gp_priors
                if gp_prior.cov.log_lengthscale.requires_grad]

    @property
    def gp_log_variances(self):
        return [gp_prior.cov.log_variance for gp_prior in self.gp_priors
                if gp_prior.cov.log_variance.requires_grad]

    @property
    def normal_log_variances(self):
        return [normal_prior.log_variance for normal_prior in self.normal_priors
                if normal_prior.log_variance.requires_grad]

    def _get_repeat(self, modules):

        repeat = 0
        for module in modules:
            repeat += module.in_channels * module.out_channels
        return repeat

    def get_idx_parameters_per_block(self):

        n_weights_all = 0
        list_idx = []
        for _, modules_under_prior in zip(
                self.priors, self.bayesianize_model.ref_modules_under_priors):
            n_weights_per_block = 0
            for layer in modules_under_prior:
                assert isinstance(layer, torch.nn.Conv2d)
                params = layer.weight.view(-1, *layer.kernel_size).view(-1, layer.kernel_size[0]**2)
                n_weights_per_block += params.numel()
            list_idx.append((n_weights_all, n_weights_all + n_weights_per_block))
            n_weights_all += n_weights_per_block
        return list_idx

    def get_idx_parameters_per_layer(self):

        n_weights_all = 0
        list_idx_per_layer = []
        for _, modules_under_prior in zip(
                self.priors, self.bayesianize_model.ref_modules_under_priors):
            n_weights_per_block = 0
            for layer in modules_under_prior:
                assert isinstance(layer, torch.nn.Conv2d)
                params = layer.weight.view(-1, *layer.kernel_size).view(-1, layer.kernel_size[0]**2)
                n_weights_per_layer = params.numel()
                list_idx_per_layer.append((n_weights_all, n_weights_all + n_weights_per_layer))
                n_weights_all += n_weights_per_layer
        return list_idx_per_layer

    def get_net_log_det_cov_mat(self):

        log_det = torch.zeros(1, device=self.store_device)
        for gp_prior, modules_under_gp_prior in zip(
                self.gp_priors, self.bayesianize_model.ref_modules_under_gp_priors):
            log_det = log_det + gp_prior.cov.log_det() * self._get_repeat(modules_under_gp_prior)
        for normal_prior, modules_under_normal_prior in zip(
                self.normal_priors, self.bayesianize_model.ref_modules_under_normal_priors):
            log_det = log_det + normal_prior.cov_log_det() * self._get_repeat(modules_under_normal_prior)
        return log_det

    def get_net_prior_log_prob(self):

        log_prob = torch.zeros(1, device=self.store_device)
        for gp_prior, modules_under_gp_prior in zip(
                self.gp_priors, self.bayesianize_model.ref_modules_under_gp_priors):
            for layer in modules_under_gp_prior:
                params = layer.weight.view(-1, *layer.kernel_size).view(-1, layer.kernel_size[0]**2)
                log_prob += gp_prior.log_prob(params).sum(dim=0)
        for normal_prior, modules_under_normal_prior in zip(
                self.normal_priors, self.bayesianize_model.ref_modules_under_normal_priors):
            for layer in modules_under_normal_prior:
                params = layer.weight.view(-1, *layer.kernel_size).view(-1, layer.kernel_size[0]**2)
                log_prob += normal_prior.log_prob(params).sum(dim=0)
        return log_prob

    def get_net_prior_log_prob_linearized_weights(self):
        log_prob = torch.zeros(1, device=self.store_device)
        list_idx_per_layer = self.get_idx_parameters_per_layer()
        all_modules_under_prior = self.bayesianize_model.get_all_modules_under_prior()
        for gp_prior, modules_under_gp_prior in zip(
                self.gp_priors, self.bayesianize_model.ref_modules_under_gp_priors):
            for layer in modules_under_gp_prior:
                layer_idx = all_modules_under_prior.index(layer)
                params = self.lin_weights[slice(*list_idx_per_layer[layer_idx])]
                params = params.view(-1, layer.kernel_size[0]**2)
                log_prob += gp_prior.log_prob(params).sum(dim=0)
        for normal_prior, modules_under_normal_prior in zip(
                self.normal_priors, self.bayesianize_model.ref_modules_under_normal_priors):
            for layer in modules_under_normal_prior:
                layer_idx = all_modules_under_prior.index(layer)
                params = self.lin_weights[slice(*list_idx_per_layer[layer_idx])]
                params = params.view(-1, layer.kernel_size[0]**2)
                log_prob += normal_prior.log_prob(params).sum(dim=0)
        return log_prob

    @property
    def priors(self):
        return chain(self.gp_priors, self.normal_priors)

    def _gp_prior_cov_mats(self, gp_prior, modules_under_gp_prior):
        cov_mat = gp_prior.cov.cov_mat(return_cholesky=False)
        repeat_fct = self._get_repeat(modules_under_gp_prior)
        return [cov_mat] * repeat_fct

    def _normal_prior_cov_mats(self, normal_prior, modules_under_normal_prior):
        cov_mat = normal_prior.cov_mat(return_cholesky=False)
        repeat_fct = self._get_repeat(modules_under_normal_prior)
        return [cov_mat] * repeat_fct

    def get_gp_prior_cov_mat(self, gp_idx=None):

        gp_priors = self.gp_priors
        ref_modules_under_gp_priors = self.bayesianize_model.ref_modules_under_gp_priors
        if gp_idx is not None:
            gp_priors = gp_priors[gp_idx]
            ref_modules_under_gp_priors = ref_modules_under_gp_priors[gp_idx]
            if not isinstance(gp_priors, Iterable):
                gp_priors = [gp_priors]
                ref_modules_under_gp_priors = [ref_modules_under_gp_priors]

        gp_cov_mat_list = []
        for gp_prior, modules_under_gp_prior in zip(
                gp_priors, ref_modules_under_gp_priors):
            gp_cov_mat_list += self._gp_prior_cov_mats(gp_prior, modules_under_gp_prior)
        gp_cov_mat = torch.stack(gp_cov_mat_list) if gp_cov_mat_list else torch.empty(0, 1)
        return gp_cov_mat

    # normal_idx is relative, i.e. an index in the self.normal_priors list
    def get_normal_prior_cov_mat(self, normal_idx=None):

        normal_priors = self.normal_priors
        ref_modules_under_normal_priors = self.bayesianize_model.ref_modules_under_normal_priors
        if normal_idx is not None:
            normal_priors = normal_priors[normal_idx]
            ref_modules_under_normal_priors = ref_modules_under_normal_priors[normal_idx]
            if not isinstance(normal_priors, Iterable):
                normal_priors = [normal_priors]
                ref_modules_under_normal_priors = [ref_modules_under_normal_priors]

        normal_cov_mat_list = []
        for normal_prior, modules_under_normal_prior in zip(
                normal_priors, ref_modules_under_normal_priors):
            normal_cov_mat_list += self._normal_prior_cov_mats(normal_prior, modules_under_normal_prior)
        normal_cov_mat = torch.stack(normal_cov_mat_list) if normal_cov_mat_list else torch.empty(0, 1)
        return normal_cov_mat

    def get_single_prior_cov_mat(self, idx):
        if idx < len(self.gp_priors):
            gp_idx = idx
            cov_mat = self.get_gp_prior_cov_mat(gp_idx=gp_idx)
        else:
            normal_idx = idx - len(self.gp_priors)
            cov_mat = self.get_normal_prior_cov_mat(normal_idx=normal_idx)
        return cov_mat

    def matrix_prior_cov_mul(self, x, idx=None):

        if idx is None:

            gp_cov_mat = self.get_gp_prior_cov_mat().to(x.device)
            normal_cov_mat = self.get_normal_prior_cov_mat().to(x.device)

            gp_x = x[:, :(gp_cov_mat.shape[0] * gp_cov_mat.shape[-1])]
            normal_x = x[:, (gp_cov_mat.shape[0] * gp_cov_mat.shape[-1]):]

            gp_out = _cov_mat_mul(gp_x, gp_cov_mat) if gp_x.shape[1] != 0 else torch.empty(x.shape[0], 0).to(x.device)
            normal_out = _cov_mat_mul(normal_x, normal_cov_mat) if normal_x.shape[1] != 0 else torch.empty(x.shape[0], 0).to(x.device)

            out = torch.cat([gp_out, normal_out], dim=-1)

        elif np.isscalar(idx):

            cov_mat = self.get_single_prior_cov_mat(idx=idx)
            out = _cov_mat_mul(x, cov_mat)

        else:
            raise NotImplementedError

        return out
