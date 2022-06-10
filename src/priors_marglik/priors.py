import torch
import torch.nn as nn
import numpy as np
import itertools
from torch import linalg
try:
    from torch.linalg import cholesky
except:
    from torch import cholesky
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal

class RadialBasisFuncCov(nn.Module):

    def __init__(
        self,
        kernel_size,
        lengthscale_init,
        variance_init,
        dist_func,
        store_device
        ):

        super(RadialBasisFuncCov, self).__init__()

        self.kernel_size = kernel_size
        self.store_device = store_device
        self.dist_mat = self.compute_dist_matrix(dist_func)
        self.log_lengthscale = nn.Parameter(torch.ones(1, device=self.store_device))
        self.log_variance = nn.Parameter(torch.ones(1, device=self.store_device))
        self._init_parameters(lengthscale_init, variance_init)

    def _init_parameters(self, lengthscale_init, variance_init):
        nn.init.constant_(self.log_lengthscale,
                          np.log(lengthscale_init))
        nn.init.constant_(self.log_variance, np.log(variance_init))
    
    def unscaled_cov_mat(self, eps=1e-6):
        
        lengthscale = torch.exp(self.log_lengthscale)
        assert not torch.isnan(lengthscale) 
        cov_mat = torch.exp(-self.dist_mat / lengthscale) + eps * torch.eye(*self.dist_mat.shape, device=self.store_device)
        return cov_mat

    def cov_mat(self, return_cholesky=True, eps=1e-6):
        
        variance = torch.exp(self.log_variance)
        assert not torch.isnan(variance)
        cov_mat = self.unscaled_cov_mat(eps=eps)
        cov_mat = variance * cov_mat
        return (cholesky(cov_mat) if return_cholesky else cov_mat)

    def compute_dist_matrix(self, dist_func):
        coords = [torch.as_tensor([i, j], dtype=torch.float32) for i in
                  range(self.kernel_size) for j in
                  range(self.kernel_size)]
        combs = [[el_1, el_2] for el_1 in coords for el_2 in coords]
        dist_mat = torch.as_tensor([dist_func(el1 - el2) for (el1,
                                   el2) in combs], dtype=torch.float32, device=self.store_device)
        return dist_mat.view(self.kernel_size ** 2, self.kernel_size
                             ** 2)
    def log_det(self):
        return 2 * self.cov_mat(return_cholesky=True).diag().log().sum()
    
    def log_lengthscale_cov_mat_grad(self):
        # we multiply by the lengthscale value (chain rule)
        return self.dist_mat * self.cov_mat(return_cholesky=False) / torch.exp(self.log_lengthscale) # we do this by removing the 2
    
    def log_varainces_cov_mat_grad(self):
        # we multiply by the variance value (chain rule)
        return self.cov_mat(return_cholesky=False, eps=1e-6)
         

class GPprior(nn.Module):

    def __init__(self, covariance_function, store_device):
        super(GPprior, self).__init__()
        self.cov = covariance_function
        self.store_device = store_device

    def sample(self, shape):
        cov = self.cov.cov_mat(return_cholesky=True)
        mean = torch.zeros(self.cov.kernel_size ** 2).to(self.store_device)
        m = MultivariateNormal(loc=mean, scale_tril=cov)
        params_shape = shape + [self.cov.kernel_size,
                                self.cov.kernel_size]
        return m.rsample(sample_shape=shape).view(params_shape)

    def log_prob(self, x):
        cov = self.cov.cov_mat(return_cholesky=True)
        mean = torch.zeros(self.cov.kernel_size ** 2).to(self.store_device)
        m = MultivariateNormal(loc=mean, scale_tril=cov)
        return m.log_prob(x)

class NormalPrior(nn.Module):

    def __init__(
        self,
        kernel_size,
        variance_init,
        store_device
        ):

        super().__init__()
        self.store_device = store_device
        self.kernel_size = kernel_size
        self.log_variance = nn.Parameter(torch.ones(1, device=self.store_device))
        self._init_parameters(variance_init)

    def _init_parameters(self, variance_init):
        nn.init.constant_(self.log_variance, np.log(variance_init))

    def sample(self, shape):
        mean = torch.zeros(self.kernel_size, device=self.store_device)
        m = Normal(loc=mean, scale=torch.exp(self.log_variance)**.5)
        return m.rsample(sample_shape=shape)

    def log_prob(self, x):
        mean = torch.zeros(self.kernel_size, device=self.store_device)
        m = Normal(loc=mean, scale=torch.exp(self.log_variance)**.5)
        return m.log_prob(x)

    def cov_mat(self, return_cholesky=True):
        eye = torch.eye(self.kernel_size).to(self.store_device)
        fct = torch.exp(0.5 * self.log_variance) if return_cholesky else torch.exp(self.log_variance)
        cov_mat = fct * eye
        return cov_mat

    def cov_log_det(self):
        return self.log_variance * self.kernel_size
