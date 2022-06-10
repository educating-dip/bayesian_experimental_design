import torch
import numpy as np
from skimage.metrics import structural_similarity
from torch.nn.modules.dropout import _DropoutNd
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

def diag_gaussian_log_prob(observation, proj_recon, sigma):

    assert len(sigma) == 1
    assert observation.shape == proj_recon.shape

    dist = torch.distributions.Normal(loc=proj_recon.flatten(), scale=sigma)
    return dist.log_prob(observation.flatten()).sum()

def tv_loss(x):
    """
    Isotropic TV loss.
    """
    dh = torch.abs(x[..., :, 1:] - x[..., :, :-1])
    dw = torch.abs(x[..., 1:, :] - x[..., :-1, :])
    return torch.sum(dh) + torch.sum(dw)  # note that this differs from Baguer et al., who used torch.sum(dh[..., :-1, :] + dw[..., :, :-1])

def PSNR(reconstruction, ground_truth, data_range=None):
    gt = np.asarray(ground_truth)
    mse = np.mean((np.asarray(reconstruction) - gt)**2)
    if mse == 0.:
        return float('inf')
    if data_range is not None:
        return 20*np.log10(data_range) - 10*np.log10(mse)
    else:
        data_range = np.max(gt) - np.min(gt)
        return 20*np.log10(data_range) - 10*np.log10(mse)

def SSIM(reconstruction, ground_truth, data_range=None):
    gt = np.asarray(ground_truth)
    if data_range is not None:
        return structural_similarity(reconstruction, gt, data_range=data_range)
    else:
        data_range = np.max(gt) - np.min(gt)
        return structural_similarity(reconstruction, gt, data_range=data_range)

def normalize(x, inplace=False):
    if inplace:
        x -= x.min()
        x /= x.max()
    else:
        x = x - x.min()
        x = x / x.max()
    return x

class mc_dropout2d(_DropoutNd):
    def forward(self, input):
        return F.dropout2d(input, self.p, True, self.inplace)

class conv2d_dropout(nn.Module):
    def __init__(self, sub_module, p):
        super().__init__()
        self.layer = sub_module
        self.dropout = mc_dropout2d(p=p)
    def forward(self, x): 
        x = self.layer(x)
        return self.dropout(x)

def bayesianize_architecture(model, p=0.05):
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Sequential):
            for name_sub_module, sub_module in module.named_children(): 
                if isinstance(sub_module, torch.nn.Conv2d):
                    if sub_module.kernel_size == (3, 3):
                        setattr(module, name_sub_module, conv2d_dropout(sub_module, p))

def sample_from_bayesianized_model(model, filtbackproj, mc_samples, device=None):
    sampled_recons = []
    if device is None: 
        device = filtbackproj.device
    for _  in tqdm(range(mc_samples), desc='sampling'):
        sampled_recons.append(model.forward(filtbackproj)[0].detach().to(device))
    return torch.cat(sampled_recons, dim=0)