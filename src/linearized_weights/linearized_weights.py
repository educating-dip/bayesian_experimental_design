import os
import socket
import datetime
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from deep_image_prior import tv_loss, PSNR
from linearized_laplace import agregate_flatten_weight_grad
from scalable_linearised_laplace import get_weight_block_vec, get_fwAD_model, fwAD_JvP, finite_diff_JvP

def log_homoGauss_grad(mean, y, ray_trafo_module_adj, prec=1):
    return - (prec * ray_trafo_module_adj(y - mean))

def tv_loss_grad(x):
    # in tv_loss:
    # dh = torch.abs(x[..., :, 1:] - x[..., :, :-1])
    # dw = torch.abs(x[..., 1:, :] - x[..., :-1, :])

    assert x.shape[-1] == x.shape[-2]

    # # for loss ``torch.sum(dh[..., :-1, :] + dw[..., :, :-1])`` (old behaviour of tv_loss)
    # pad_x = torch.zeros((1, 1, x.shape[-2], 1), device = x.device)
    # pad_x_short = pad_x[:, :, :-1, :]
    # pad_y = torch.zeros((1, 1, 1, x.shape[-1]), device = x.device)
    # pad_y_short = pad_y[:, :, :, :-1]
    # sign_diff_x = torch.sign(torch.diff(-x[:, :, :-1, :], n=1, dim=-1))
    # diff_x_pad = torch.cat([pad_x, torch.cat([sign_diff_x, pad_y_short], dim=-2), pad_x], dim=-1)
    # grad_tv_x = torch.diff(diff_x_pad, n=1, dim=-1)
    # sign_diff_y = torch.sign(torch.diff(-x[:, :, :, :-1], n=1, dim=-2))
    # diff_y_pad = torch.cat([pad_y, torch.cat([sign_diff_y, pad_x_short], dim=-1), pad_y], dim=-2)
    # grad_tv_y = torch.diff(diff_y_pad, n=1, dim=-2)

    # for loss ``torch.sum(dh) + torch.sum(dw)``
    # (summing over all elements in dh and dw)
    sign_diff_x = torch.sign(torch.diff(-x, n=1, dim=-1))
    pad = torch.zeros((1, 1, x.shape[-2], 1), device = x.device)
    diff_x_pad = torch.cat([pad, sign_diff_x, pad], dim=-1)
    grad_tv_x = torch.diff(diff_x_pad, n=1, dim=-1)
    sign_diff_y = torch.sign(torch.diff(-x, n=1, dim=-2))
    pad = torch.zeros((1, 1, 1, x.shape[-1]), device = x.device)
    diff_y_pad = torch.cat([pad, sign_diff_y, pad], dim=-2)
    grad_tv_y = torch.diff(diff_y_pad, n=1, dim=-2)
    
    return grad_tv_x + grad_tv_y

def list_norm_layers(model):

    """ compute list of names of all GroupNorm (or BatchNorm2d) layers in the model """
    norm_layers = []
    for (name, module) in model.named_modules():
        name = name.replace('module.', '')
        if isinstance(module, torch.nn.GroupNorm) or isinstance(module,
                torch.nn.BatchNorm2d) or isinstance(module, torch.nn.InstanceNorm2d):
            norm_layers.append(name + '.weight')
            norm_layers.append(name + '.bias')
    return norm_layers

def weights_linearization(cfg, bayesianised_model, filtbackproj, observation, ground_truth, reconstructor, ray_trafos):

    filtbackproj = filtbackproj.to(reconstructor.device)
    observation = observation.to(reconstructor.device)
    ground_truth = ground_truth.to(reconstructor.device)
    recon_no_activation = reconstructor.model.forward(filtbackproj)[1].detach()

    all_modules_under_prior = bayesianised_model.get_all_modules_under_prior()
    map_weights = get_weight_block_vec(all_modules_under_prior).detach()
    ray_trafo_module = ray_trafos['ray_trafo_module'].to(reconstructor.device)
    ray_trafo_module_adj = ray_trafos['ray_trafo_module_adj'].to(reconstructor.device)

    lin_w_fd = nn.Parameter(torch.zeros_like(map_weights).clone()).to(reconstructor.device)    
    optimizer = torch.optim.Adam([lin_w_fd], **{'lr': cfg.lin_params.lr}, weight_decay=0)
    
    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    comment = 'lin_weights_optim'
    logdir = os.path.join(
        './',
        current_time + '_' + socket.gethostname() + comment)
    writer = SummaryWriter(log_dir=logdir)
    loss_vec_fd, psnr = [], []

    if cfg.mrglik.impl.use_fwAD_for_jvp:
        fwAD_model, fwAD_module_mapping = get_fwAD_model(reconstructor.model, return_module_mapping=True, share_parameters=True)
        fwAD_modules = [fwAD_module_mapping[m] for m in all_modules_under_prior]

    reconstructor.model.eval()
    with tqdm(range(cfg.lin_params.iterations), miniters=cfg.lin_params.iterations//1000) as pbar:
        for i in pbar:

            if cfg.lin_params.simplified_eqn:
                fd_vector = lin_w_fd
            else:
                fd_vector = lin_w_fd - map_weights

            if cfg.mrglik.impl.use_fwAD_for_jvp:
                lin_pred = fwAD_JvP(filtbackproj, fwAD_model, fd_vector, fwAD_modules, pre_activation=True, saturation_safety=False).detach()
            else:
                lin_pred = finite_diff_JvP(filtbackproj, reconstructor.model, fd_vector, all_modules_under_prior, pre_activation=True, saturation_safety=False).detach()
            
            if not cfg.lin_params.simplified_eqn:
                lin_pred = lin_pred + recon_no_activation

            if cfg.net.arch.use_sigmoid:
                lin_pred = lin_pred.sigmoid()
            
            loss = torch.nn.functional.mse_loss(ray_trafo_module(lin_pred), observation.to(reconstructor.device)) \
                + cfg.net.optim.gamma * tv_loss(lin_pred)

            v = 2 / observation.numel() * log_homoGauss_grad(ray_trafo_module(lin_pred), observation, ray_trafo_module_adj).flatten() \
                + cfg.net.optim.gamma * tv_loss_grad(lin_pred).flatten() 

            if cfg.net.arch.use_sigmoid:
                v = v * lin_pred.flatten() * (1 - lin_pred.flatten())
            
            optimizer.zero_grad()
            reconstructor.model.zero_grad()
            to_grad = reconstructor.model(filtbackproj)[1].flatten() * v
            to_grad.sum().backward()
            lin_w_fd.grad = agregate_flatten_weight_grad(all_modules_under_prior) + cfg.lin_params.wd * lin_w_fd.detach()
            optimizer.step()

            loss_vec_fd.append(loss.detach().item())
            psnr.append(PSNR(lin_pred.detach().cpu().numpy(), ground_truth.cpu().numpy()))
            pbar.set_description('psnr={:.1f}'.format(psnr[-1]), refresh=False)
            writer.add_scalar('loss', loss_vec_fd[-1], i)
            writer.add_scalar('psnr', psnr[-1], i)

    return lin_w_fd.detach(), lin_pred.detach()