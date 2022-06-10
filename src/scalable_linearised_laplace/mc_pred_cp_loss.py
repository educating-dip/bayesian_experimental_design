from math import ceil
import torch
import torch.autograd as autograd
from .vec_weight_prior_mul_closure import compose_cov_from_modules, fast_prior_cov_mul
from .jvp import fwAD_JvP_batch_ensemble, finite_diff_JvP_batch_ensemble
from .batch_jac import vec_jac_mul_batch

def batch_tv_grad(x):

    assert x.shape[-1] == x.shape[-2]
    batch_size = x.shape[0]
    sign_diff_x = torch.sign(torch.diff(-x, n=1, dim=-1))
    pad = torch.zeros((batch_size, 1, x.shape[-2], 1), device = x.device)
    diff_x_pad = torch.cat([pad, sign_diff_x, pad], dim=-1)
    grad_tv_x = torch.diff(diff_x_pad, n=1, dim=-1)
    sign_diff_y = torch.sign(torch.diff(-x, n=1, dim=-2))
    pad = torch.zeros((batch_size, 1, 1, x.shape[-1]), device = x.device)
    diff_y_pad = torch.cat([pad, sign_diff_y, pad], dim=-2)
    grad_tv_y = torch.diff(diff_y_pad, n=1, dim=-2)
    
    return grad_tv_x + grad_tv_y


def _sample_from_prior_over_weights(bayesianized_model, mc_samples):

    chol_under_gp_prior, chol_under_normal_prior = compose_cov_from_modules(
            bayesianized_model, return_inverse=False, return_cholesky=True) # num_filts x kernel_size^2 x kernel_size^2, num_filts x 1 x 1
    samples_gp_params = torch.randn(
        mc_samples, chol_under_gp_prior.shape[0]*chol_under_gp_prior.shape[1],
                        device=bayesianized_model.store_device)
    samples_from_gp_priors = (fast_prior_cov_mul(
            samples_gp_params, chol_under_gp_prior)
            if samples_gp_params.shape[1] != 0 else torch.empty(samples_gp_params.shape[0], 0).to(samples_gp_params.device))
    samples_normal_params = torch.randn(
        mc_samples, chol_under_normal_prior.shape[0]*chol_under_normal_prior.shape[1],
                        device=bayesianized_model.store_device)
    samples_from_normal_priors = (fast_prior_cov_mul(
            samples_normal_params, chol_under_normal_prior)
            if samples_normal_params.shape[1] != 0 else torch.empty(samples_normal_params.shape[0], 0).to(samples_normal_params.device))
    return torch.cat([samples_from_gp_priors, samples_from_normal_priors], dim=-1)

def compute_log_hyperparams_grads(first_derivative_grad_log_hyperparams, second_derivative_grad_log_hyperparams, tv_scaling_fct):
    grads = []
    for first_derivative_grad_log_hyperparam, second_derivative_grad_log_hyperparam in zip(first_derivative_grad_log_hyperparams, second_derivative_grad_log_hyperparams): 
        grads.append( - ( -first_derivative_grad_log_hyperparam + second_derivative_grad_log_hyperparam) * tv_scaling_fct)
    return grads


def set_gp_priors_grad_predcp(hooked_model, filtbackproj, bayesianized_model, be_model, be_modules, mc_samples, vec_batch_size, tv_scaling_fct, use_fwAD_for_jvp=True):
    num_batches = ceil(mc_samples / vec_batch_size)
    mc_samples = num_batches * vec_batch_size
    sample_weight_vec = _sample_from_prior_over_weights(bayesianized_model, mc_samples)
    v_jac_mul = []
    for i in range(num_batches):
        sample_weight_vec_batch = sample_weight_vec.detach()[i*vec_batch_size:(i+1)*vec_batch_size]
        if use_fwAD_for_jvp:
            s_image = fwAD_JvP_batch_ensemble(filtbackproj, be_model, sample_weight_vec_batch, be_modules)
        else:
            s_image = finite_diff_JvP_batch_ensemble(filtbackproj, be_model, sample_weight_vec_batch, be_modules)
        s_image = s_image.squeeze(dim=1) # remove trivial sample-batch dimension (be_model uses B_ensemble x B_sample x C x H x W)
        tv_s_image_grad = batch_tv_grad(s_image)
        v_jac_mul_batch = vec_jac_mul_batch(hooked_model, filtbackproj, tv_s_image_grad.view(vec_batch_size, -1), bayesianized_model).detach()
        v_jac_mul.append(v_jac_mul_batch)
    v_jac_mul = torch.cat(v_jac_mul, axis=0)
    loss = (sample_weight_vec * v_jac_mul.detach()).sum(dim=1).mean(dim=0)
    first_derivative_grad_log_lengthscales = autograd.grad(loss, bayesianized_model.gp_log_lengthscales, allow_unused=True, create_graph=True, retain_graph=True)
    first_derivative_grad_log_variances = autograd.grad(loss, bayesianized_model.gp_log_variances, allow_unused=True, create_graph=True, retain_graph=True)
    log_dets = [grad.abs().log() for grad in first_derivative_grad_log_lengthscales]
    second_derivative_grad_log_lengthscales = [autograd.grad(log_det, log_lengthscale, allow_unused=True, retain_graph=True)[0] for log_det, log_lengthscale in zip(log_dets, bayesianized_model.gp_log_lengthscales)]
    second_derivative_grad_log_variances = [autograd.grad(log_det, log_variance, allow_unused=True, retain_graph=True)[0] for log_det, log_variance in zip(log_dets, bayesianized_model.gp_log_variances)]

    log_lengthscales_grads = compute_log_hyperparams_grads(first_derivative_grad_log_lengthscales, second_derivative_grad_log_lengthscales, tv_scaling_fct)
    bayesianized_model.set_gp_log_lengthscales_grad(log_lengthscales_grads)

    log_variances_grads = compute_log_hyperparams_grads(first_derivative_grad_log_variances, second_derivative_grad_log_variances, tv_scaling_fct)
    bayesianized_model.set_gp_log_variances_grad(log_variances_grads)
    loss = tv_scaling_fct * (loss.detach() - torch.stack(log_dets).sum().detach())

    return loss