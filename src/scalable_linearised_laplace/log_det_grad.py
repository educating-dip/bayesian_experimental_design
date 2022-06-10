import torch
import numpy as np
from .prior_cov_obs import get_prior_cov_obs_mat

def _get_gp_priors_lengthscales_grad(bayesianized_model):
    grad_cov_gp_prior = []
    for num_filters, prior in zip(
            bayesianized_model.ref_num_filters_per_modules_under_gp_priors,
                bayesianized_model.gp_priors):
        grad_cov_gp_prior.append(
                prior.cov.log_lengthscale_cov_mat_grad().expand(num_filters, 9, 9)
            )
    return grad_cov_gp_prior

def _get_gp_prior_mrg_variances_grad(bayesianized_model):

    grad_cov_gp_prior = []
    for num_filters, prior in zip(
            bayesianized_model.ref_num_filters_per_modules_under_gp_priors,
                bayesianized_model.gp_priors):
        grad_cov_gp_prior.append(
                prior.cov.log_varainces_cov_mat_grad().expand(num_filters, 9, 9)
            )
    return grad_cov_gp_prior

def _get_normal_prior_mrg_variances_grad(bayesianized_model):

    grad_cov_normal_prior = []
    for num_params, prior in zip(bayesianized_model.ref_num_params_per_modules_under_normal_priors, bayesianized_model.normal_priors): 
        grad_cov_normal_prior.append(
                torch.exp(prior.log_variance) * torch.ones(1, device=bayesianized_model.store_device).expand(num_params, 1, 1)
                )
    return grad_cov_normal_prior

def _construct_grad_list_from_modules(bayesianized_model):

    return (_get_gp_priors_lengthscales_grad(bayesianized_model), 
                _get_gp_prior_mrg_variances_grad(bayesianized_model), 
                _get_normal_prior_mrg_variances_grad(bayesianized_model)
                )

def compose_masked_cov_grad_from_modules(bayesianized_model, log_noise_model_variance_obs):

    gp_priors_lengthscales_grad, gp_priors_mrg_var_grad, normal_priors_mrg_var_grad = _construct_grad_list_from_modules(bayesianized_model)

    masked_gp_priors_zeros = torch.zeros(
        (np.sum(bayesianized_model.ref_num_filters_per_modules_under_gp_priors), 9, 9),
        device=bayesianized_model.store_device
        )
    masked_normal_priors_zeros = torch.zeros(
        (np.sum(bayesianized_model.ref_num_params_per_modules_under_normal_priors), 1, 1),
        device=bayesianized_model.store_device
        )

    gp_priors_grad_dict = {'lengthscales': {}, 'variances': {}}
    gp_priors_grad_dict['all_zero'] = masked_gp_priors_zeros
    normal_priors_grad_dict = {'variances': {}}
    normal_priors_grad_dict['all_zero'] = masked_normal_priors_zeros

    num_params_start = 0
    for prior, num_params, prior_lengthscale_grad, prior_mrg_var_grad in zip(bayesianized_model.gp_priors,
        bayesianized_model.ref_num_filters_per_modules_under_gp_priors, gp_priors_lengthscales_grad, gp_priors_mrg_var_grad):

            grads = masked_gp_priors_zeros.clone()
            grads[num_params_start: num_params_start + num_params, :, :] = prior_lengthscale_grad
            gp_priors_grad_dict['lengthscales'][prior] = grads
            grads = masked_gp_priors_zeros.clone()
            grads[num_params_start: num_params_start + num_params, :, :] = prior_mrg_var_grad
            gp_priors_grad_dict['variances'][prior] = grads
            num_params_start += num_params
    
    num_params_start = 0
    for prior, num_params, normal_prior_mrg_var_grad in zip(bayesianized_model.normal_priors, 
        bayesianized_model.ref_num_params_per_modules_under_normal_priors, normal_priors_mrg_var_grad):
            grads = masked_normal_priors_zeros.clone()
            grads[num_params_start: num_params_start + num_params, :, :] = normal_prior_mrg_var_grad
            normal_priors_grad_dict['variances'][prior] = grads
            num_params_start += num_params
    
    # including grad w.r.t. σ^2_y
    log_noise_variance_obs_grad_dict = {}
    log_noise_variance_obs_grad_dict['gp_prior'] = torch.exp(log_noise_model_variance_obs) * torch.ones(*masked_gp_priors_zeros.shape, device=bayesianized_model.store_device)
    log_noise_variance_obs_grad_dict['normal_prior'] = torch.exp(log_noise_model_variance_obs) * torch.ones(*masked_normal_priors_zeros.shape, device=bayesianized_model.store_device)
    
    return gp_priors_grad_dict, normal_priors_grad_dict, log_noise_variance_obs_grad_dict

def compute_exact_log_det_grad(ray_trafos, filtbackproj, bayesianized_model, hooked_model, fwAD_be_model, fwAD_be_modules, log_noise_model_variance_obs, vec_batch_size, use_fwAD_for_jvp=True):
    
    cov_obs_mat = get_prior_cov_obs_mat(ray_trafos, filtbackproj, bayesianized_model, hooked_model, fwAD_be_model, fwAD_be_modules, log_noise_model_variance_obs, vec_batch_size, use_fwAD_for_jvp=use_fwAD_for_jvp)
    cov_obs_mat_inv = torch.inverse(cov_obs_mat)
    del cov_obs_mat
    gp_priors_grad_dict, normal_priors_grad_dict, log_noise_variance_obs_grad_dict = compose_masked_cov_grad_from_modules(bayesianized_model, log_noise_model_variance_obs)
    grads = {}
    
    for gp_prior in bayesianized_model.gp_priors:
        # building A * J * (δΣ_θ / δ_hyperparams) * J.T * A.T
        AJGradsATJT_lengthscale = get_prior_cov_obs_mat(ray_trafos, filtbackproj, bayesianized_model, hooked_model, fwAD_be_model, fwAD_be_modules, 
            log_noise_model_variance_obs, vec_batch_size, use_fwAD_for_jvp=use_fwAD_for_jvp, masked_cov_grads=(gp_priors_grad_dict['lengthscales'][gp_prior], normal_priors_grad_dict['all_zero']), add_noise_model_variance_obs=False)
        grads[gp_prior.cov.log_lengthscale] = 0.5 * torch.trace(cov_obs_mat_inv @ AJGradsATJT_lengthscale).detach()

        AJGradsATJT_variances = get_prior_cov_obs_mat(ray_trafos, filtbackproj, bayesianized_model, hooked_model, fwAD_be_model, fwAD_be_modules, 
            log_noise_model_variance_obs, vec_batch_size, use_fwAD_for_jvp=use_fwAD_for_jvp, masked_cov_grads=(gp_priors_grad_dict['variances'][gp_prior], normal_priors_grad_dict['all_zero']), add_noise_model_variance_obs=False)
        grads[gp_prior.cov.log_variance] = 0.5 * torch.trace(cov_obs_mat_inv @ AJGradsATJT_variances).detach()

    for normal_prior in bayesianized_model.normal_priors:
        # building A * J * (δΣ_θ / δ_hyperparams) * J.T * A.T
        AJGradsATJT = get_prior_cov_obs_mat(ray_trafos, filtbackproj, bayesianized_model, hooked_model, fwAD_be_model, fwAD_be_modules, 
            log_noise_model_variance_obs, vec_batch_size, use_fwAD_for_jvp=use_fwAD_for_jvp, masked_cov_grads=(gp_priors_grad_dict['all_zero'], normal_priors_grad_dict['variances'][normal_prior]), add_noise_model_variance_obs=False)
        grads[normal_prior.log_variance] = 0.5 * torch.trace(cov_obs_mat_inv @ AJGradsATJT).detach()

    AJGradsATJT = get_prior_cov_obs_mat(ray_trafos, filtbackproj, bayesianized_model, hooked_model, fwAD_be_model, fwAD_be_modules, 
            log_noise_model_variance_obs, vec_batch_size, use_fwAD_for_jvp=use_fwAD_for_jvp, masked_cov_grads=(log_noise_variance_obs_grad_dict['gp_prior'], log_noise_variance_obs_grad_dict['normal_prior']), add_noise_model_variance_obs=False)
    grads[log_noise_model_variance_obs] = 0.5 * torch.trace(cov_obs_mat_inv @ AJGradsATJT).detach()

    return grads