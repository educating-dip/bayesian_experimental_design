import torch
import numpy as np
from tqdm import tqdm
from math import ceil 
from scalable_linearised_laplace import ( fwAD_JvP_batch_ensemble, _sample_from_prior_over_weights, 
    cov_image_mul,
    )

def apply_inversion_lemma_tsvd_mul(v, U, S, Vh, log_noise_model_variance_obs, eps=1e-6, full_diag_eps=0.):

    noise_model_variance_obs_and_eps = torch.exp(log_noise_model_variance_obs) + full_diag_eps
    return ( v / noise_model_variance_obs_and_eps) - ( U @ torch.linalg.solve( 
        (torch.diag(1 / (S + eps) ) + Vh @ U / noise_model_variance_obs_and_eps), 
        Vh @ v.T / (noise_model_variance_obs_and_eps ** 2) ) ).T

def sample_weights_from_objective_prior(bayesianized_model, vec_batch_size, tuple_scale_vec):
    
    scale_vec, g_coeff = tuple_scale_vec
    num_weights = np.sum(
        bayesianized_model.ref_num_params_per_modules_under_gp_priors + bayesianized_model.ref_num_params_per_modules_under_normal_priors
        )
    return ( g_coeff**.5 ) * torch.ones(
        (vec_batch_size, num_weights), device=bayesianized_model.store_device).normal_() / scale_vec

def sample_observations_shifted(ray_trafo_module, ray_trafo_module_adj, ray_trafo_comp_module, 
    filtbackproj, 
    cov_obs_mat_chol_or_tsvd, 
    hooked_model, bayesianized_model, be_model, be_modules, 
    log_noise_model_variance_obs,
    mc_samples, 
    vec_batch_size,
    tuple_scale_vec=None,
    device=None
    ):

    if device is None:
        device = bayesianized_model.store_device

    num_batches = ceil(mc_samples / vec_batch_size)
    mc_samples = num_batches * vec_batch_size
    s_observation_samples = []
    s_images_samples = []
    for _ in tqdm(range(num_batches), desc='sample_from_posterior', miniters=num_batches//100):
        if tuple_scale_vec is None:
            sample_weight_vec = _sample_from_prior_over_weights(bayesianized_model, vec_batch_size).detach()
        else:
            sample_weight_vec = sample_weights_from_objective_prior(bayesianized_model, vec_batch_size, tuple_scale_vec)

        s_image = fwAD_JvP_batch_ensemble(filtbackproj, be_model, sample_weight_vec, be_modules)
        s_image = s_image.squeeze(dim=1) # remove trivial sample-batch dimension (be_model uses B_ensemble x B_sample x C x H x W)
        s_observation = ray_trafo_module(s_image)
        s_observation_comp = ray_trafo_comp_module(s_image)

        noise_term = (log_noise_model_variance_obs.exp()**.5 ) * torch.ones(
                        (vec_batch_size, s_observation.numel() // vec_batch_size),
                        device=device
                ).normal_()

        obs = noise_term - s_observation.view(vec_batch_size, -1)
        if not isinstance(cov_obs_mat_chol_or_tsvd, tuple):
            inv_obs = torch.triangular_solve(torch.triangular_solve(obs.T, cov_obs_mat_chol_or_tsvd, upper=False)[0], cov_obs_mat_chol_or_tsvd.T, upper=True)[0].T
        else:
            U, S, Vh = cov_obs_mat_chol_or_tsvd
            inv_obs = apply_inversion_lemma_tsvd_mul(obs, U, S, Vh, log_noise_model_variance_obs)
        inv_image = ray_trafo_module_adj(inv_obs.view(*s_observation.shape))
        delta_x = cov_image_mul(inv_image.view(vec_batch_size, -1), filtbackproj, hooked_model, bayesianized_model, be_model, be_modules, tuple_scale_vec=tuple_scale_vec)
        delta_y = ray_trafo_comp_module(delta_x.view(*s_image.shape))

        s_observation_samples.append((s_observation_comp + delta_y).detach().to(device))
        s_images_samples.append(s_image + delta_x.view(*s_image.shape))

    s_observation_samples = torch.cat(s_observation_samples, axis=0)
    s_images_samples = torch.cat(s_images_samples, axis=0)

    return s_observation_samples, s_images_samples

    