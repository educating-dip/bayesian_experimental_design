import torch
import numpy as np
from tqdm import tqdm
from math import ceil 
from sklearn.neighbors import KernelDensity
from .jvp import fwAD_JvP_batch_ensemble
from .mc_pred_cp_loss import _sample_from_prior_over_weights
from .approx_density import cov_image_mul, cov_image_mul_via_jac, cov_image_mul_via_jac_low_rank
from .prior_cov_obs import LowRankCovObsMat, prior_cov_obs_mat_mul_jac_low_rank
from .low_rank_preconditioner import apply_inversion_lemma_mul
from .approx_log_det_grad import generate_closure, generate_low_rank_closure
from .gpytorch_linear_cg import linear_cg

def sample_from_posterior(ray_trafos, observation, filtbackproj, cov_obs_mat_chol_or_low_rank, hooked_model, bayesianized_model, be_model, be_modules, log_noise_model_variance_obs, mc_samples, vec_batch_size, device=None, conj_grad_kwards=None):

    if device is None:
        device = bayesianized_model.store_device

    if conj_grad_kwards is None:
        conj_grad_kwards = {'use_conj_grad_inv': False}

    num_batches = ceil(mc_samples / vec_batch_size)
    s_images = []
    res_norm_list = []
    for _ in tqdm(range(num_batches), desc='sample_from_posterior', miniters=num_batches//100):
        sample_weight_vec = _sample_from_prior_over_weights(bayesianized_model, vec_batch_size).detach()
        s_image = fwAD_JvP_batch_ensemble(filtbackproj, be_model, sample_weight_vec, be_modules)
        s_image = s_image.squeeze(dim=1) # remove trivial sample-batch dimension (be_model uses B_ensemble x B_sample x C x H x W)
        s_observation = ray_trafos['ray_trafo_module'](s_image)

        noise_term = (log_noise_model_variance_obs.exp()**.5 ) * torch.ones(
                        s_observation.shape,
                        device=device
                ).normal_()
        
        obs_diff = (observation.expand(*s_observation.shape) + noise_term - s_observation).view(vec_batch_size, -1)
        if not conj_grad_kwards['use_conj_grad_inv']: 
            if isinstance(cov_obs_mat_chol_or_low_rank, LowRankCovObsMat):
                U, L, log_noise_model_variance_obs, full_diag_eps = cov_obs_mat_chol_or_low_rank
                inv_obs_diff = apply_inversion_lemma_mul(obs_diff, U, L, log_noise_model_variance_obs, full_diag_eps=full_diag_eps)
            else:
                inv_obs_diff = torch.triangular_solve(torch.triangular_solve(obs_diff.T, cov_obs_mat_chol_or_low_rank, upper=False)[0], cov_obs_mat_chol_or_low_rank.T, upper=True)[0].T
        else:
            main_closure = generate_closure(
                ray_trafos, filtbackproj, 
                bayesianized_model, hooked_model, be_model, be_modules,
                conj_grad_kwards['log_noise_model_variance_obs'],
                vec_batch_size, 
                side_length=observation.shape[1:],
                masked_cov_grads=None,
                use_fwAD_for_jvp=True,
                add_noise_model_variance_obs=True
            )
            preconditioning_closure = generate_low_rank_closure(
                conj_grad_kwards['preconditioner']
            )
            max_cg_iter = conj_grad_kwards['max_cg_iter']
            inv_obs_diff_T, _, res_norm = linear_cg(main_closure, obs_diff.T, n_tridiag=vec_batch_size, tolerance=conj_grad_kwards['tolerance'],
                    eps=1e-10, stop_updating_after=1e-10, max_iter=max_cg_iter,
                    max_tridiag_iter=max_cg_iter-1, preconditioner=preconditioning_closure)
            res_norm_list.append(
                res_norm
                )
            inv_obs_diff = inv_obs_diff_T.T
        inv_diff = ray_trafos['ray_trafo_module_adj'](inv_obs_diff.view(vec_batch_size, *observation.shape[1:]))
        delta_x = cov_image_mul(inv_diff.view(vec_batch_size, -1), filtbackproj, hooked_model, bayesianized_model, be_model, be_modules)

        s_images.append((s_image + delta_x.view(vec_batch_size, *s_image.shape[1:])).detach().to(device))
    s_images = torch.cat(s_images, axis=0)

    return s_images, res_norm_list

def generate_closure_via_jac_low_rank(ray_trafos, bayesianized_model, jac, 
        log_noise_model_variance_obs, vec_batch_size, side_length, 
        masked_cov_grads=None, add_noise_model_variance_obs=True):

    def closure(v):
        # takes input (side_length x batchsize)
        v = v.T.view(vec_batch_size, *side_length)
        out = prior_cov_obs_mat_mul_jac_low_rank(ray_trafos, bayesianized_model, jac, v, log_noise_model_variance_obs, masked_cov_grads=masked_cov_grads, add_noise_model_variance_obs=add_noise_model_variance_obs)
        out = out.view(vec_batch_size, np.prod(side_length))
        return out.T
    return closure

def sample_from_posterior_via_jac(ray_trafos, observation, jac, cov_obs_mat_chol_or_low_rank, bayesianized_model, log_noise_model_variance_obs, mc_samples, vec_batch_size, device=None, low_rank_jac=False, conj_grad_kwards=None):

    if device is None:
        device = bayesianized_model.store_device

    if conj_grad_kwards is None:
        conj_grad_kwards = {'use_conj_grad_inv': False}

    image_shape = (1, 1,) + ray_trafos['space'].shape

    num_batches = ceil(mc_samples / vec_batch_size)
    s_images = []
    res_norm_list = []
    for _ in tqdm(range(num_batches), desc='sample_from_posterior', miniters=num_batches//100):
        sample_weight_vec = _sample_from_prior_over_weights(bayesianized_model, vec_batch_size).detach()
        if low_rank_jac:
            U, S, Vh = jac
            sample_weight_vec = sample_weight_vec.to(U.device)
            s_image = (U @ (S[:, None] * (Vh @ sample_weight_vec.T))).T
            s_image = s_image.to(bayesianized_model.store_device)
        else:
            s_image = (jac @ sample_weight_vec.T).T
        s_image = s_image.view(*image_shape)
        s_observation = ray_trafos['ray_trafo_module'](s_image)
        
        noise_term = (log_noise_model_variance_obs.exp()**.5 ) * torch.ones(
                        s_observation.shape,
                        device=device
                ).normal_()
        
        obs_diff = (observation.expand(*s_observation.shape) + noise_term - s_observation).view(vec_batch_size, -1)
        if not conj_grad_kwards['use_conj_grad_inv']: 
            if isinstance(cov_obs_mat_chol_or_low_rank, LowRankCovObsMat):
                U, L, log_noise_model_variance_obs, full_diag_eps = cov_obs_mat_chol_or_low_rank
                inv_obs_diff = apply_inversion_lemma_mul(obs_diff, U, L, log_noise_model_variance_obs, full_diag_eps=full_diag_eps)
            else:
                inv_obs_diff = torch.triangular_solve(torch.triangular_solve(obs_diff.T, cov_obs_mat_chol_or_low_rank, upper=False)[0], cov_obs_mat_chol_or_low_rank.T, upper=True)[0].T
        else:
            main_closure = generate_closure_via_jac_low_rank(
                ray_trafos, 
                bayesianized_model, jac,
                conj_grad_kwards['log_noise_model_variance_obs'],
                vec_batch_size, 
                side_length=observation.shape[1:],
                masked_cov_grads=None,
                add_noise_model_variance_obs=True
            )
            preconditioning_closure = generate_low_rank_closure(
                conj_grad_kwards['preconditioner']
            )
            max_cg_iter = conj_grad_kwards['max_cg_iter']
            inv_obs_diff_T, _, res_norm = linear_cg(main_closure, obs_diff.T, n_tridiag=vec_batch_size, tolerance=conj_grad_kwards['tolerance'],
                    eps=1e-10, stop_updating_after=1e-10, max_iter=max_cg_iter,
                    max_tridiag_iter=max_cg_iter-1, preconditioner=preconditioning_closure)
            res_norm_list.append(
                res_norm
            )
            inv_obs_diff = inv_obs_diff_T.T
        inv_diff = ray_trafos['ray_trafo_module_adj'](inv_obs_diff.view(vec_batch_size, *observation.shape[1:]))
        if low_rank_jac:
            U, S, Vh = jac
            delta_x = cov_image_mul_via_jac_low_rank(inv_diff.view(vec_batch_size, -1), (U, S, Vh), bayesianized_model)
        else:
            delta_x = cov_image_mul_via_jac(inv_diff.view(vec_batch_size, -1), jac, bayesianized_model)

        s_images.append((s_image + delta_x.view(vec_batch_size, *s_image.shape[1:])).detach().to(device))
    s_images = torch.cat(s_images, axis=0)

    return s_images, res_norm_list


def approx_density_from_samples(recon, example_image, mc_sample_images, noise_x_correction_term=None):

    assert example_image.shape[1:] == mc_sample_images.shape[1:]
    
    mc_samples = mc_sample_images.shape[0]
    assert noise_x_correction_term is not None

    std = ( torch.var(mc_sample_images.view(mc_samples, -1), dim=0) + noise_x_correction_term) **.5
    dist = torch.distributions.normal.Normal(recon.flatten(), std)
    return dist.log_prob(example_image.flatten()).sum()


def approx_density_from_samples_mult_normal(recon, example_image, mc_sample_images, noise_x_correction_term=None):

    mc_samples = mc_sample_images.shape[0]
    assert noise_x_correction_term is not None
    assert example_image.shape[1:] == mc_sample_images.shape[1:]

    covariance_matrix = torch.cov(mc_sample_images.view(mc_samples, -1).T) 
    covariance_matrix[np.diag_indices(covariance_matrix.shape[0])] += noise_x_correction_term

    dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=recon.flatten(), covariance_matrix=covariance_matrix)
    return dist.log_prob(example_image.flatten())


def approx_kernel_density(example_image, mc_sample_images, bw=0.1, noise_x_correction_term=None):

    if noise_x_correction_term is not None:
        mc_sample_images = mc_sample_images + torch.randn_like(mc_sample_images) * noise_x_correction_term **.5

    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(mc_sample_images.view(mc_sample_images.shape[0], -1).numpy())
    return kde.score_samples(example_image.flatten().numpy()[None, :])
