from math import ceil
import torch
from .prior_cov_obs import prior_cov_obs_mat_mul, prior_cov_obs_mat_mul_jac_low_rank
from tqdm import tqdm


def apply_inversion_lemma(U, L, log_noise_model_variance_obs, eps=1e-6):

    noise_model_variance_obs = torch.exp(log_noise_model_variance_obs)
    nosie_mat = torch.eye(U.shape[0], device=U.device) / noise_model_variance_obs
    return nosie_mat - U @ torch.linalg.solve( 
        (torch.diag(1 / (torch.clamp(L, min=eps)) ) + U.T @ U / noise_model_variance_obs), 
        U.T / (noise_model_variance_obs ** 2) )

def apply_inversion_lemma_mul(v, U, L, log_noise_model_variance_obs, eps=1e-6, full_diag_eps=0.):

    noise_model_variance_obs_and_eps = torch.exp(log_noise_model_variance_obs) + full_diag_eps
    return ( v / noise_model_variance_obs_and_eps) - ( U @ torch.linalg.solve( 
        (torch.diag(1 / (L + eps) ) + U.T @ U / noise_model_variance_obs_and_eps), 
        U.T @ v.T / (noise_model_variance_obs_and_eps ** 2) ) ).T

def get_cov_obs_low_rank(random_matrix_T, ray_trafos, filtbackproj, bayesianized_model, 
    hooked_model, be_model, be_modules, 
    log_noise_model_variance_obs,
    vec_batch_size, reduced_rank_dim, oversampling_param, 
    use_fwAD_for_jvp=True,
    use_cpu=False,
    return_inverse=True
    ):
    
    num_batches = ceil((reduced_rank_dim + oversampling_param ) / vec_batch_size)
    v_cov_obs_mat = []
    for i in tqdm(range(num_batches), miniters=num_batches//100, desc='get_cov_obs_low_rank'):
        rnd_vect = random_matrix_T[i * vec_batch_size:(i * vec_batch_size) + vec_batch_size, :].unsqueeze(dim=1)
        eff_batch_size = rnd_vect.shape[0]
        if eff_batch_size < vec_batch_size:
            rnd_vect = torch.cat([rnd_vect, torch.zeros((vec_batch_size-eff_batch_size, *rnd_vect.shape[1:]), dtype=rnd_vect.dtype, device=rnd_vect.device)])
        v = prior_cov_obs_mat_mul(ray_trafos, filtbackproj, bayesianized_model,
            hooked_model, be_model, be_modules, 
            rnd_vect, log_noise_model_variance_obs,
            masked_cov_grads=None, 
            use_fwAD_for_jvp=use_fwAD_for_jvp, add_noise_model_variance_obs=False)
        if eff_batch_size < vec_batch_size:
            v = v[:eff_batch_size]
        v_cov_obs_mat.append(v)
    v_cov_obs_mat = torch.cat(v_cov_obs_mat)
    v_cov_obs_mat = v_cov_obs_mat.view(*v_cov_obs_mat.shape[:1], -1).T
    Q, _ = torch.linalg.qr(v_cov_obs_mat.detach().cpu() if use_cpu else v_cov_obs_mat)
    Q = Q if not use_cpu else Q.to(bayesianized_model.store_device)
    random_matrix_T = random_matrix_T.view(random_matrix_T.shape[0], -1)
    B = torch.linalg.solve(random_matrix_T @ Q, v_cov_obs_mat.T @ Q)
    L, V = torch.linalg.eig(B)
    U = Q @ V.real
    return ( U[:, :reduced_rank_dim], L.real[:reduced_rank_dim], apply_inversion_lemma(U[:, :reduced_rank_dim], L.real[:reduced_rank_dim], log_noise_model_variance_obs) ) \
        if return_inverse else ( U[:, :reduced_rank_dim], L.real[:reduced_rank_dim] )

def get_cov_obs_low_rank_via_jac_low_rank(random_matrix_T, ray_trafos, filtbackproj, bayesianized_model, 
    jac,
    log_noise_model_variance_obs,
    vec_batch_size, reduced_rank_dim, oversampling_param, 
    use_fwAD_for_jvp=True,
    use_cpu=False,
    return_inverse=True
    ):
    
    num_batches = ceil((reduced_rank_dim + oversampling_param ) / vec_batch_size)
    v_cov_obs_mat = []
    for i in tqdm(range(num_batches), miniters=num_batches//100, desc='get_cov_obs_low_rank_via_jac_low_rank'):
        rnd_vect = random_matrix_T[i * vec_batch_size:(i * vec_batch_size) + vec_batch_size, :].unsqueeze(dim=1)
        eff_batch_size = rnd_vect.shape[0]
        if eff_batch_size < vec_batch_size:
            rnd_vect = torch.cat([rnd_vect, torch.zeros((vec_batch_size-eff_batch_size, *rnd_vect.shape[1:]), dtype=rnd_vect.dtype, device=rnd_vect.device)])
        v = prior_cov_obs_mat_mul_jac_low_rank(ray_trafos, bayesianized_model,
            jac,
            rnd_vect, log_noise_model_variance_obs,
            masked_cov_grads=None, 
            add_noise_model_variance_obs=False)
        if eff_batch_size < vec_batch_size:
            v = v[:eff_batch_size]
        v_cov_obs_mat.append(v)
    v_cov_obs_mat = torch.cat(v_cov_obs_mat)
    v_cov_obs_mat = v_cov_obs_mat.view(*v_cov_obs_mat.shape[:1], -1).T
    Q, _ = torch.linalg.qr(v_cov_obs_mat.detach().cpu() if use_cpu else v_cov_obs_mat)
    Q = Q if not use_cpu else Q.to(bayesianized_model.store_device)
    random_matrix_T = random_matrix_T.view(random_matrix_T.shape[0], -1)
    B = torch.linalg.solve(random_matrix_T @ Q, v_cov_obs_mat.T @ Q)
    L, V = torch.linalg.eig(B)
    U = Q @ V.real
    return ( U[:, :reduced_rank_dim], L.real[:reduced_rank_dim], apply_inversion_lemma(U[:, :reduced_rank_dim], L.real[:reduced_rank_dim], log_noise_model_variance_obs) ) \
        if return_inverse else ( U[:, :reduced_rank_dim], L.real[:reduced_rank_dim] )