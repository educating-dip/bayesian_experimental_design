from functools import lru_cache
import torch
import numpy as np
from tqdm import tqdm
from .vec_weight_prior_mul_closure import vec_weight_prior_cov_mul
from .batch_jac import vec_jac_mul_batch
from .jvp import fwAD_JvP_batch_ensemble
from .prior_cov_obs import get_prior_cov_obs_mat
from .utils import bisect_left  # for python >= 3.10 one can use instead: from bisect import bisect_left

def predictive_cov_image_block_norm(v, predictive_cov_image_block):
    v_out = torch.linalg.solve(predictive_cov_image_block, v)
    norm = torch.sum(v * v_out, dim=-1)
    return norm

def predictive_image_block_log_prob(recon_masked, ground_truth_masked, predictive_cov_image_block):
    approx_slogdet = torch.slogdet(predictive_cov_image_block)
    assert approx_slogdet[0] > 0.
    approx_log_det = approx_slogdet[1]
    diff = (ground_truth_masked - recon_masked).flatten()
    norm = predictive_cov_image_block_norm(diff, predictive_cov_image_block)
    approx_log_prob = -0.5 * norm - 0.5 * approx_log_det + -0.5 * np.log(2. * np.pi) * ground_truth_masked.numel()
    return approx_log_prob

def predictive_image_block_log_prob_batched(recon_masked, ground_truth_masked, predictive_cov_image_block):
    approx_slogdet = torch.slogdet(predictive_cov_image_block)
    assert torch.all(approx_slogdet[0] > 0.)
    approx_log_det = approx_slogdet[1]
    diff = (ground_truth_masked - recon_masked).view(ground_truth_masked.shape[0], -1)
    norm = predictive_cov_image_block_norm(diff, predictive_cov_image_block)
    approx_log_prob = -0.5 * norm - 0.5 * approx_log_det + -0.5 * np.log(2. * np.pi) * np.prod(ground_truth_masked.shape[1:])
    return approx_log_prob

# speed bottleneck, for walnut: vec_jac_mul_batch ~ 63%; fwAD_JvP_batch_ensemble ~ 37%
def cov_image_mul(v, filtbackproj, hooked_model, bayesianized_model, fwAD_be_model, fwAD_be_modules, tuple_scale_vec=None):
    v = vec_jac_mul_batch(hooked_model, filtbackproj, v, bayesianized_model)  # v * J
    if tuple_scale_vec is None:
        v = vec_weight_prior_cov_mul(bayesianized_model, v)  # v * Σ_θ
    else:
        scale_vec, g_coeff = tuple_scale_vec
        assert scale_vec.size() == torch.Size([v.shape[1]])
        v  = v * g_coeff * scale_vec[None, :]**-2
    v = fwAD_JvP_batch_ensemble(filtbackproj, fwAD_be_model, v, fwAD_be_modules)  # v * J.T
    v = v.view(v.shape[0], -1)
    return v

def cov_image_mul_via_jac(v, jac, bayesianized_model):
    v = v @ jac  # v * J
    v = vec_weight_prior_cov_mul(bayesianized_model, v)  # v * Σ_θ
    v = (jac @ v.T).T  # v * J.T
    return v

def cov_image_mul_via_jac_low_rank(v, jac, bayesianized_model):
    jac_U, jac_S, jac_Vh = jac
    v = v.to(jac_U.device)
    v = ((v @ jac_U) * jac_S[None, :]) @ jac_Vh  # v * J
    v = v.to(bayesianized_model.store_device)
    v = vec_weight_prior_cov_mul(bayesianized_model, v)  # v * Σ_θ
    v = v.to(jac_U.device)
    v = (jac_U @ (jac_S[:, None] * (jac_Vh @ v.T))).T  # v * J.T
    v = v.to(bayesianized_model.store_device)
    return v

# v @ K_f|y
def predictive_cov_image_block_mul(v, mask, cov_obs_mat_chol, ray_trafos, filtbackproj, hooked_model, bayesianized_model, fwAD_be_model, fwAD_be_modules, cov_image_eps=None):
    # v_input = v
    v_cov_image = cov_image_mul(v, filtbackproj, hooked_model, bayesianized_model, fwAD_be_model, fwAD_be_modules)
    if cov_image_eps is not None:
        v_cov_image = v_cov_image + cov_image_eps * v
    v = v_cov_image
    v = ray_trafos['ray_trafo_module'](v.view(v.shape[0], 1, *ray_trafos['space'].shape)).view(v.shape[0], -1)
    v = torch.triangular_solve(torch.triangular_solve(v.T, cov_obs_mat_chol, upper=False)[0], cov_obs_mat_chol.T, upper=True)[0].T
    v = ray_trafos['ray_trafo_module_adj'](v.view(v.shape[0], 1, *ray_trafos['ray_trafo'].range.shape)).view(v.shape[0], -1)
    v_cov_image_2 = cov_image_mul(v, filtbackproj, hooked_model, bayesianized_model, fwAD_be_model, fwAD_be_modules)
    if cov_image_eps is not None:
        v_cov_image_2 = v_cov_image_2 + cov_image_eps * v
    v = v_cov_image_2
    # v = v_input @ K_ff @ A.T @ Kyy^-1 @ A @ K_ff
    v = v_cov_image[:, mask] - v[:, mask]
    return v

def get_image_block_slices(image_shape, block_size):
    image_size_0, image_size_1 = image_shape
    block_size = min(block_size, min(*image_shape))

    block_slices_0 = []
    for start_0 in range(0, image_size_0 - (block_size-1), block_size):
        if start_0 + block_size < image_size_0 - (block_size-1):
            end_0 = start_0 + block_size
        else:
            # last full block, also include the remaining pixels
            end_0 = image_size_0
        block_slices_0.append(slice(start_0, end_0))
    block_slices_1 = []
    for start_1 in range(0, image_size_1 - (block_size-1), block_size):
        if start_1 + block_size < image_size_1 - (block_size-1):
            end_1 = start_1 + block_size
        else:
            # last full block, also include the remaining pixels
            end_1 = image_size_1
        block_slices_1.append(slice(start_1, end_1))
    return block_slices_0, block_slices_1

def get_image_block_masks(image_shape, block_size, flatten=True):
    block_slices_0, block_slices_1 = get_image_block_slices(image_shape, block_size)

    block_masks = []
    for slice_0 in block_slices_0:
        for slice_1 in block_slices_1:
            mask = np.zeros(image_shape, dtype=bool)
            mask[slice_0, slice_1] = True
            if flatten:
                mask = mask.flatten()
            block_masks.append(mask)
    return block_masks

def get_image_block_mask_inds(image_shape, block_size, flatten=True):
    block_slices_0, block_slices_1 = get_image_block_slices(image_shape, block_size)

    block_mask_inds = []
    for slice_0 in block_slices_0:
        for slice_1 in block_slices_1:
            mask_inds = np.ravel_multi_index(np.mgrid[slice_0,slice_1], image_shape)
            if flatten:
                mask_inds = mask_inds.flatten()
            block_mask_inds.append(mask_inds)
    return block_mask_inds

# block of K_ff
def get_cov_image_mat_block(mask, ray_trafos, filtbackproj, bayesianized_model, hooked_model, fwAD_be_model, fwAD_be_modules, vec_batch_size, eps=None):

    image_shape = (1, 1,) + ray_trafos['space'].shape
    mask_inds = torch.from_numpy(np.nonzero(mask)[0]).to(filtbackproj.device)
    block_numel = len(mask_inds)
    rows = []
    v = torch.empty((vec_batch_size, np.prod(image_shape)), device=filtbackproj.device)
    for i in tqdm(range(0, block_numel, vec_batch_size), desc='get_cov_image_mat_block', miniters=block_numel//vec_batch_size//100):
        v[:] = 0.
        # set v[:, mask] to be a subset of rows of torch.eye(block_numel); in last batch, it may contain some additional (zero) rows
        mask_inds_batch = mask_inds[i:i+vec_batch_size]
        v[list(range(len(mask_inds_batch))), mask_inds_batch] = 1.
        rows_batch = cov_image_mul(v, filtbackproj, hooked_model, bayesianized_model, fwAD_be_model, fwAD_be_modules)
        rows_batch = rows_batch[:, mask_inds]
        if i+vec_batch_size > block_numel:  # last batch
            rows_batch = rows_batch[:block_numel%vec_batch_size]
        rows.append(rows_batch)
    cov_image_mat_block = torch.cat(rows, dim=0)

    cov_image_block = 0.5 * (cov_image_block + cov_image_block.T)  # in case of numerical issues leading to asymmetry

    if eps is not None:
        cov_image_mat_block[np.diag_indices(cov_image_mat_block.shape[0])] += eps

    return cov_image_mat_block

# block of K_f|y
def get_predictive_cov_image_block(mask, cov_obs_mat_chol, ray_trafos, filtbackproj, bayesianized_model, hooked_model, fwAD_be_model, fwAD_be_modules, vec_batch_size, eps=None, cov_image_eps=None, return_cholesky=False):
    image_shape = (1, 1,) + ray_trafos['space'].shape
    mask = np.asarray(mask)
    mask_inds = np.nonzero(mask)[0] if mask.dtype == bool else mask
    block_numel = len(mask_inds)
    rows = []
    v = torch.empty((vec_batch_size, np.prod(image_shape)), device=filtbackproj.device)
    for i in tqdm(range(0, block_numel, vec_batch_size), desc='get_predictive_cov_image_block', miniters=block_numel//vec_batch_size//100):
        v[:] = 0.
        # set v[:, mask_inds] to be a subset of rows of torch.eye(block_numel); in last batch, it may contain some additional (zero) rows
        mask_inds_batch = mask_inds[i:i+vec_batch_size]
        v[list(range(len(mask_inds_batch))), mask_inds_batch] = 1.
        rows_batch = predictive_cov_image_block_mul(v, mask_inds, cov_obs_mat_chol, ray_trafos, filtbackproj, hooked_model, bayesianized_model, fwAD_be_model, fwAD_be_modules, cov_image_eps=cov_image_eps)
        if i+vec_batch_size > block_numel:  # last batch
            rows_batch = rows_batch[:block_numel%vec_batch_size]
        rows.append(rows_batch)
    predictive_cov_image_block = torch.cat(rows, dim=0)

    predictive_cov_image_block = 0.5 * (predictive_cov_image_block + predictive_cov_image_block.T)  # in case of numerical issues leading to asymmetry

    if eps is not None:
        predictive_cov_image_block[np.diag_indices(predictive_cov_image_block.shape[0])] += eps

    return torch.linalg.cholesky(predictive_cov_image_block) if return_cholesky else predictive_cov_image_block

def get_predictive_cov_image_block_via_jac(mask_inds, ray_trafos, bayesianized_model, jac, jac_obs, log_noise_model_variance_obs, cov_obs_mat=None, cov_obs_mat_qr=None, eps=None, assert_diag_pos=True, cov_obs_mat_eps=None, return_cholesky=False):
    image_shape = (1, 1,) + ray_trafos['space'].shape
    block_numel = len(mask_inds)

    if cov_obs_mat is None:
        cov_obs_mat = vec_weight_prior_cov_mul(bayesianized_model, jac_obs) @ jac_obs.transpose(1, 0)  # A * J * Sigma_theta * J.T * A.T
        cov_obs_mat[np.diag_indices(cov_obs_mat.shape[0])] += log_noise_model_variance_obs.exp()
    if cov_obs_mat_qr is None:
        cov_obs_mat_qr = torch.linalg.qr(cov_obs_mat)
    cov_obs_mat_Q, cov_obs_mat_R = cov_obs_mat_qr
    prior_cov_jac_in_mask = vec_weight_prior_cov_mul(bayesianized_model, jac)[mask_inds]
    Kff_in_mask = prior_cov_jac_in_mask @ jac[mask_inds].transpose(1, 0)
    Kxy_in_mask = prior_cov_jac_in_mask @ jac_obs.transpose(1, 0)
    Kyx_in_mask = Kxy_in_mask.T
    model_post_pred_cov_in_mask = Kff_in_mask - Kxy_in_mask @ torch.triangular_solve(cov_obs_mat_Q.T @ Kyx_in_mask, cov_obs_mat_R)[0]

    if eps is not None:
        model_post_pred_cov_in_mask[np.diag_indices(model_post_pred_cov_in_mask.shape[0])] += eps

    if assert_diag_pos:
        assert model_post_pred_cov_in_mask.diag().min() > 0.

    return torch.linalg.cholesky(model_post_pred_cov_in_mask) if return_cholesky else model_post_pred_cov_in_mask

def approx_predictive_cov_image_block_from_samples(mc_sample_images, noise_x_correction_term=None):

    mc_samples = mc_sample_images.shape[0]

    mc_sample_images = mc_sample_images.view(mc_samples, -1)

    # mean = torch.mean(mc_sample_images, dim=0, keepdim=True)
    # diffs = mc_sample_images - mean  # samples x image

    # cov = diffs.T @ diffs / diffs.shape[0]  # image x image

    cov = torch.cov(mc_sample_images.T, correction=0)
    cov = torch.atleast_2d(cov)

    if noise_x_correction_term is not None:
        cov[np.diag_indices(cov.shape[0])] += noise_x_correction_term

    return cov

def approx_predictive_cov_image_block_from_samples_batched(mc_sample_images, noise_x_correction_term=None):

    batch_size, mc_samples, im_numel = mc_sample_images.shape

    mc_sample_images = mc_sample_images.view(batch_size, mc_samples, -1)

    mean = torch.mean(mc_sample_images, dim=1, keepdim=True)
    diffs = mc_sample_images - mean  # batch x samples x image
    diffs = diffs.view(batch_size * mc_samples, -1)

    prods = torch.bmm(diffs[:, :, None], diffs[:, None, :]).view(batch_size, mc_samples, im_numel, im_numel)
    cov = prods.sum(dim=1) / prods.shape[1]  # image x image

    if noise_x_correction_term is not None:
        cov[(slice(None), *np.diag_indices(im_numel))] += noise_x_correction_term

    return cov

def stabilize_predictive_cov_image_block(predictive_cov_image_block, eps_mode, eps):
    block_diag_mean = predictive_cov_image_block.diag().mean().detach().cpu().numpy()
    if eps_mode == 'abs':
        block_eps = eps or 0.
    elif eps_mode == 'rel':
        block_eps = (eps or 0.) * block_diag_mean
    elif eps_mode == 'auto':
        @lru_cache(maxsize=None)
        def predictive_cov_image_block_pos_definite(eps_value):
            try:
                _ = torch.linalg.cholesky(predictive_cov_image_block + eps_value * torch.eye(predictive_cov_image_block.shape[0], device=predictive_cov_image_block.device))
            except RuntimeError:
                return False
            return True
        eps_to_search = [0.] + (list(np.logspace(-6, 0, 1000) * eps * block_diag_mean) if eps else [])
        i_eps = bisect_left(eps_to_search, True, key=predictive_cov_image_block_pos_definite)
        assert i_eps < len(eps_to_search), 'failed to make Kf|y block ({}x{}) cholesky decomposable, max eps is {} == {} * Kf|y.diag().mean()'.format(*predictive_cov_image_block.shape, eps_to_search[-1], eps_to_search[-1] / block_diag_mean)
        block_eps = eps_to_search[i_eps]
    elif eps_mode is None or eps_mode.lower() == 'none':
        block_eps = 0.
    else:
        raise NotImplementedError
    if block_eps != 0.:
        predictive_cov_image_block[np.diag_indices(predictive_cov_image_block.shape[0])] += block_eps
        print('increased diagonal of Kf|y block by {} == {} * Kf|y.diag().mean()'.format(block_eps, block_eps / block_diag_mean))
    return block_eps

def predictive_image_log_prob(
        recon, ground_truth, ray_trafos, bayesianized_model, filtbackproj, hooked_model, fwAD_be_model, fwAD_be_modules, log_noise_model_variance_obs, eps_mode, eps, cov_image_eps, block_size, vec_batch_size, cov_obs_mat_chol=None, noise_x_correction_term=None):


    block_mask_inds = get_image_block_mask_inds(ray_trafos['space'].shape, block_size, flatten=True)

    if cov_obs_mat_chol is None:
        cov_obs_mat = get_prior_cov_obs_mat(ray_trafos, filtbackproj, bayesianized_model, hooked_model, fwAD_be_model, fwAD_be_modules, log_noise_model_variance_obs, vec_batch_size, use_fwAD_for_jvp=True, add_noise_model_variance_obs=True)
        cov_obs_mat = 0.5 * (cov_obs_mat + cov_obs_mat.T)  # in case of numerical issues leading to asymmetry
        cov_obs_mat_chol = torch.linalg.cholesky(cov_obs_mat)

    image_block_diags = []
    image_block_log_probs = []
    block_eps_values = []
    for mask_inds in tqdm(block_mask_inds, desc='image_block_log_probs'):
        predictive_cov_image_block = get_predictive_cov_image_block(
                mask_inds, cov_obs_mat_chol, ray_trafos, filtbackproj, bayesianized_model, hooked_model, fwAD_be_model, fwAD_be_modules, vec_batch_size,
                eps=None,  # handle later
                cov_image_eps=cov_image_eps, return_cholesky=False)
        if noise_x_correction_term is not None: 
            predictive_cov_image_block[np.diag_indices(predictive_cov_image_block.shape[0])] += noise_x_correction_term

        block_eps = stabilize_predictive_cov_image_block(predictive_cov_image_block, eps_mode=eps_mode, eps=eps)
        block_eps_values.append(block_eps)

        image_block_diags.append(predictive_cov_image_block.diag())
        image_block_log_probs.append(predictive_image_block_log_prob(recon.flatten()[mask_inds], ground_truth.flatten()[mask_inds], predictive_cov_image_block))

    approx_image_log_prob = torch.sum(torch.stack(image_block_log_probs))

    return approx_image_log_prob, block_mask_inds, image_block_log_probs, image_block_diags, block_eps_values

def predictive_image_log_prob_from_samples(
        recon, observation, ground_truth, ray_trafos, bayesianized_model, filtbackproj, hooked_model, fwAD_be_model, fwAD_be_modules, log_noise_model_variance_obs, eps_mode, eps, block_size, vec_batch_size, cov_obs_mat_chol=None, mc_sample_images=None, noise_x_correction_term=None, mc_samples=8192):

    block_mask_inds = get_image_block_mask_inds(ray_trafos['space'].shape, block_size, flatten=True)

    if cov_obs_mat_chol is None:
        cov_obs_mat = get_prior_cov_obs_mat(ray_trafos, filtbackproj, bayesianized_model, hooked_model, fwAD_be_model, fwAD_be_modules, log_noise_model_variance_obs, vec_batch_size, use_fwAD_for_jvp=True, add_noise_model_variance_obs=True)
        cov_obs_mat = 0.5 * (cov_obs_mat + cov_obs_mat.T)  # in case of numerical issues leading to asymmetry
        cov_obs_mat_chol = torch.linalg.cholesky(cov_obs_mat)

    if mc_sample_images is None:
        from .sample_based_approx_density import sample_from_posterior
        mc_sample_images = sample_from_posterior(ray_trafos, observation, filtbackproj, cov_obs_mat_chol, hooked_model, bayesianized_model, fwAD_be_model, fwAD_be_modules, log_noise_model_variance_obs, mc_samples, vec_batch_size)

    image_block_diags = []
    image_block_log_probs = []
    block_eps_values = []
    for mask_inds in block_mask_inds:
        predictive_cov_image_block = approx_predictive_cov_image_block_from_samples(
                mc_sample_images.view(mc_sample_images.shape[0], -1)[:, mask_inds], noise_x_correction_term=noise_x_correction_term)

        block_eps = stabilize_predictive_cov_image_block(predictive_cov_image_block, eps_mode=eps_mode, eps=eps)
        block_eps_values.append(block_eps)

        image_block_diags.append(predictive_cov_image_block.diag())
        image_block_log_probs.append(predictive_image_block_log_prob(recon.flatten()[mask_inds], ground_truth.flatten()[mask_inds], predictive_cov_image_block))

    approx_image_log_prob = torch.sum(torch.stack(image_block_log_probs))

    return approx_image_log_prob, block_mask_inds, image_block_log_probs, image_block_diags, block_eps_values
