import torch
import numpy as np
from .vec_weight_prior_mul_closure import vec_weight_prior_cov_mul

def get_cov_image_mat(bayesianized_model, jac, eps=None):
    cov_image_mat = vec_weight_prior_cov_mul(bayesianized_model, jac) @ jac.T  # J * Σ_θ * J.T
    if eps is not None:
        cov_image_mat[np.diag_indices(cov_image_mat.shape[0])] += eps
    return cov_image_mat

def get_exact_predictive_cov_image_mat(ray_trafos, bayesianized_model, jac, log_noise_model_variance_obs, eps=None, cov_image_eps=None, cov_obs_mat=None):
    device = jac.device
    ray_trafo_mat = torch.from_numpy(ray_trafos['ray_trafo_mat'])
    ray_trafo_mat = ray_trafo_mat.view(ray_trafo_mat.shape[0] * ray_trafo_mat.shape[1], -1).to(jac.dtype).to(device)
    cov_image_mat = get_cov_image_mat(bayesianized_model, jac, eps=cov_image_eps)
    if cov_obs_mat is None:
        print('computing cov_obs_mat')
        cov_obs_mat = ray_trafo_mat @ cov_image_mat @ ray_trafo_mat.T + torch.exp(log_noise_model_variance_obs) * torch.eye(ray_trafo_mat.shape[0], device=device)
    else:
        print('using custom cov_obs_mat')
    predictive_cov_image_mat = (cov_image_mat - cov_image_mat @ ray_trafo_mat.T @ torch.linalg.solve(
                cov_obs_mat,
                ray_trafo_mat @ cov_image_mat.T))
    if eps is not None:
        predictive_cov_image_mat[np.diag_indices(cov_image_mat.shape[0])] += eps
    return predictive_cov_image_mat

# def compute_exact_predictive_cov_image_log_det(
#         ray_trafos, bayesianized_model, jac, log_noise_model_variance_obs, cov_image_eps):

#     device = jac.device

#     ray_trafo_mat = torch.from_numpy(ray_trafos['ray_trafo_mat'])
#     ray_trafo_mat = ray_trafo_mat.view(ray_trafo_mat.shape[0] * ray_trafo_mat.shape[1], -1).to(device)

#     cov_image_mat = get_cov_image_mat(bayesianized_model, jac, eps=cov_image_eps)  # K_ff

#     cov_obs_mat = (ray_trafo_mat @ cov_image_mat @ ray_trafo_mat.T +
#             torch.exp(log_noise_model_variance_obs) * torch.eye(ray_trafo_mat.shape[0], device=device))

#     ray_trafo_mat = ray_trafo_mat.double()
#     cov_image_mat = cov_image_mat.double()
#     cov_obs_mat = cov_obs_mat.double()

# #     predictive_cov_image_mat_inv_explicit = (
# #             torch.inverse(cov_image_mat) +
# #             1./torch.exp(log_noise_model_variance_obs) * ray_trafo_mat.T @ ray_trafo_mat)

# #     predictive_cov_image_mat_direct_inversion = (
# #             cov_image_mat -
# #             cov_image_mat @ ray_trafo_mat.T @ torch.inverse(cov_obs_mat) @
# #                 ray_trafo_mat @ cov_image_mat.T)

# #     cov_obs_mat_chol = torch.linalg.cholesky(cov_obs_mat)

# #     predictive_cov_image_mat = (
# #             cov_image_mat -
# #             cov_image_mat @ ray_trafo_mat.T @ torch.triangular_solve(
# #                     torch.triangular_solve(ray_trafo_mat @ cov_image_mat.T, cov_obs_mat_chol, upper=False)[0],
# #                     cov_obs_mat_chol.T, upper=True)[0])

# #     predictive_cov_image_slogdet_explicit = torch.slogdet(predictive_cov_image_mat_inv_explicit)
# #     assert predictive_cov_image_slogdet_explicit[0] == 1.
# #     predictive_cov_image_log_det_explicit = -predictive_cov_image_slogdet_explicit[1]
# #     predictive_cov_image_log_det_direct_inversion = torch.logdet(predictive_cov_image_mat_direct_inversion)
# #     predictive_cov_image_log_det_expanded = torch.logdet(predictive_cov_image_mat)

#     predictive_cov_image_log_det = -(torch.logdet(cov_obs_mat) - torch.logdet(cov_image_mat) - log_noise_model_variance_obs * cov_obs_mat.shape[0])

# #     print(predictive_cov_image_log_det_explicit)
# #     print(predictive_cov_image_log_det_direct_inversion)
# #     print(predictive_cov_image_log_det_expanded)
# #     print(predictive_cov_image_log_det)

#     return predictive_cov_image_log_det
