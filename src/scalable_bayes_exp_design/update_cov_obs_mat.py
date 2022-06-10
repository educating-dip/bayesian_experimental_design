import torch
from scalable_linearised_laplace import cov_image_mul

def update_cov_obs_mat_no_noise(
            cov_obs_mat_no_noise, ray_trafo_module, ray_trafo_top_k_module,
            filtbackproj, hooked_model, bayesianized_model, be_model, be_modules, 
            vec_batch_size, tuple_scale_vec=None
    ):
    
    row_numel = ray_trafo_top_k_module.matrix.shape[0]
    top_k_cov_obs_diag = []
    top_k_cov_obs_off_diag = []
    for i in range(0, row_numel, vec_batch_size):
        if ray_trafo_top_k_module.matrix.is_sparse:
            v = torch.stack([ray_trafo_top_k_module.matrix[j] for j in range(i, min(i+vec_batch_size, row_numel))]).to_dense()
        else:
            v = ray_trafo_top_k_module.matrix[i:i+vec_batch_size, :]
        eff_batch_size = v.shape[0]
        if eff_batch_size < vec_batch_size:
            v = torch.nn.functional.pad(v, (0, 0, 0, vec_batch_size-eff_batch_size))
        v = cov_image_mul(v, filtbackproj, hooked_model, bayesianized_model, be_model, be_modules, tuple_scale_vec=tuple_scale_vec)
        if eff_batch_size < vec_batch_size:
            v = v[:eff_batch_size]
        top_k_cov_obs_diag.append((ray_trafo_top_k_module.matrix @ v.T).T)
        top_k_cov_obs_off_diag.append((ray_trafo_module.matrix @ v.T).T)

    top_k_cov_obs_diag = torch.cat(top_k_cov_obs_diag, dim=0)
    top_k_cov_obs_diag = 0.5 * (top_k_cov_obs_diag + top_k_cov_obs_diag.T)  # numerical stability
    top_k_cov_obs_off_diag = torch.cat(top_k_cov_obs_off_diag, dim=0)

    updated_top_cov_obs_mat = torch.cat([cov_obs_mat_no_noise, top_k_cov_obs_off_diag.T], dim=1)
    updated_bottom_cov_obs_mat = torch.cat([top_k_cov_obs_off_diag, top_k_cov_obs_diag], dim=1)
    updated_cov_obs_mat = torch.cat([updated_top_cov_obs_mat, updated_bottom_cov_obs_mat], dim=0)
    
    return updated_cov_obs_mat
