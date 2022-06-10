import torch
import numpy as np

def assert_positive_diag(K, eps=1e-6):

    K[np.diag_indices(K.shape[0])] += eps
    assert K.diag().min() > 0

def image_space_lin_model_post_pred_cov(block_priors, Jac_x, Jac_y, noise_model_variance_y):

    Kyy = block_priors.matrix_prior_cov_mul(Jac_y) @ Jac_y.transpose(1, 0)  # A * J * Sigma_theta * J.T * A.T
    Kyy[np.diag_indices(Kyy.shape[0])] += noise_model_variance_y
    prior_cov_Jac_x = block_priors.matrix_prior_cov_mul(Jac_x)
    Kff = prior_cov_Jac_x @ Jac_x.transpose(1, 0)
    Kxy = prior_cov_Jac_x @ Jac_y.transpose(1, 0)
    Kyx = Kxy.T
    model_post_pred_cov = Kff - Kxy @ torch.linalg.solve(Kyy, Kyx)
    assert_positive_diag(model_post_pred_cov, eps=1e-6)
    assert_positive_diag(Kff, eps=1e-6)

    return model_post_pred_cov.diag(), model_post_pred_cov, Kff


def submatrix_image_space_lin_model_prior_cov(block_priors, Jac_x):

    idx_list = block_priors.get_idx_parameters_per_block()
    Kff_diag_list = []
    Kff_list = []
    for i, idx in enumerate(idx_list): 

        Kff = block_priors.matrix_prior_cov_mul(Jac_x[:, idx[-2]:idx[-1]], idx=i) @ Jac_x[:, idx[-2]:idx[-1]].transpose(1, 0) 
        assert_positive_diag(Kff, eps=1e-6)
        Kff_diag_list.append(Kff.diag())
        Kff_list.append(Kff)
        
    return Kff_diag_list, Kff_list
