import torch
import numpy as np
from torch.linalg import cholesky

def gaussian_log_prob(
    X,
    mu,
    model_post_pred_cov = None,
    noise_model_cov = None
    ):

    assert mu.shape == X.shape

    if model_post_pred_cov is not None and noise_model_cov is None:
        covariance_matrix = model_post_pred_cov
    elif noise_model_cov is not None and model_post_pred_cov is None:
        covariance_matrix = noise_model_cov
    elif model_post_pred_cov is not None and noise_model_cov is not None:
        covariance_matrix = noise_model_cov + model_post_pred_cov

    suceed = False
    cnt = 0
    while not suceed:
        try:
            dist = torch.distributions.multivariate_normal.MultivariateNormal(
                loc=mu, 
                scale_tril=cholesky(covariance_matrix)
            )
            suceed = True
        except:
            covariance_matrix[np.diag_indices(X.shape[0])] += 1e-6
            cnt += 1
            assert cnt < 1000 # safety 

    log_prob = dist.log_prob(X)
    return log_prob / X.shape[0]