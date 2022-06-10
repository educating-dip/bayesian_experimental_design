from math import ceil
from tqdm import tqdm
import torch
from .jvp import fwAD_JvP_batch_ensemble
from .batch_jac import vec_jac_mul_batch

def get_batched_jac_low_rank(random_matrix, filtbackproj, bayesianized_model, 
    hooked_model, be_model, be_modules, 
    vec_batch_size, low_rank_rank_dim, oversampling_param, 
    use_cpu=False,
    return_on_cpu=False,
    ):
    
    num_batches = ceil((low_rank_rank_dim + oversampling_param) / vec_batch_size)
    low_rank_jac_v_mat = []
    for i in tqdm(range(num_batches), miniters=num_batches//100, desc='get_batched_jac_low_rank forward'):
        rnd_vect = random_matrix[:, i * vec_batch_size:(i * vec_batch_size) + vec_batch_size]
        low_rank_jac_v_mat_row = fwAD_JvP_batch_ensemble(filtbackproj, be_model, rnd_vect.T, be_modules).detach()
        low_rank_jac_v_mat.append( low_rank_jac_v_mat_row.cpu() if use_cpu else low_rank_jac_v_mat_row )
    low_rank_jac_v_mat = torch.cat(low_rank_jac_v_mat)
    low_rank_jac_v_mat = low_rank_jac_v_mat.view(*low_rank_jac_v_mat.shape[:1], -1).T
    Q, _ = torch.linalg.qr(low_rank_jac_v_mat)
    if not return_on_cpu:
        Q = Q.to(bayesianized_model.store_device)

    qT_low_rank_jac_mat = []
    for i in tqdm(range(num_batches), miniters=num_batches//100, desc='get_batched_jac_low_rank backward'):
        qT_i = Q[:, i * vec_batch_size:(i * vec_batch_size) + vec_batch_size].T
        qT_low_rank_jac_mat_row = vec_jac_mul_batch(hooked_model, filtbackproj, qT_i.to(bayesianized_model.store_device), bayesianized_model).detach()
        qT_low_rank_jac_mat.append( qT_low_rank_jac_mat_row.cpu() if use_cpu else qT_low_rank_jac_mat_row )
    B = torch.cat(qT_low_rank_jac_mat)
    U, S, Vh = torch.linalg.svd(B, full_matrices=False)
    if not return_on_cpu:
        U = U.to(bayesianized_model.store_device)
        S = S.to(bayesianized_model.store_device)
        Vh = Vh.to(bayesianized_model.store_device)
    return  Q[:, :low_rank_rank_dim] @ U[:low_rank_rank_dim, :low_rank_rank_dim], S[:low_rank_rank_dim], Vh[:low_rank_rank_dim, :]