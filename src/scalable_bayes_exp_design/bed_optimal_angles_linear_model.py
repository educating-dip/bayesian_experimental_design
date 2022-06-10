from builtins import breakpoint
import os
import gc
import socket
import datetime
import io
import torch
import numpy as np
import scipy.sparse
import copy
import yaml
import tensorboardX
import tensorly as tl
tl.set_backend('pytorch')
from math import ceil
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from dataset.matrix_ray_trafo_utils import get_matrix_ray_trafo_module

def get_save_obj_callback_linear_model(obj_list):

    def save_obj_callback(all_acq_angle_inds, init_angle_inds, cur_angle_inds, acq_angle_inds, top_projs_idx, local_vars):
        
        s_images = local_vars['f_samples']
        obj = local_vars['obj']
        obj_list.append({'obj': obj, 
            'acq_angle_inds': acq_angle_inds, 
            's_var_images': s_images.T.pow(2).mean(dim=0).squeeze().cpu().numpy()
            }
        )
    
    return save_obj_callback

def generate_dist_mtx_torch(side):
    coords = torch.stack([torch.repeat_interleave(torch.arange(side), side), torch.tile(torch.arange(side), (side,))], axis=1)
    coords_exp1 = coords[:,None,:]
    coords_exp0 = coords[None,:,:]
    dist_mtx = ((coords_exp1 - coords_exp0) ** 2).sum(axis=-1) ** 0.5
    return dist_mtx
      
def RadialBasisFuncCov_torch(side, marg_var, log_lengthscale, device):
    # lengthscale = -1 / log_ar_p
    eps = 1e-5
    dist_mtx = generate_dist_mtx_torch(side).to(device=device)
    cov_mat = marg_var * (torch.exp(- dist_mtx / torch.exp(log_lengthscale)) + eps * torch.eye(side ** 2).to(device=device))
    return cov_mat

def linear_posterior_matheron(cov_obs_mat_chol, cov_image_mat, ray_trafo, ray_trafo_adj, ray_trafo_comp, noise_obs_std, mc_samples):

    dx = cov_image_mat.shape[0]
    dy = ray_trafo.shape[0]

    f_cov_chol = torch.linalg.cholesky(cov_image_mat) # dx x dx
    f_noise = torch.randn((dx, mc_samples), device=cov_obs_mat_chol.device)
    y_noise = torch.randn((dy, mc_samples), device=cov_obs_mat_chol.device)

    f_samples = f_cov_chol @ f_noise # df x mc_samples
    y_samples = ray_trafo @ f_samples + y_noise * noise_obs_std  # dy x mc_samples
    y_post_samples = torch.cholesky_solve(y_samples, cov_obs_mat_chol, upper=False) # dy x mc_samples
    y_non_obs_samples = ray_trafo_comp @ (f_samples - cov_image_mat @ (ray_trafo_adj @ y_post_samples)) # dy_complement x Nsamples
    return y_non_obs_samples, f_samples

def sampled_diag_EIG(y_samples_per_detector_per_angle, noise_obs_std):
    # y_samples_per_detector_per_angle -> (angles, deterctors, samples)
    log_var_per_detector_per_angle = torch.log( (y_samples_per_detector_per_angle ** 2).mean(axis=-1) + (noise_obs_std ** 2) ) # log variance per detector pixel per angle
    diag_EIG_per_angle = log_var_per_detector_per_angle.sum(axis=1) # sum across detector pixels
    return diag_EIG_per_angle

def sampled_EIG(y_samples_per_detector_per_angle, noise_obs_std):
    # y_samples_per_detector_per_angle -d> (angles, deterctors, samples)
    mc_samples = y_samples_per_detector_per_angle.shape[-1]
    angle_cov = torch.bmm(y_samples_per_detector_per_angle, y_samples_per_detector_per_angle.transpose(1,2)) / mc_samples
    angle_cov += (noise_obs_std ** 2) * torch.eye(angle_cov.shape[1], device=angle_cov.device)[None, :, :]
    s, EIG = torch.linalg.slogdet(angle_cov)
    assert all(s == 1)
    return EIG

def get_cov_obs_mat_chol(cov_image_mat, ray_trafo, noise_obs_std):

    cov_obs_mat = ray_trafo @ (ray_trafo @ cov_image_mat).T
    cov_obs_mat[np.diag_indices(cov_obs_mat.shape[0])] += (noise_obs_std ** 2)  # dy x dy
    cov_obs_mat_chol = torch.linalg.cholesky(cov_obs_mat)
    return cov_obs_mat_chol

def get_isotropic_image_cov_mat(var, image_flat_shape, device='cpu'):
    return var * torch.eye(image_flat_shape, device=device)

def linear_MLL(observation, cov_obs_mat_chol):

    half_fit = torch.cholesky_solve(observation[:, None], cov_obs_mat_chol, upper=False).squeeze()
    fit_term = (half_fit * observation).sum()
    logdet_term = torch.log(torch.diag(cov_obs_mat_chol)).sum() * 2
    return - 0.5 * (fit_term + logdet_term)

def optimize_MLL(observation, ray_trafo_module, lr=3e-2, n_steps=1000, patience=10, use_gp_model=False, device='cpu', parameters_init=None):
    # patience is the number of steps without improvement used for early stopping
    log_noise_obs_std = torch.nn.Parameter(-1 * torch.ones(1, device = device))
    log_prior_var = torch.nn.Parameter(torch.zeros(1, device = device))
    parameters = [{'params': log_noise_obs_std}, {'params': log_prior_var}]
    if use_gp_model:
        log_lengthscale = torch.nn.Parameter(2 * torch.ones(1, device = device))
        parameters += [{'params': log_lengthscale}]
    if parameters_init is not None:
        for ii, p in enumerate(parameters):
            p['params'].data = parameters_init[ii]
    optimizer = torch.optim.Adam(parameters, lr, weight_decay=0)

    MLL_values = []
    best_parameters = None 
    steps = trange(n_steps)
    for i in steps:
        if not use_gp_model: 
            cov_image_mat = get_isotropic_image_cov_mat(
                image_flat_shape=ray_trafo_module.matrix.shape[1],
                var=log_prior_var.exp(), 
                device=device)
        else: 
            cov_image_mat =  RadialBasisFuncCov_torch(int(np.sqrt(ray_trafo_module.matrix.shape[1])), log_prior_var.exp(), log_lengthscale, device=device)
            
            
        cov_obs_mat_chol = get_cov_obs_mat_chol(cov_image_mat, ray_trafo_module.matrix, log_noise_obs_std.exp())

        loss = - linear_MLL(observation, cov_obs_mat_chol)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        MLL_values.append(-loss.item())
        steps.set_postfix_str(f"MLL: {-loss.item(): 8.3f}")
        best_i = np.argmax(MLL_values)
        if best_i == i:
            best_parameters = copy.deepcopy(parameters)
        if (patience is not None) and (i - best_i) > patience:
            break
    return [i['params'][0].detach() for i in best_parameters], MLL_values

def _get_ray_trafo_modules(ray_trafo_mat_flat, cur_proj_inds_list, acq_proj_inds_list, filtbackproj, num_projs_per_angle, device):
    ray_trafo_module = get_matrix_ray_trafo_module(
            # reshaping of matrix rows to (len(cur_proj_inds_list), num_projs_per_angle) is row-major
            ray_trafo_mat_flat[np.concatenate(cur_proj_inds_list)],
            im_shape=filtbackproj.shape[2:], proj_shape=(len(cur_proj_inds_list), num_projs_per_angle),
            adjoint=False, sparse=scipy.sparse.isspmatrix(ray_trafo_mat_flat)).to(dtype=filtbackproj.dtype, device=device)
    ray_trafo_module_adj = get_matrix_ray_trafo_module(
            # reshaping of matrix rows to (len(cur_proj_inds_list), num_projs_per_angle) is row-major
            ray_trafo_mat_flat[np.concatenate(cur_proj_inds_list)],
            im_shape=filtbackproj.shape[2:], proj_shape=(len(cur_proj_inds_list), num_projs_per_angle),
            adjoint=True, sparse=scipy.sparse.isspmatrix(ray_trafo_mat_flat)).to(dtype=filtbackproj.dtype, device=device)
    if len(acq_proj_inds_list) > 0:
        ray_trafo_comp_module = get_matrix_ray_trafo_module(
                # reshaping of matrix rows to (len(acq_proj_inds_list), num_projs_per_angle) is row-major
                ray_trafo_mat_flat[np.concatenate(acq_proj_inds_list)],
                im_shape=filtbackproj.shape[2:], proj_shape=(len(acq_proj_inds_list), num_projs_per_angle),
                adjoint=False, sparse=scipy.sparse.isspmatrix(ray_trafo_mat_flat)).to(dtype=filtbackproj.dtype, device=device)
    else:
        ray_trafo_comp_module = None
    return ray_trafo_module, ray_trafo_module_adj, ray_trafo_comp_module

# note: observation needs to have shape (len(init_angle_inds), num_projs_per_angle)
def bed_optimal_angles_search_linear_model(
    ray_trafo_mat_flat, proj_inds_per_angle, init_angle_inds, acq_angle_inds,
    observation_full, ray_trafos_full, filtbackproj, ground_truth,
    total_num_acq_projs, acq_projs_batch_size,
    mc_samples,
    use_EIG=True,
    use_diagonal_EIG=False, 
    reconstruct_every_k_step=None,
    log_path='./',
    mll_optim_kwargs=None,
    device=None,
    hyperparam_fun=None,  # hyperparam_fun(num_acq) -> (gamma, iterations)
    callbacks=(),
    logged_plot_callbacks=None,
    use_sparse=True,
    opt_parameters=None
    ):

    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    comment = 'bayesian_experimental_design'
    logdir = os.path.join(
    log_path,
    current_time + '_' + socket.gethostname() + comment)
    writer = tensorboardX.SummaryWriter(logdir=logdir)

    if mll_optim_kwargs is None:
        mll_optim_kwargs = {}

    if logged_plot_callbacks is None:
        logged_plot_callbacks = {}

    if device is None: 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_projs_per_angle = len(proj_inds_per_angle[acq_angle_inds[0]])
    assert all(len(proj_inds_per_angle[a_ind]) == num_projs_per_angle for a_ind in init_angle_inds)
    assert all(len(proj_inds_per_angle[a_ind]) == num_projs_per_angle for a_ind in acq_angle_inds)

    all_acq_angle_inds = list(acq_angle_inds)

    cur_angle_inds = list(init_angle_inds)
    cur_proj_inds_list = [proj_inds_per_angle[a_ind] for a_ind in cur_angle_inds]

    acq_angle_inds = list(acq_angle_inds)
    acq_proj_inds_list = [proj_inds_per_angle[a_ind] for a_ind in acq_angle_inds]

    if use_sparse:
        ray_trafo_mat_flat = scipy.sparse.csr_matrix(ray_trafo_mat_flat)

    ray_trafo_module, ray_trafo_module_adj, ray_trafo_comp_module = _get_ray_trafo_modules(
            ray_trafo_mat_flat, cur_proj_inds_list, acq_proj_inds_list, filtbackproj, num_projs_per_angle, device)

    num_batches = ceil(total_num_acq_projs / acq_projs_batch_size)

    obs = observation_full.flatten()[np.concatenate(cur_proj_inds_list)].to(device)
    opt_parameters, _ = optimize_MLL(obs, ray_trafo_module, **mll_optim_kwargs, device=device, parameters_init=opt_parameters)

    noise_obs_std = opt_parameters[0].exp()
    prior_var = opt_parameters[1].exp()
    print('noise_obs_std', noise_obs_std)
    print('prior_var', prior_var)

    use_gp_model = mll_optim_kwargs['use_gp_model']
    if use_gp_model:
        gp_log_lengthscale = opt_parameters[2].exp()
        print('gp_log_lengthscale', gp_log_lengthscale)
        cov_image_mat =  RadialBasisFuncCov_torch(int(np.sqrt(ray_trafo_mat_flat.shape[-1])), prior_var, gp_log_lengthscale, device=device)
    else:
        cov_image_mat = prior_var * torch.eye(ray_trafo_mat_flat.shape[-1], device=device)

    for i in tqdm(range(num_batches), miniters=num_batches//100, desc='bed_optimal_angles_search'):
        
        cov_obs_mat_chol = get_cov_obs_mat_chol(cov_image_mat, ray_trafo_module.matrix, noise_obs_std)

        s_comp_observations, f_samples = linear_posterior_matheron(
            cov_obs_mat_chol=cov_obs_mat_chol,
            cov_image_mat=cov_image_mat, 
            ray_trafo=ray_trafo_module.matrix, 
            ray_trafo_adj=ray_trafo_module_adj.matrix, 
            ray_trafo_comp=ray_trafo_comp_module.matrix, 
            noise_obs_std=noise_obs_std, 
            mc_samples=mc_samples
        )

        s_comp_observations_reshaped = s_comp_observations.reshape(
            len(all_acq_angle_inds) + len(init_angle_inds) - len(cur_angle_inds),
            num_projs_per_angle,
            mc_samples
        ) # N_angles, N_detectors, N_Samples

        if use_EIG:
            if use_diagonal_EIG: 
                obj = sampled_diag_EIG(s_comp_observations_reshaped, noise_obs_std).cpu().numpy()
            else: 
                obj = sampled_EIG(s_comp_observations_reshaped, noise_obs_std).cpu().numpy()
        else:
            obj = torch.mean(s_comp_observations_reshaped.pow(2), dim=(1, 2)).cpu().numpy()

        top_projs_idx = np.argpartition(obj, -acq_projs_batch_size)[-acq_projs_batch_size:] # choose top k angles while avoiding sorting 
        top_k_acq_angle_inds = [acq_angle_inds[idx] for idx in top_projs_idx]
        
        for callback in callbacks:
            callback(all_acq_angle_inds, init_angle_inds, cur_angle_inds, acq_angle_inds, top_projs_idx, local_vars=locals())

        for name, plot_callback in logged_plot_callbacks.items():
            fig = plot_callback(all_acq_angle_inds, init_angle_inds, cur_angle_inds, acq_angle_inds, top_projs_idx, local_vars=locals())
            # writer.add_figure(name, fig, i)  # includes a log of margin
            with io.BytesIO() as buff:
                fig.savefig(buff, format='png', bbox_inches='tight')
                buff.seek(0)
                im = plt.imread(buff)
                im = im.transpose((2, 0, 1))
            writer.add_image(name, im, i)

        # update lists of acquired and not yet acquired projections
        cur_angle_inds += top_k_acq_angle_inds
        cur_proj_inds_list = [proj_inds_per_angle[a_ind] for a_ind in cur_angle_inds]

        _reduced_acq_inds = np.setdiff1d(np.arange(len(acq_proj_inds_list)), top_projs_idx)
        acq_angle_inds = [acq_angle_inds[idx] for idx in _reduced_acq_inds]
        acq_proj_inds_list = [proj_inds_per_angle[a_ind] for a_ind in acq_angle_inds]

        # update transforms
        ray_trafo_module, ray_trafo_module_adj, ray_trafo_comp_module = _get_ray_trafo_modules(
                ray_trafo_mat_flat, cur_proj_inds_list, acq_proj_inds_list, filtbackproj, num_projs_per_angle, device)
        
    writer.close()
    best_inds_acquired = [int(ind) for ind in cur_angle_inds if ind not in init_angle_inds]

    return best_inds_acquired, opt_parameters