import os
import gc
import socket
import datetime
import io
import torch
import numpy as np
import scipy.sparse
from copy import deepcopy
import yaml
import tensorboardX
import tensorly as tl
tl.set_backend('pytorch')
from math import ceil
from tqdm import tqdm
import matplotlib.pyplot as plt
from .sample_observations import sample_observations_shifted
from .update_cov_obs_mat import update_cov_obs_mat_no_noise
from scalable_linearised_laplace import stabilize_prior_cov_obs_mat
from deep_image_prior import normalize, DeepImagePriorReconstructor, PSNR
from scalable_bayes_exp_design.tvadam import TVAdamReconstructor
from dataset.matrix_ray_trafo_utils import get_matrix_ray_trafo_module
from .greedy_optimal_angle_selection import adjust_filtbackproj_module, _get_ray_trafo_adj_module
from priors_marglik import BayesianizeModel
from scalable_linearised_laplace import (  
    add_batch_grad_hooks, get_unet_batch_ensemble, get_fwAD_model, get_prior_cov_obs_mat, 
    optim_marginal_lik_low_rank, get_objective_prior_scale_vec
    )

def sampled_diag_EIG(y_samples_per_detector_per_angle, noise_obs_std):
    # y_samples_per_detector_per_angle -> (angles, deterctors, samples)
    log_var_per_detector_per_angle = torch.log( (y_samples_per_detector_per_angle ** 2).mean(axis=-1) + (noise_obs_std ** 2) ) # log variance per detector pixel per angle
    diag_EIG_per_angle = log_var_per_detector_per_angle.sum(axis=1) # sum across detector pixels
    return diag_EIG_per_angle

def sampled_EIG(y_samples_per_detector_per_angle, noise_obs_std):
    # y_samples_per_detector_per_angle -d> (angles, deterctors, samples)
    mc_samples = y_samples_per_detector_per_angle.shape[-1]
    angle_cov = torch.bmm(y_samples_per_detector_per_angle, y_samples_per_detector_per_angle.transpose(1,2)) / mc_samples # (angles, deterctors, deterctors)
    angle_cov += (noise_obs_std ** 2) * torch.eye(angle_cov.shape[1], device=angle_cov.device)[None, :, :]
    s, EIG = torch.linalg.slogdet(angle_cov)
    assert all(s == 1)
    return EIG

def find_optimal_proj(s_observations, log_noise_model_variance_obs, acq_projs_batch_size, criterion='EIG', return_obj=False):
    # mc_samples x 1 x num_acq x num_projs_per_angle
    if criterion == 'diagonal_EIG':
        obj = sampled_diag_EIG(s_observations.squeeze(1).moveaxis(0,-1), torch.exp(log_noise_model_variance_obs)**.5).cpu().numpy()
    elif criterion == 'EIG':
        obj = sampled_EIG(s_observations.squeeze(1).moveaxis(0,-1), torch.exp(log_noise_model_variance_obs)**.5).cpu().numpy()
    elif criterion == 'var':
        obj = torch.mean(s_observations.pow(2), dim=(0, -1)).squeeze(0).cpu().numpy()
    else:
        raise ValueError
    top_projs_idx = np.argpartition(obj, -acq_projs_batch_size)[-acq_projs_batch_size:]
    return top_projs_idx, obj if return_obj else top_projs_idx


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

def get_hyperparam_fun_from_yaml(path, data, noise_stddev):
    with open(path, 'r') as f:
        d = yaml.safe_load(f)
    d_per_angle = {n_a: d[data][n_a][noise_stddev] for n_a in d[data].keys() if noise_stddev in d[data][n_a]}
    def hyperparam_fun(num_angles):
        num_angles_provided = list(d_per_angle.keys())
        nearest_num_angles_i = min(enumerate(abs(n_a - num_angles) for n_a in num_angles_provided), key=lambda x: x[1])[0]
        nearest_num_angles = num_angles_provided[nearest_num_angles_i]
        if nearest_num_angles != num_angles:
            print('did not find hyperparameters for {:d} angles, using hyperparameters from {:d} angles instead'.format(num_angles, nearest_num_angles))
        h = d_per_angle[nearest_num_angles]
        return float(h['gamma']), int(h['iterations'])
    return hyperparam_fun

def get_save_obj_callback(obj_list):

    def save_obj_callback(all_acq_angle_inds, init_angle_inds, cur_angle_inds, acq_angle_inds, top_projs_idx, local_vars):
        if 'obj' in local_vars:
            s_images = local_vars['s_images']
            obj = local_vars.get['obj']
            obj_list.append({'obj': obj,
                'acq_angle_inds': acq_angle_inds,
                's_var_images': s_images.pow(2).mean(dim=0).squeeze().cpu().numpy()
                }
            )
        else:
            obj_list.append(None)
    
    return save_obj_callback

def plot_obj_callback(all_acq_angle_inds, init_angle_inds, cur_angle_inds, acq_angle_inds, top_projs_idx, local_vars):
    fig, ax = plt.subplots()
    top_k_acq_angle_inds = [acq_angle_inds[idx] for idx in top_projs_idx]
    for a in init_angle_inds:
        ax.axvline(a, color='gray')
    ax.plot(acq_angle_inds, local_vars['obj'], 'x', color='tab:blue')
    ax.plot(top_k_acq_angle_inds, local_vars['obj'][top_projs_idx], 'o', color='tab:red')
    ax.set_xlabel('angle')
    ax.set_ylabel('mean variance')
    return fig

def plot_angles_callback(all_acq_angle_inds, init_angle_inds, cur_angle_inds, acq_angle_inds, top_projs_idx, local_vars):
    full_angles = local_vars['ray_trafos_full']['geometry'].angles
    top_k_acq_angle_inds = [acq_angle_inds[idx] for idx in top_projs_idx]
    if (len(full_angles) % (len(cur_angle_inds) + len(top_k_acq_angle_inds)) == 0
            and (len(cur_angle_inds) + len(top_k_acq_angle_inds)) % len(init_angle_inds) == 0):
        baseline_step = len(full_angles) // (len(cur_angle_inds) + len(top_k_acq_angle_inds))
        baseline_angle_inds = np.arange(0, len(full_angles), baseline_step)
    else:
        baseline_angle_inds = None
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    # for theta in full_angles[init_angle_inds]:
    #     ax.plot([theta, theta], [0.1, 1.], color='gray')
    for theta in full_angles[acq_angle_inds]:
        ax.plot([theta, theta], [0.1, 1.], color='gray', alpha=0.025)
    for theta in full_angles[cur_angle_inds]:
        ax.plot([theta, theta], [0.1, 1.], color='tab:red', alpha=0.425)
    for theta in full_angles[top_k_acq_angle_inds]:
        ax.plot([theta, theta], [0.1, 1.], color='tab:red')
    if baseline_angle_inds is not None:
        for theta in full_angles[baseline_angle_inds]:
            ax.plot([theta, theta], [0.1, 1.], color='gray', linestyle='dotted')
    ax.set_yticks([])
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_thetagrids(full_angles[init_angle_inds]/np.pi*180.)
    ax.grid(linewidth=1.5, color='black')
    return fig

# note: observation needs to have shape (len(init_angle_inds), num_projs_per_angle)
def bed_optimal_angles_search(
    ray_trafo_mat_flat, proj_inds_per_angle, init_angle_inds, acq_angle_inds,
    observation_full, ray_trafos_full, filtbackproj, ground_truth,
    init_cov_obs_mat_no_noise, log_noise_model_variance_obs,
    hooked_model, bayesianized_model, be_model, be_modules,
    total_num_acq_projs, acq_projs_batch_size,
    mc_samples, 
    vec_batch_size,
    model_basename,
    reconstruct_every_k_step=None,
    update_network_params=False,
    update_prior_hyperparams='mrglik',
    criterion='EIG',
    use_precomputed_best_inds=None,
    log_path='./',
    device=None,
    cov_obs_mat_eps_mode='abs',
    cov_obs_mat_eps=0.1,
    cov_obs_mat_eps_min_for_auto=0.,
    cfg_net=None,
    cfg_marginal_lik=None,
    override_mrglik_iterations=100,
    hyperparam_fun=None,  # hyperparam_fun(num_acq) -> (gamma, iterations)
    callbacks=(),
    logged_plot_callbacks=None,
    return_recons=False,
    use_alternative_recon=None,
    tvadam_hyperparam_fun=None,
    cfg_tvadam=None,
    init_state_dict=None,
    tuple_scale_vec=None,
    update_scale_vec_via_refined_jac=False,
    g_prior_scale_fct=1.,
    use_sparse=True,
    cov_obs_mat_stab_by_tsvd=False,
    cov_obs_mat_stab_by_tsvd_reduce_dim_fraction=0.1,
    ):

    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    comment = 'bayesian_experimental_design'
    logdir = os.path.join(
    log_path,
    current_time + '_' + socket.gethostname() + comment)
    writer = tensorboardX.SummaryWriter(logdir=logdir)

    if logged_plot_callbacks is None:
        logged_plot_callbacks = {}

    if use_precomputed_best_inds is not None:
        use_precomputed_best_inds = list(use_precomputed_best_inds)

    if device is None: 
        device = bayesianized_model.store_device

    fbp_op = adjust_filtbackproj_module( 
        ray_trafos_full['space'], 
        ray_trafos_full['geometry']
        )

    num_projs_per_angle = len(proj_inds_per_angle[acq_angle_inds[0]])
    assert all(len(proj_inds_per_angle[a_ind]) == num_projs_per_angle for a_ind in init_angle_inds)
    assert all(len(proj_inds_per_angle[a_ind]) == num_projs_per_angle for a_ind in acq_angle_inds)

    cov_obs_mat_no_noise = init_cov_obs_mat_no_noise

    all_acq_angle_inds = list(acq_angle_inds)

    cur_angle_inds = list(init_angle_inds)
    cur_proj_inds_list = [proj_inds_per_angle[a_ind] for a_ind in cur_angle_inds]

    acq_angle_inds = list(acq_angle_inds)
    acq_proj_inds_list = [proj_inds_per_angle[a_ind] for a_ind in acq_angle_inds]

    if use_sparse:
        ray_trafo_mat_flat = scipy.sparse.csr_matrix(ray_trafo_mat_flat)

    ray_trafo_module, ray_trafo_module_adj, ray_trafo_comp_module = _get_ray_trafo_modules(
            ray_trafo_mat_flat, cur_proj_inds_list, acq_proj_inds_list, filtbackproj, num_projs_per_angle, device)

    cfg_net = deepcopy(cfg_net)

    if tuple_scale_vec is not None and None in tuple_scale_vec: 
        tuple_scale_vec = None 

    if init_state_dict is None:
        random_init_reconstructor = DeepImagePriorReconstructor(ray_trafo_module, filtbackproj.shape[2:], cfg=cfg_net)
        init_state_dict = random_init_reconstructor.model.state_dict()

    dtype = ray_trafo_module.matrix.dtype

    num_params_under_priors = np.sum(bayesianized_model.ref_num_params_per_modules_under_gp_priors + 
                    bayesianized_model.ref_num_params_per_modules_under_normal_priors)

    bayesianized_model_state_dict = bayesianized_model.state_dict()
    recons = []
    num_batches = ceil(total_num_acq_projs / acq_projs_batch_size)
    for i in tqdm(range(num_batches), miniters=num_batches//100, desc='bed_optimal_angles_search'):
        if not use_precomputed_best_inds:
            if not cov_obs_mat_stab_by_tsvd:
                cov_obs_mat_eps_abs = stabilize_prior_cov_obs_mat(
                        cov_obs_mat_no_noise + torch.exp(log_noise_model_variance_obs) * torch.eye(
                                cov_obs_mat_no_noise.shape[0], device=cov_obs_mat_no_noise.device),
                        eps_mode=cov_obs_mat_eps_mode, eps=cov_obs_mat_eps, eps_min_for_auto=cov_obs_mat_eps_min_for_auto)
            else:
                U, S, Vh = tl.truncated_svd(cov_obs_mat_no_noise, n_eigenvecs=ceil(cov_obs_mat_no_noise.shape[0]*(1.-cov_obs_mat_stab_by_tsvd_reduce_dim_fraction)))

            s_observations, s_images = sample_observations_shifted(ray_trafo_module, ray_trafo_module_adj, ray_trafo_comp_module,
                filtbackproj,
                torch.linalg.cholesky(
                    cov_obs_mat_no_noise + (torch.exp(log_noise_model_variance_obs) + cov_obs_mat_eps_abs) * torch.eye(
                            cov_obs_mat_no_noise.shape[0], device=cov_obs_mat_no_noise.device)
                    ) if not cov_obs_mat_stab_by_tsvd else (U, S, Vh),
                hooked_model, bayesianized_model, be_model, be_modules,
                log_noise_model_variance_obs,
                mc_samples,
                vec_batch_size,
                tuple_scale_vec=tuple_scale_vec,
                device=device
                )
            top_projs_idx, obj = find_optimal_proj(s_observations, log_noise_model_variance_obs, acq_projs_batch_size, criterion=criterion, return_obj=True)
        else:
            top_projs_idx = [
                    acq_angle_inds.index(ind)  # absolute to relative in acq_angle_inds
                    for ind in use_precomputed_best_inds[i*acq_projs_batch_size:(i+1)*acq_projs_batch_size]]

        top_k_acq_angle_inds = [acq_angle_inds[idx] for idx in top_projs_idx]
        top_k_acq_proj_inds_list = [proj_inds_per_angle[a_ind] for a_ind in top_k_acq_angle_inds]
        ray_trafo_top_k_module = get_matrix_ray_trafo_module(
                # reshaping of matrix rows to (acq_projs_batch_size, num_projs_per_angle) is row-major
                ray_trafo_mat_flat[np.concatenate(top_k_acq_proj_inds_list)],
                im_shape=filtbackproj.shape[2:], proj_shape=(acq_projs_batch_size, num_projs_per_angle),
                adjoint=False, sparse=scipy.sparse.isspmatrix(ray_trafo_mat_flat)).to(dtype=filtbackproj.dtype, device=device)

        if not use_precomputed_best_inds:
            cov_obs_mat_no_noise = update_cov_obs_mat_no_noise(
                cov_obs_mat_no_noise, ray_trafo_module, ray_trafo_top_k_module,
                filtbackproj, hooked_model, bayesianized_model, be_model, be_modules, vec_batch_size,
                tuple_scale_vec=tuple_scale_vec
                )

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

        if reconstruct_every_k_step is not None and (i+1) % reconstruct_every_k_step == 0:
            if not use_alternative_recon:
                if hyperparam_fun is not None:
                    cfg_net.optim.gamma, cfg_net.optim.iterations = hyperparam_fun(len(cur_proj_inds_list))
                refine_reconstructor = DeepImagePriorReconstructor(deepcopy(ray_trafo_module).float(), filtbackproj.shape[2:], cfg=cfg_net)

                refine_reconstructor.model.float()

                obs = observation_full.flatten()[np.concatenate(cur_proj_inds_list)].view(1, 1, *ray_trafo_module.out_shape)
                recon, _ = refine_reconstructor.reconstruct(obs.float(), fbp=filtbackproj.float(), ground_truth=ground_truth.float(),
                        use_init_model=False, init_state_dict=init_state_dict)
                torch.save(refine_reconstructor.model.state_dict(),
                        './{}_acq_{}.pt'.format(model_basename, i+1))
                recons.append(recon)

                refine_reconstructor.model.to(dtype=dtype)

                if ( update_network_params or tuple_scale_vec is not None ) and not use_precomputed_best_inds:
                    class ray_trafo: 
                        def __init__(self, shape):
                            class range: 
                                def __init__(self, shape):
                                    self.shape = shape
                            self.range = range(shape=shape)

                    ray_trafos = {
                        'ray_trafo': ray_trafo(tuple(obs.shape[2:])), 
                        'ray_trafo_module': ray_trafo_module,
                        'ray_trafo_module_adj': ray_trafo_module_adj
                    }
                        
                    if tuple_scale_vec is None: 
                        bayesianized_model = BayesianizeModel(
                            refine_reconstructor,
                            **{'lengthscale_init': cfg_marginal_lik.mrglik.priors.lengthscale_init, 'variance_init': cfg_marginal_lik.mrglik.priors.variance_init}, 
                            include_normal_priors=cfg_marginal_lik.mrglik.priors.include_normal_priors,
                            exclude_gp_priors_list=cfg_marginal_lik.density.exclude_gp_priors_list, 
                            exclude_normal_priors_list=cfg_marginal_lik.density.exclude_normal_priors_list
                        )
                        bayesianized_model.load_state_dict(bayesianized_model_state_dict)

                        hooked_model = refine_reconstructor.model
                        modules = bayesianized_model.get_all_modules_under_prior()
                        be_model, be_module_mapping = get_unet_batch_ensemble(refine_reconstructor.model, vec_batch_size, return_module_mapping=True)
                        be_modules = [be_module_mapping[m] for m in modules]
                        be_model, fwAD_be_module_mapping = get_fwAD_model(be_model, return_module_mapping=True, share_parameters=True)
                        be_modules = [fwAD_be_module_mapping[m] for m in be_modules]
                        #TODO: reduced model

                        if 'batch_grad_hooks' not in hooked_model.__dict__:
                            add_batch_grad_hooks(hooked_model, modules)

                        if update_prior_hyperparams == 'mrglik':
                            cfg_marginal_lik.mrglik.optim.iterations = override_mrglik_iterations 
                            proj_recon = ray_trafos['ray_trafo_module'](
                                torch.from_numpy(recon[None, None]).to(dtype=dtype, device=device)
                                )
                            log_noise_model_variance_obs = optim_marginal_lik_low_rank(
                            cfg_marginal_lik,
                            obs,
                            (torch.from_numpy(recon[None, None]).to(dtype=dtype, device=device), proj_recon),
                            ray_trafos, filtbackproj, bayesianized_model, hooked_model, be_model, be_modules, 
                            linearized_weights=None, # TODO linearized weights 
                            comment = '_recon_num_' + str(i),
                            )
                            bayesianized_model_state_dict = bayesianized_model.state_dict()
                        else: 
                            pass
                    else: 
                        empirical_g_coeff = tuple_scale_vec[1]
                        if update_scale_vec_via_refined_jac:
                            bayesianized_model = BayesianizeModel(
                                refine_reconstructor,
                                **{'lengthscale_init': cfg_marginal_lik.mrglik.priors.lengthscale_init, 'variance_init': cfg_marginal_lik.mrglik.priors.variance_init}, 
                                include_normal_priors=cfg_marginal_lik.mrglik.priors.include_normal_priors,
                                exclude_gp_priors_list=cfg_marginal_lik.density.exclude_gp_priors_list, 
                                exclude_normal_priors_list=cfg_marginal_lik.density.exclude_normal_priors_list
                            )
                            bayesianized_model.load_state_dict(bayesianized_model_state_dict)

                            empirical_g_coeff = g_prior_scale_fct * (
                                ((obs**2) -  torch.exp(log_noise_model_variance_obs).cpu()).sum() / (
                                    num_params_under_priors * obs.numel() * torch.exp(log_noise_model_variance_obs).cpu())
                                ).to(bayesianized_model.store_device)

                            hooked_model = refine_reconstructor.model
                            modules = bayesianized_model.get_all_modules_under_prior()
                            
                            be_model, be_module_mapping = get_unet_batch_ensemble(refine_reconstructor.model, vec_batch_size, return_module_mapping=True)
                            be_modules = [be_module_mapping[m] for m in modules]
                            be_model, fwAD_be_module_mapping = get_fwAD_model(be_model, return_module_mapping=True, share_parameters=True)
                            be_modules = [fwAD_be_module_mapping[m] for m in be_modules]

                            if 'batch_grad_hooks' not in hooked_model.__dict__:
                                add_batch_grad_hooks(hooked_model, modules)

                        scale_vec = get_objective_prior_scale_vec(
                            ray_trafos,
                            filtbackproj,
                            bayesianized_model, hooked_model, 
                            log_noise_model_variance_obs,
                            vec_batch_size
                            )

                        tuple_scale_vec = (scale_vec, empirical_g_coeff)
                        
                    # compute cov_obs_mat via closure 
                    cov_obs_mat_no_noise = get_prior_cov_obs_mat(
                        ray_trafos, filtbackproj, 
                        bayesianized_model, hooked_model, be_model, be_modules, 
                        log_noise_model_variance_obs,
                        vec_batch_size, 
                        use_fwAD_for_jvp=True, 
                        add_noise_model_variance_obs=False,
                        tuple_scale_vec=tuple_scale_vec
                        ) #TODO approx jacs
                                                    
            elif use_alternative_recon == 'fbp':
                obs = observation_full.flatten()[np.concatenate(cur_proj_inds_list)].view(1, 1, *ray_trafo_module.out_shape)
                fbp = fbp_op(obs, _get_ray_trafo_adj_module(ray_trafo_mat_flat, cur_proj_inds_list, filtbackproj.shape[2:], num_projs_per_angle, dtype=observation_full.dtype))
                recon = fbp[0, 0].numpy()
                recons.append(recon)
            elif use_alternative_recon == 'tvadam':
                obs = observation_full.flatten()[np.concatenate(cur_proj_inds_list)].view(1, 1, *ray_trafo_module.out_shape)
                if tvadam_hyperparam_fun is not None:
                    cfg_tvadam.gamma, cfg_tvadam.iterations = tvadam_hyperparam_fun(len(cur_proj_inds_list))
                tvadam_reconstructor = TVAdamReconstructor(deepcopy(ray_trafo_module).float(), cfg=cfg_tvadam)
                recon = tvadam_reconstructor.reconstruct(
                        obs.float(), fbp=filtbackproj.float().to(tvadam_reconstructor.device), ground_truth=ground_truth.float().to(tvadam_reconstructor.device),
                        log=True)
                recons.append(recon)
            else:
                raise ValueError('Unknown alternative reconstruction method "{}"'.format(use_alternative_recon))
            writer.add_image('reco', normalize(recon[None]), i)
            writer.add_image('abs(reco-gt)', normalize(np.abs(recon[None] - ground_truth[0, 0].cpu().numpy())), i)

            print('\nPSNR with {:d} acquisitions: {}'.format(len(cur_proj_inds_list), PSNR(recon, ground_truth[0, 0].cpu().numpy()), '\n'))
    
    writer.add_image('gt', normalize(ground_truth[0].cpu().numpy()), i)
    writer.close()
    best_inds_acquired = [int(ind) for ind in cur_angle_inds if ind not in init_angle_inds]

    del cov_obs_mat_no_noise, bayesianized_model
    gc.collect(); torch.cuda.empty_cache()

    return best_inds_acquired, recons if return_recons else best_inds_acquired


# note: observation needs to have shape (len(init_angle_inds), num_projs_per_angle)
def bed_eqdist_angles_baseline(
    ray_trafo_mat_flat, proj_inds_per_angle, init_angle_inds, acq_angle_inds,
    observation_full, ray_trafos_full, filtbackproj, ground_truth,
    init_cov_obs_mat_no_noise, log_noise_model_variance_obs,
    hooked_model, bayesianized_model, be_model, be_modules,
    total_num_acq_projs, acq_projs_batch_size,
    mc_samples, 
    vec_batch_size,
    model_basename,
    reconstruct_every_k_step=None,
    log_path='./',
    device=None,
    cfg_net=None,
    hyperparam_fun=None,  # hyperparam_fun(num_acq) -> (gamma, iterations)
    callbacks=(),
    logged_plot_callbacks=None,
    use_alternative_recon=None,
    tvadam_hyperparam_fun=None,
    cfg_tvadam=None,
    init_state_dict=None,
    ):

    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    comment = 'bayesian_experimental_design_baseline'
    logdir = os.path.join(
    log_path,
    current_time + '_' + socket.gethostname() + comment)
    writer = tensorboardX.SummaryWriter(logdir=logdir)

    if logged_plot_callbacks is None:
        logged_plot_callbacks = {}

    if device is None: 
        device = bayesianized_model.store_device

    fbp_op = adjust_filtbackproj_module(
            ray_trafos_full['space'], 
            ray_trafos_full['geometry']
        )

    num_projs_per_angle = len(proj_inds_per_angle[acq_angle_inds[0]])
    assert all(len(proj_inds_per_angle[a_ind]) == num_projs_per_angle for a_ind in init_angle_inds)
    assert all(len(proj_inds_per_angle[a_ind]) == num_projs_per_angle for a_ind in acq_angle_inds)

    cov_obs_mat_no_noise = init_cov_obs_mat_no_noise

    all_acq_angle_inds = list(acq_angle_inds)

    init_angle_inds = list(init_angle_inds)
    init_proj_inds_list = [proj_inds_per_angle[a_ind] for a_ind in init_angle_inds]

    ray_trafo_mat_flat = scipy.sparse.csr_matrix(ray_trafo_mat_flat)

    ray_trafo_module, _, _ = _get_ray_trafo_modules(
            ray_trafo_mat_flat, init_proj_inds_list, [], filtbackproj, num_projs_per_angle, device)

    cfg_net = deepcopy(cfg_net)

    # random_init_reconstructor = DeepImagePriorReconstructor(ray_trafo_module, filtbackproj.shape[2:], cfg=cfg_net)
    # init_state_dict = random_init_reconstructor.model.state_dict()

    dtype = ray_trafo_module.matrix.dtype

    recons = []
    num_batches = ceil(total_num_acq_projs / acq_projs_batch_size)
    for i in tqdm(range(num_batches), miniters=num_batches//100, desc='bed_eqdist_angles_baseline'):

        num_full_angles = len(ray_trafos_full['geometry'].angles)
        num_add_acq = (i + 1) * acq_projs_batch_size
        num_cur = len(init_angle_inds) + num_add_acq
        if (num_full_angles % num_cur == 0 and num_cur % len(init_angle_inds) == 0):
            baseline_step = num_full_angles // num_cur

            new_cur_angle_inds = np.arange(0, num_full_angles, baseline_step)
            cur_proj_inds_list = [proj_inds_per_angle[a_ind] for a_ind in new_cur_angle_inds]

            add_acq_angle_inds = np.setdiff1d(new_cur_angle_inds, init_angle_inds).tolist()
            assert len(add_acq_angle_inds) == len(new_cur_angle_inds) - len(init_angle_inds)

            for callback in callbacks:
                callback(all_acq_angle_inds, init_angle_inds, init_angle_inds, add_acq_angle_inds, np.arange(len(add_acq_angle_inds)), local_vars=locals())

            for name, plot_callback in logged_plot_callbacks.items():
                fig = plot_callback(all_acq_angle_inds, init_angle_inds, init_angle_inds, add_acq_angle_inds, np.arange(len(add_acq_angle_inds)), local_vars=locals())
                # writer.add_figure(name, fig, i)  # includes a log of margin
                with io.BytesIO() as buff:
                    fig.savefig(buff, format='png', bbox_inches='tight')
                    buff.seek(0)
                    im = plt.imread(buff)
                    im = im.transpose((2, 0, 1))
                writer.add_image(name, im, i)

            # update transform
            ray_trafo_module, _, _ = _get_ray_trafo_modules(
                    ray_trafo_mat_flat, cur_proj_inds_list, [], filtbackproj, num_projs_per_angle, device)
        else:
            new_cur_angle_inds = None

        if reconstruct_every_k_step is not None and (i+1) % reconstruct_every_k_step == 0:
            if new_cur_angle_inds is not None:
                if not use_alternative_recon: 
                    if hyperparam_fun is not None:
                        cfg_net.optim.gamma, cfg_net.optim.iterations = hyperparam_fun(len(cur_proj_inds_list))
                    refine_reconstructor = DeepImagePriorReconstructor(deepcopy(ray_trafo_module).float(), filtbackproj.shape[2:], cfg=cfg_net)

                    refine_reconstructor.model.float()

                    obs = observation_full.flatten()[np.concatenate(cur_proj_inds_list)].view(1, 1, *ray_trafo_module.out_shape)
                    recon, _ = refine_reconstructor.reconstruct(obs.float(), fbp=filtbackproj.float(), ground_truth=ground_truth.float(),
                            use_init_model=False, init_state_dict=init_state_dict)
                    torch.save(refine_reconstructor.model.state_dict(),
                            './{}_acq_{}.pt'.format(model_basename, i+1))
                    recons.append(recon)

                    refine_reconstructor.model.to(dtype=dtype)

                elif use_alternative_recon == 'fbp':
                    obs = observation_full.flatten()[np.concatenate(cur_proj_inds_list)].view(1, 1, *ray_trafo_module.out_shape)
                    fbp = fbp_op(obs, _get_ray_trafo_adj_module(ray_trafo_mat_flat, cur_proj_inds_list, filtbackproj.shape[2:], num_projs_per_angle, dtype=observation_full.dtype))
                    recon = fbp[0, 0].numpy()
                    recons.append(recon)
                elif use_alternative_recon == 'tvadam':
                    obs = observation_full.flatten()[np.concatenate(cur_proj_inds_list)].view(1, 1, *ray_trafo_module.out_shape)
                    if tvadam_hyperparam_fun is not None:
                        cfg_tvadam.gamma, cfg_tvadam.iterations = tvadam_hyperparam_fun(len(cur_proj_inds_list))
                    tvadam_reconstructor = TVAdamReconstructor(deepcopy(ray_trafo_module).float(), cfg=cfg_tvadam)
                    recon = tvadam_reconstructor.reconstruct(
                            obs.float(), fbp=filtbackproj.float().to(tvadam_reconstructor.device), ground_truth=ground_truth.float().to(tvadam_reconstructor.device),
                            log=True)
                    recons.append(recon)
                else:
                    raise ValueError('Unknown alternative reconstruction method "{}"'.format(use_alternative_recon))
                writer.add_image('reco', normalize(recon[None]), i)
                writer.add_image('abs(reco-gt)', normalize(np.abs(recon[None] - ground_truth[0, 0].cpu().numpy())), i)
                print('\nPSNR with {:d} acquisitions: {}'.format(len(cur_proj_inds_list), PSNR(recon, ground_truth[0, 0].cpu().numpy())))
            else:
                print('\ncould not find baseline angle inds with {:d} acquisitions (init={:d}, full={:d})'.format(num_cur, len(init_angle_inds), num_full_angles))
                recons.append(None)
    writer.add_image('gt', normalize(ground_truth[0].cpu().numpy()), 0)

    writer.close()
    return recons
