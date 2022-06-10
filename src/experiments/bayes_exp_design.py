from gc import callbacks
import os
import pprint
from itertools import islice
from math import ceil
import numpy as np
import hydra
from omegaconf import OmegaConf, DictConfig
from dataset.utils import (
        get_standard_ray_trafos,
        load_testset_MNIST_dataset, load_testset_KMNIST_dataset,
        load_testset_rectangles_dataset, load_testset_walnut_patches_dataset, 
        load_testset_walnut)
from dataset.mnist import simulate
import torch
import scipy
from hydra.utils import get_original_cwd
from deep_image_prior import DeepImagePriorReconstructor
from deep_image_prior.utils import PSNR, SSIM
from priors_marglik import BayesianizeModel
import matplotlib.pyplot as plt
from scalable_linearised_laplace import (
        add_batch_grad_hooks, get_unet_batch_ensemble, get_fwAD_model, 
        clamp_params,  get_reduced_model, get_inactive_and_leaf_modules_unet, get_prior_cov_obs_mat, get_objective_prior_scale_vec
        )
from scalable_bayes_exp_design import bed_optimal_angles_search, bed_eqdist_angles_baseline, plot_angles_callback, plot_obj_callback, get_hyperparam_fun_from_yaml, get_save_obj_callback

@hydra.main(config_path='../cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    if cfg.use_double:
        torch.set_default_tensor_type(torch.DoubleTensor)

    ray_trafos = get_standard_ray_trafos(cfg, return_torch_module=True, return_op_mat=True)
    ray_trafos_full = get_standard_ray_trafos(cfg, return_torch_module=True, return_op_mat=True, override_angular_sub_sampling=1)

    ray_trafo = {'ray_trafo_module': ray_trafos['ray_trafo_module'],
                 'reco_space': ray_trafos['space'],
                 }

    # data: observation, filtbackproj, example_image
    if cfg.name == 'mnist':
        loader = load_testset_MNIST_dataset()
    elif cfg.name == 'kmnist':
        loader = load_testset_KMNIST_dataset()
    elif cfg.name == 'rectangles':
        loader = load_testset_rectangles_dataset(cfg)
    elif cfg.name == 'walnut_patches':
        loader = load_testset_walnut_patches_dataset(cfg)
    elif cfg.name == 'walnut':
        loader = load_testset_walnut(cfg)
    else:
        raise NotImplementedError

    assert cfg.density.compute_single_predictive_cov_block.load_path is not None, "no previous run path specified (density.compute_single_predictive_cov_block.load_path)"
    # assert cfg.density.compute_single_predictive_cov_block.block_idx is not None, "no block index specified (density.compute_single_predictive_cov_block.block_idx)"

    load_path = cfg.density.compute_single_predictive_cov_block.load_path
    load_cfg = OmegaConf.load(os.path.join(load_path, '.hydra', 'config.yaml'))

    for i, data_sample in enumerate(islice(loader, cfg.num_images)):
        if i < cfg.get('skip_first_images', 0):
            continue

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed + i)  # for reproducible noise in simulate

        if cfg.name in ['mnist', 'kmnist', 'rectangles', 'walnut_patches']:
            example_image = data_sample[0] if cfg.name in ['mnist', 'kmnist'] else data_sample
            non_normalized_example_image = example_image
            ray_trafos['ray_trafo_module'].to(example_image.device)
            ray_trafos['ray_trafo_module_adj'].to(example_image.device)
            if cfg.use_double:
                ray_trafos['ray_trafo_module'].to(torch.float64)
                ray_trafos['ray_trafo_module_adj'].to(torch.float64)
            observation, filtbackproj, example_image = simulate(
                example_image.double() if cfg.use_double else example_image, 
                ray_trafos,
                cfg.noise_specs
                )
            sample_dict = torch.load(os.path.join(load_path, 'sample_{}.pt'.format(i)), map_location=example_image.device)
            assert torch.allclose(sample_dict['filtbackproj'], filtbackproj, atol=1e-6)
            filtbackproj = sample_dict['filtbackproj']
            observation = sample_dict['observation']
            example_image = sample_dict['ground_truth']
        elif cfg.name == 'walnut':
            observation, filtbackproj, example_image = data_sample
        else:
            raise NotImplementedError

        if cfg.use_double:
            observation = observation.double()
            filtbackproj = filtbackproj.double()
            example_image = example_image.double()
            non_normalized_example_image = non_normalized_example_image.double()

        reconstructor = DeepImagePriorReconstructor(**ray_trafo, cfg=cfg.net)

        if cfg.name in ['mnist', 'kmnist', 'rectangles', 'walnut_patches']:
            # model from previous run
            if cfg.load_dip_models_from_path is not None: 
                assert cfg.load_dip_models_from_path.rstrip('/').split('/')[-1] == load_cfg.load_dip_models_from_path.rstrip('/').split('/')[-1]
                dip_load_path = cfg.load_dip_models_from_path
            else: 
                dip_load_path = load_path if load_cfg.load_dip_models_from_path is None else load_cfg.load_dip_models_from_path
            path = os.path.join(dip_load_path, 'dip_model_{}.pt'.format(i))
        elif cfg.name == 'walnut':
            # fine-tuned model
            path = os.path.join(get_original_cwd(), reconstructor.cfg.finetuned_params_path 
                if reconstructor.cfg.finetuned_params_path.endswith('.pt') else reconstructor.cfg.finetuned_params_path + '.pt')
        else:
            raise NotImplementedError

        reconstructor.model.load_state_dict(torch.load(path, map_location=reconstructor.device))

        with torch.no_grad():
            reconstructor.model.eval()
            recon, _ = reconstructor.model.forward(filtbackproj.to(reconstructor.device))
        recon = recon[0, 0].cpu().numpy()

        print('DIP reconstruction of sample {:d}'.format(i))
        print('PSNR:', PSNR(recon, example_image[0, 0].cpu().numpy()))
        print('SSIM:', SSIM(recon, example_image[0, 0].cpu().numpy()))

        bayesianized_model = BayesianizeModel(reconstructor, **{'lengthscale_init': cfg.mrglik.priors.lengthscale_init ,
            'variance_init': cfg.mrglik.priors.variance_init}, include_normal_priors=cfg.mrglik.priors.include_normal_priors,
            exclude_gp_priors_list=cfg.density.exclude_gp_priors_list, exclude_normal_priors_list=cfg.density.exclude_normal_priors_list)

        recon = torch.from_numpy(recon[None, None])

        model = reconstructor.model
        modules = bayesianized_model.get_all_modules_under_prior()
        be_model, be_module_mapping = get_unet_batch_ensemble(reconstructor.model, cfg.mrglik.impl.vec_batch_size, return_module_mapping=True)
        be_modules = [be_module_mapping[m] for m in modules]
        fwAD_be_model, fwAD_be_module_mapping = get_fwAD_model(be_model, return_module_mapping=True, share_parameters=True)
        fwAD_be_modules = [fwAD_be_module_mapping[m] for m in be_modules]

        if cfg.density.impl.reduce_model:
            inactive_modules, leaf_modules = get_inactive_and_leaf_modules_unet(model, keep_modules=modules)
            reduced_model, reduced_module_mapping = get_reduced_model(
                model, filtbackproj.to(reconstructor.device),
                replace_inactive=inactive_modules, replace_leaf=leaf_modules, return_module_mapping=True, share_parameters=True)
            modules = [reduced_module_mapping[m] for m in modules]
            assert all(m in reduced_model.modules() for m in modules), 'some module(s) in the reduced set of modules under prior cannot be found in the reduced unet model; usually this indicates that get_inactive_and_leaf_modules_unet() was called with a too small keep_num_blocks'
            model = reduced_model

            fwAD_be_inactive_modules, fwAD_be_leaf_modules = get_inactive_and_leaf_modules_unet(fwAD_be_model, keep_modules=fwAD_be_modules)
            reduced_fwAD_be_model, reduced_fwAD_be_module_mapping = get_reduced_model(
                fwAD_be_model, torch.broadcast_to(filtbackproj.to(reconstructor.device), (cfg.mrglik.impl.vec_batch_size,) + filtbackproj.shape),
                replace_inactive=fwAD_be_inactive_modules, replace_leaf=fwAD_be_leaf_modules, return_module_mapping=True, share_parameters=True)
            fwAD_be_modules = [reduced_fwAD_be_module_mapping[m] for m in fwAD_be_modules]
            assert all(m in reduced_fwAD_be_model.modules() for m in fwAD_be_modules), 'some module(s) in the reduced set of modules under prior cannot be found in the reduced unet model; usually this indicates that get_inactive_and_leaf_modules_unet() was called with a too small keep_num_blocks'
            fwAD_be_model = reduced_fwAD_be_model


        if 'batch_grad_hooks' not in model.__dict__:
            add_batch_grad_hooks(model, modules)

        ray_trafos['ray_trafo_module'].to(bayesianized_model.store_device)
        ray_trafos['ray_trafo_module_adj'].to(bayesianized_model.store_device)
        if cfg.use_double:
            ray_trafos['ray_trafo_module'].to(torch.float64)
            ray_trafos['ray_trafo_module_adj'].to(torch.float64)

        load_iter = cfg.density.compute_single_predictive_cov_block.get('load_mrglik_opt_iter', None)
        missing_keys, _ = bayesianized_model.load_state_dict(torch.load(os.path.join(
                load_path, 'bayesianized_model_{}.pt'.format(i) if load_iter is None else 'bayesianized_model_mrglik_opt_recon_num_{}_iter_{}.pt'.format(i, load_iter)),
                map_location=reconstructor.device), strict=False)
        assert not missing_keys
        log_noise_model_variance_obs = torch.load(os.path.join(
                load_path, 'log_noise_model_variance_obs_{}.pt'.format(i) if load_iter is None else 'log_noise_model_variance_obs_mrglik_opt_recon_num_{}_iter_{}.pt'.format(i, load_iter)),
                map_location=reconstructor.device)['log_noise_model_variance_obs']

        if cfg.mrglik.priors.clamp_variances:  # this only has an effect if clamping was turned off during optimization; if post-hoc clamping, we expect the user to load a cov_obs_mat that was computed with clamping, too
            clamp_params(bayesianized_model.gp_log_variances, min=cfg.mrglik.priors.clamp_variances_min_log)
            clamp_params(bayesianized_model.normal_log_variances, min=cfg.mrglik.priors.clamp_variances_min_log)

        cov_obs_mat_no_noise = None
        if cfg.density.estimate_density_from_samples.load_cov_obs_mat:  # user is responsible for checking that the cov_obs_mat is meaningful, then can activate this option
            cov_obs_mat_load_path = cfg.density.compute_single_predictive_cov_block.get('cov_obs_mat_load_path', None)
            if cov_obs_mat_load_path is None:
                cov_obs_mat_load_path = load_path
            try:
                cov_obs_mat_no_noise = torch.load(os.path.join(cov_obs_mat_load_path, 'cov_obs_mat_{}.pt'.format(i)), map_location=reconstructor.device)['cov_obs_mat'].detach()
                cov_obs_mat_no_noise = cov_obs_mat_no_noise.to(torch.float64 if cfg.use_double else torch.float32)
                cov_obs_mat_no_noise[np.diag_indices(cov_obs_mat_no_noise.shape[0])] -= log_noise_model_variance_obs.exp()
                print('loaded cov_obs_mat')
            except FileNotFoundError:
                print('cov_obs_mat file not found')

        if cov_obs_mat_no_noise is not None:
            # note about potentially different prior selection
            cov_obs_mat_load_cfg = OmegaConf.load(os.path.join(cov_obs_mat_load_path, '.hydra', 'config.yaml'))
            if not (sorted(cfg.density.exclude_gp_priors_list) == sorted(cov_obs_mat_load_cfg.density.get('exclude_gp_priors_list', [])) and
                    sorted(cfg.density.exclude_normal_priors_list) == sorted(cov_obs_mat_load_cfg.density.get('exclude_normal_priors_list', []))):
                print('note: prior selection seems to differ for the loaded cov_obs_mat')
        else:
            print('assembling cov_obs_mat')

            if not cfg.bed.use_objective_prior:
                scale_vec, empirical_g_coeff = None, None 
                # compute cov_obs_mat via closure
                cov_obs_mat_no_noise = get_prior_cov_obs_mat(
                    ray_trafos,
                    filtbackproj.to(reconstructor.device),
                    bayesianized_model,
                    model, fwAD_be_model, fwAD_be_modules,
                    log_noise_model_variance_obs,
                    cfg.mrglik.impl.vec_batch_size,
                    use_fwAD_for_jvp=cfg.mrglik.impl.use_fwAD_for_jvp,
                    add_noise_model_variance_obs=False
                    )
            else:
                scale_vec = get_objective_prior_scale_vec(
                    ray_trafos,
                    filtbackproj.to(reconstructor.device),
                    bayesianized_model, model, 
                    log_noise_model_variance_obs,
                    cfg.mrglik.impl.vec_batch_size
                    )
            
                num_params_under_priors = np.sum(bayesianized_model.ref_num_params_per_modules_under_gp_priors + 
                    bayesianized_model.ref_num_params_per_modules_under_normal_priors
                    )
                
                # empirical g-prior coefficient 
                empirical_g_coeff = cfg.bed.g_prior_scale_fct * (
                    ((observation**2) -  torch.exp(log_noise_model_variance_obs).cpu()).sum() / (
                        num_params_under_priors * observation.numel() * torch.exp(log_noise_model_variance_obs).cpu())
                    ).to(bayesianized_model.store_device)

                cov_obs_mat_no_noise = get_prior_cov_obs_mat(
                    ray_trafos, 
                    filtbackproj.to(reconstructor.device),
                    bayesianized_model,
                    model, fwAD_be_model, fwAD_be_modules,
                    log_noise_model_variance_obs,
                    cfg.mrglik.impl.vec_batch_size,
                    use_fwAD_for_jvp=cfg.mrglik.impl.use_fwAD_for_jvp,
                    add_noise_model_variance_obs=False,
                    tuple_scale_vec=(
                        scale_vec,
                        empirical_g_coeff
                        )
                )
        
        cov_obs_mat_no_noise = 0.5 * (cov_obs_mat_no_noise + cov_obs_mat_no_noise.T)

        if cfg.density.estimate_density_from_samples.save_cov_obs_mat:
            torch.save({'cov_obs_mat_no_noise': cov_obs_mat_no_noise}, './cov_obs_mat_no_noise_{}.pt'.format(i))

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed + i)  # for reproducible noise in simulate

        # simulate full obs
        if cfg.name in ['mnist', 'kmnist', 'rectangles', 'walnut_patches']:
            ray_trafos_full['ray_trafo_module'].to(example_image.device)
            ray_trafos_full['ray_trafo_module_adj'].to(example_image.device)
            if cfg.use_double:
                ray_trafos_full['ray_trafo_module'].to(torch.float64)
                ray_trafos_full['ray_trafo_module_adj'].to(torch.float64)
            observation_full, filtbackproj_full, _ = simulate(
                non_normalized_example_image.double() if cfg.use_double else non_normalized_example_image, 
                ray_trafos_full,
                cfg.noise_specs
                )
        elif cfg.name == 'walnut':
            raise NotImplementedError  # TODO from walnut_interface
        else:
            raise NotImplementedError

        if cfg.name in ['mnist', 'kmnist', 'rectangles', 'walnut_patches']:
            full_angles = ray_trafos_full['geometry'].angles
            full_num_angles = len(full_angles)
            init_angle_inds = np.arange(0, full_num_angles, cfg.angular_sub_sampling)
            acq_angle_inds = np.setdiff1d(np.arange(full_num_angles), np.arange(0, full_num_angles, cfg.angular_sub_sampling))

            proj_inds_per_angle = np.arange(np.prod(ray_trafos_full['ray_trafo'].range.shape)).reshape(ray_trafos_full['ray_trafo'].range.shape)
            assert proj_inds_per_angle.shape[0] == full_num_angles

            ray_trafo_full_mat_flat = ray_trafos_full['ray_trafo_mat'].reshape(-1, np.prod(ray_trafos_full['space'].shape))
            init_angles = full_angles[init_angle_inds]

            hyperparam_fun = None
            if cfg.bed.hyperparam_path_baseline is not None:
                hyperparam_fun = get_hyperparam_fun_from_yaml(
                        os.path.join(get_original_cwd(), cfg.bed.hyperparam_path_baseline),
                        data=cfg.name,
                        noise_stddev=cfg.noise_specs.stddev)

            tvadam_hyperparam_fun = None
            if cfg.bed.use_alternative_recon == 'tvadam':
                tvadam_hyperparam_fun = get_hyperparam_fun_from_yaml(
                        os.path.join(get_original_cwd(), cfg.bed.tvadam_hyperparam_path_baseline),
                        data=cfg.name,
                        noise_stddev=cfg.noise_specs.stddev)
        elif cfg.name == 'walnut':
            raise NotImplementedError  # TODO from walnut_interface
        else:
            raise NotImplementedError

        criterion = ('diagonal_EIG' if cfg.bed.use_diagonal_EIG else 'EIG') if cfg.bed.use_EIG else 'var'

        logged_plot_callbacks = {}
        logged_plot_callbacks['angles'] = plot_angles_callback
        if not cfg.bed.compute_equidistant_baseline and cfg.bed.use_best_inds_from_path is None:
            logged_plot_callbacks[criterion] = plot_obj_callback
        
        obj_list = []
        save_obj_callback = get_save_obj_callback(obj_list)
        callbacks = [save_obj_callback]

        if not cfg.bed.compute_equidistant_baseline:
            use_precomputed_best_inds = None
            if cfg.bed.use_best_inds_from_path is not None:
                use_precomputed_best_inds = np.concatenate(np.load(os.path.join(
                        cfg.bed.use_best_inds_from_path,
                        'bayes_exp_design_{}.npz'.format(i)))['best_inds_per_batch'])

            best_inds, recons = bed_optimal_angles_search(
                ray_trafo_full_mat_flat, proj_inds_per_angle, init_angle_inds, acq_angle_inds,
                observation_full, ray_trafos_full, filtbackproj.to(reconstructor.device), example_image.to(reconstructor.device),
                cov_obs_mat_no_noise, log_noise_model_variance_obs,
                model, bayesianized_model, fwAD_be_model, fwAD_be_modules,
                cfg.bed.total_num_acq_projs, cfg.bed.acq_projs_batch_size,
                cfg.bed.mc_samples, 
                cfg.mrglik.impl.vec_batch_size,
                model_basename='refined_dip_model_{}'.format(i),
                reconstruct_every_k_step=cfg.bed.reconstruct_every_k_step,
                update_network_params=cfg.bed.update_network_params,
                update_prior_hyperparams=cfg.bed.update_prior_hyperparams,
                criterion=criterion,
                use_precomputed_best_inds=use_precomputed_best_inds,
                log_path=cfg.bed.log_path,
                device=reconstructor.device,
                cov_obs_mat_eps_mode=cfg.density.cov_obs_mat_eps_mode,
                cov_obs_mat_eps=cfg.density.cov_obs_mat_eps,
                cov_obs_mat_eps_min_for_auto=cfg.density.cov_obs_mat_eps_min_for_auto,
                cfg_net=cfg.net,
                cfg_marginal_lik=cfg, #TODO
                override_mrglik_iterations=cfg.bed.override_mrglik_iterations,
                hyperparam_fun=hyperparam_fun,
                callbacks=callbacks,
                logged_plot_callbacks=logged_plot_callbacks,
                return_recons=True,
                use_alternative_recon=cfg.bed.use_alternative_recon,
                tvadam_hyperparam_fun=tvadam_hyperparam_fun,
                cfg_tvadam=cfg.bed.tvadam,
                init_state_dict=model.state_dict() if cfg.bed.init_dip_from_mll else None,
                tuple_scale_vec=(
                    scale_vec, 
                    empirical_g_coeff 
                    ),
                g_prior_scale_fct=cfg.bed.g_prior_scale_fct,
                update_scale_vec_via_refined_jac=cfg.bed.update_scale_vec_via_refined_jac,
                cov_obs_mat_stab_by_tsvd=cfg.bed.cov_obs_mat_stab_by_tsvd,
                cov_obs_mat_stab_by_tsvd_reduce_dim_fraction=cfg.bed.cov_obs_mat_stab_by_tsvd_reduce_dim_fraction,
            )

            best_inds_per_batch = [
                    best_inds[j:j+cfg.bed.acq_projs_batch_size]
                    for j in range(0, cfg.bed.total_num_acq_projs, cfg.bed.acq_projs_batch_size)]

            print('angles to acquire (in this order, batch size {:d}):'.format(cfg.bed.acq_projs_batch_size))
            pprint.pprint(dict(zip(best_inds, full_angles[best_inds])), sort_dicts=False, indent=1)

        else:
            recons = bed_eqdist_angles_baseline(
                ray_trafo_full_mat_flat, proj_inds_per_angle, init_angle_inds, acq_angle_inds,
                observation_full, ray_trafos_full, filtbackproj.to(reconstructor.device), example_image.to(reconstructor.device),
                cov_obs_mat_no_noise, log_noise_model_variance_obs,
                model, bayesianized_model, fwAD_be_model, fwAD_be_modules,
                cfg.bed.total_num_acq_projs, cfg.bed.acq_projs_batch_size,
                cfg.bed.mc_samples, 
                cfg.mrglik.impl.vec_batch_size,
                model_basename='baseline_refined_dip_model_{}'.format(i),
                reconstruct_every_k_step=cfg.bed.reconstruct_every_k_step,
                log_path=cfg.bed.log_path,
                device=reconstructor.device,
                cfg_net=cfg.net,
                hyperparam_fun=hyperparam_fun,
                logged_plot_callbacks=logged_plot_callbacks,
                use_alternative_recon=cfg.bed.use_alternative_recon,
                tvadam_hyperparam_fun=tvadam_hyperparam_fun,
                cfg_tvadam=cfg.bed.tvadam,
                init_state_dict=model.state_dict() if cfg.bed.init_dip_from_mll else None,
                )

        bayes_exp_design_dict = {}
        bayes_exp_design_dict['recons'] = recons
        bayes_exp_design_dict['reconstruct_every_k_step'] = cfg.bed.reconstruct_every_k_step
        bayes_exp_design_dict['ground_truth'] = example_image.cpu().numpy()[0, 0]
        bayes_exp_design_dict['obj_per_batch'] = obj_list
        if not cfg.bed.compute_equidistant_baseline:
            bayes_exp_design_dict['best_inds_per_batch'] = best_inds_per_batch

        np.savez('./bayes_exp_design_{}.npz'.format(i), **bayes_exp_design_dict)

if __name__ == '__main__':
    coordinator()
