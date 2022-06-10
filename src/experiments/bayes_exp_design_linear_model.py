import os
import pprint
from itertools import islice
from math import ceil
import numpy as np
import hydra
import torch
from omegaconf import OmegaConf, DictConfig
from dataset.utils import (
        get_standard_ray_trafos,
        load_testset_MNIST_dataset, load_testset_KMNIST_dataset,
        load_testset_rectangles_dataset, load_testset_walnut_patches_dataset, 
        load_testset_walnut)
from dataset.mnist import simulate
from hydra.utils import get_original_cwd
import matplotlib.pyplot as plt
from scalable_bayes_exp_design import bed_optimal_angles_search_linear_model, plot_angles_callback, get_hyperparam_fun_from_yaml, plot_obj_callback, get_save_obj_callback_linear_model

@hydra.main(config_path='../cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.use_double:
        torch.set_default_tensor_type(torch.DoubleTensor)

    ray_trafos = get_standard_ray_trafos(cfg, return_torch_module=True, return_op_mat=True)
    ray_trafos_full = get_standard_ray_trafos(cfg, return_torch_module=True, return_op_mat=True, override_angular_sub_sampling=1)

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

    opt_parameters = None

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
        else:
            raise NotImplementedError

        if cfg.use_double:
            observation = observation.double()
            filtbackproj = filtbackproj.double()
            example_image = example_image.double()
            non_normalized_example_image = non_normalized_example_image.double()

        ray_trafos['ray_trafo_module'].to(device)
        ray_trafos['ray_trafo_module_adj'].to(device)
        if cfg.use_double:
            ray_trafos['ray_trafo_module'].to(torch.float64)
            ray_trafos['ray_trafo_module_adj'].to(torch.float64)

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
        else:
            raise NotImplementedError

        logged_plot_callbacks = {}
        logged_plot_callbacks['angles'] = plot_angles_callback
        if not cfg.bed.compute_equidistant_baseline:
            logged_plot_callbacks['EIG'] = plot_obj_callback
        
        obj_list = []
        save_obj_callback = get_save_obj_callback_linear_model(obj_list)
        callbacks = [save_obj_callback]

      
        best_inds, opt_parameters = bed_optimal_angles_search_linear_model(
                ray_trafo_mat_flat=ray_trafo_full_mat_flat,
                proj_inds_per_angle=proj_inds_per_angle,
                init_angle_inds=init_angle_inds,
                acq_angle_inds=acq_angle_inds,
                observation_full=observation_full,
                ray_trafos_full=ray_trafos_full,
                filtbackproj=filtbackproj,
                ground_truth=example_image,
                total_num_acq_projs=cfg.bed.total_num_acq_projs,
                acq_projs_batch_size=cfg.bed.acq_projs_batch_size, 
                reconstruct_every_k_step=cfg.bed.reconstruct_every_k_step,
                callbacks=callbacks,
                mc_samples=cfg.bed.mc_samples,
                use_EIG=cfg.bed.use_EIG,
                use_diagonal_EIG=cfg.bed.use_diagonal_EIG,
                log_path=cfg.bed.log_path,
                mll_optim_kwargs={
                    'lr': cfg.bed.linear_model_mll_optim.lr,
                    'n_steps': cfg.bed.linear_model_mll_optim.n_steps,
                    'patience': cfg.bed.linear_model_mll_optim.patience,
                    'use_gp_model': cfg.bed.linear_model_mll_optim.use_gp_model},
                device=device,
                hyperparam_fun=hyperparam_fun,
                callbacks=callbacks,
                logged_plot_callbacks=logged_plot_callbacks,
                opt_parameters=opt_parameters
            )
        
        best_inds_per_batch = [
                best_inds[j:j+cfg.bed.acq_projs_batch_size]
                for j in range(0, cfg.bed.total_num_acq_projs, cfg.bed.acq_projs_batch_size)]

        print('angles to acquire (in this order, batch size {:d}):'.format(cfg.bed.acq_projs_batch_size))
        pprint.pprint(dict(zip(best_inds, full_angles[best_inds])), sort_dicts=False, indent=1)

        bayes_exp_design_dict = {}
        bayes_exp_design_dict['reconstruct_every_k_step'] = cfg.bed.reconstruct_every_k_step
        bayes_exp_design_dict['ground_truth'] = example_image.cpu().numpy()[0, 0]
        bayes_exp_design_dict['obj_per_batch'] = obj_list
        bayes_exp_design_dict['best_inds_per_batch'] = best_inds_per_batch

        np.savez('./bayes_exp_design_{}.npz'.format(i), **bayes_exp_design_dict)



if __name__ == '__main__':
    coordinator()
