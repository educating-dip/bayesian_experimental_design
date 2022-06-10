import os
from itertools import islice
import numpy as np
import random
import hydra
import warnings
from gpytorch.utils.warnings import NumericalWarning
from omegaconf import DictConfig
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
from linearized_weights import weights_linearization
from scalable_linearised_laplace import (
        add_batch_grad_hooks, get_unet_batch_ensemble, get_fwAD_model,
        optim_marginal_lik_low_rank, predictive_image_log_prob, get_prior_cov_obs_mat,
        stabilize_prior_cov_obs_mat, clamp_params)

@hydra.main(config_path='../cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    if cfg.use_double:
        torch.set_default_tensor_type(torch.DoubleTensor)

    if cfg.ignore_gpytorch_numerical_warnings:
        warnings.simplefilter('ignore', NumericalWarning)

    ray_trafos = get_standard_ray_trafos(cfg, return_torch_module=True, return_op_mat=True)

    ray_trafo = {'ray_trafo_module': ray_trafos['ray_trafo_module'],
                 'reco_space': ray_trafos['space'],
                 }

    # data: observation, filtbackproj, example_image
    if cfg.name == 'mnist':
        loader = load_testset_MNIST_dataset()
    elif cfg.name == 'kmnist':
        loader = load_testset_KMNIST_dataset()
    elif cfg.name == 'walnut':
        loader = load_testset_walnut(cfg)
    elif cfg.name == 'rectangles':
        loader = load_testset_rectangles_dataset(cfg)
    elif cfg.name == 'walnut_patches':
        loader = load_testset_walnut_patches_dataset(cfg)
    else:
        raise NotImplementedError

    for i, data_sample in enumerate(islice(loader, cfg.num_images)):
        if i < cfg.get('skip_first_images', 0):
            continue

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed + i)  # for reproducible noise in simulate

        if cfg.name in ['mnist', 'kmnist', 'rectangles', 'walnut_patches']:
            example_image = data_sample[0] if cfg.name in ['mnist', 'kmnist'] else data_sample
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
            torch.save({'observation': observation, 'filtbackproj': filtbackproj, 'ground_truth': example_image},
                    './sample_{}.pt'.format(i))
        elif cfg.name == 'walnut':
            observation, filtbackproj, example_image = data_sample
        else:
            raise NotImplementedError

        if cfg.use_double:
            observation = observation.double()
            filtbackproj = filtbackproj.double()
            example_image = example_image.double()

        reconstructor = DeepImagePriorReconstructor(**ray_trafo, cfg=cfg.net)

        if cfg.name in ['mnist', 'kmnist', 'rectangles', 'walnut_patches']:
            if cfg.load_dip_models_from_path is not None:
                path = os.path.join(cfg.load_dip_models_from_path, 'dip_model_{}.pt'.format(i))
                print('loading model for {} reconstruction from {}'.format(cfg.name, path))
                reconstructor.model.load_state_dict(torch.load(path, map_location=reconstructor.device))
                with torch.no_grad():
                    reconstructor.model.eval()
                    recon, _ = reconstructor.model.forward(filtbackproj.to(reconstructor.device))
                recon = recon[0, 0].cpu().numpy()
            else:
                recon, _ = reconstructor.reconstruct(
                    observation, fbp=filtbackproj, ground_truth=example_image)
                torch.save(reconstructor.model.state_dict(),
                        './dip_model_{}.pt'.format(i))
        elif cfg.name == 'walnut':
            if cfg.load_dip_models_from_path is not None:
                raise NotImplementedError('model for walnut reconstruction cannot be loaded from a previous run, use net.finetuned_params_path instead')
            path = os.path.join(get_original_cwd(), reconstructor.cfg.finetuned_params_path 
                if reconstructor.cfg.finetuned_params_path.endswith('.pt') else reconstructor.cfg.finetuned_params_path + '.pt')
            reconstructor.model.load_state_dict(torch.load(path, map_location=reconstructor.device))
            with torch.no_grad():
                reconstructor.model.eval()
                recon, _ = reconstructor.model.forward(filtbackproj.to(reconstructor.device))
            recon = recon[0, 0].cpu().numpy()
        else:
            raise NotImplementedError
        
        print('DIP reconstruction of sample {:d}'.format(i))
        print('PSNR:', PSNR(recon, example_image[0, 0].cpu().numpy()))
        print('SSIM:', SSIM(recon, example_image[0, 0].cpu().numpy()))

        bayesianized_model = BayesianizeModel(reconstructor, **{'lengthscale_init': cfg.mrglik.priors.lengthscale_init ,
            'variance_init': cfg.mrglik.priors.variance_init}, include_normal_priors=cfg.mrglik.priors.include_normal_priors)

        recon = torch.from_numpy(recon[None, None])
        if cfg.linearize_weights:
            linearized_weights, lin_pred = weights_linearization(cfg, bayesianized_model, filtbackproj, observation, example_image, reconstructor, ray_trafos)
            print('linear reconstruction sample {:d}'.format(i))
            print('PSNR:', PSNR(lin_pred[0, 0].cpu().numpy(), example_image[0, 0].cpu().numpy()))
            print('SSIM:', SSIM(lin_pred[0, 0].cpu().numpy(), example_image[0, 0].cpu().numpy()))

            torch.save({'linearized_weights': linearized_weights, 'linearized_prediction': lin_pred},  
                './linearized_weights_{}.pt'.format(i))

        else:
            linearized_weights = None
            lin_pred = None
        
        # type-II MAP
        modules = bayesianized_model.get_all_modules_under_prior()
        add_batch_grad_hooks(reconstructor.model, modules)

        be_model, be_module_mapping = get_unet_batch_ensemble(reconstructor.model, cfg.mrglik.impl.vec_batch_size, return_module_mapping=True)
        be_modules = [be_module_mapping[m] for m in modules]

        fwAD_be_model, fwAD_be_module_mapping = get_fwAD_model(be_model, return_module_mapping=True, share_parameters=True)
        fwAD_be_modules = [fwAD_be_module_mapping[m] for m in be_modules]

        ray_trafos['ray_trafo_module'].to(bayesianized_model.store_device)
        ray_trafos['ray_trafo_module_adj'].to(bayesianized_model.store_device)
        if cfg.use_double:
            ray_trafos['ray_trafo_module'].to(torch.float64)
            ray_trafos['ray_trafo_module_adj'].to(torch.float64)

        proj_recon = ray_trafos['ray_trafo_module'](recon.to(bayesianized_model.store_device))

        log_noise_model_variance_obs = optim_marginal_lik_low_rank(
            cfg,
            observation,
            (recon, proj_recon),
            ray_trafos, filtbackproj.to(reconstructor.device), bayesianized_model, reconstructor.model, fwAD_be_model, fwAD_be_modules, 
            linearized_weights=linearized_weights, 
            comment = '_recon_num_' + str(i),
            )

        torch.save(bayesianized_model.state_dict(), 
            './bayesianized_model_{}.pt'.format(i))
        torch.save({'log_noise_model_variance_obs': log_noise_model_variance_obs},
            './log_noise_model_variance_obs_{}.pt'.format(i))

        if cfg.mrglik.priors.clamp_variances:  # this only has an effect if clamping was turned off during optimization
            clamp_params(bayesianized_model.gp_log_variances, min=cfg.mrglik.priors.clamp_variances_min_log)
            clamp_params(bayesianized_model.normal_log_variances, min=cfg.mrglik.priors.clamp_variances_min_log)

        cov_obs_mat = get_prior_cov_obs_mat(ray_trafos, filtbackproj.to(reconstructor.device), bayesianized_model, reconstructor.model,
                fwAD_be_model, fwAD_be_modules, log_noise_model_variance_obs,
                cfg.mrglik.impl.vec_batch_size, use_fwAD_for_jvp=cfg.mrglik.impl.use_fwAD_for_jvp, add_noise_model_variance_obs=True)

        torch.save({'cov_obs_mat': cov_obs_mat}, './cov_obs_mat_{}.pt'.format(i))

        if cfg.density.estimate_density_block_wise_exact:
            cov_obs_mat = 0.5 * (cov_obs_mat + cov_obs_mat.T)  # in case of numerical issues leading to asymmetry
            stabilize_prior_cov_obs_mat(cov_obs_mat, eps_mode=cfg.density.cov_obs_mat_eps_mode, eps=cfg.density.cov_obs_mat_eps, eps_min_for_auto=cfg.density.cov_obs_mat_eps_min_for_auto)

            lik_hess_inv_diag_mean = None
            if cfg.name in ['mnist', 'kmnist', 'rectangles', 'walnut_patches']:
                from dataset import extract_trafos_as_matrices
                import tensorly as tl
                tl.set_backend('pytorch')
                # pseudo-inverse computation
                trafos = extract_trafos_as_matrices(ray_trafos)
                trafo = trafos[0]
                if cfg.use_double:
                    trafo = trafo.to(torch.float64)
                trafo = trafo.to(reconstructor.device)
                trafo_T_trafo = trafo.T @ trafo
                U, S, Vh = tl.truncated_svd(trafo_T_trafo, n_eigenvecs=100) # costructing tsvd-pseudoinverse
                lik_hess_inv_diag_mean = (Vh.T @ torch.diag(1/S) @ U.T * torch.exp(log_noise_model_variance_obs)).diag().mean()
            elif cfg.name == 'walnut':
                # pseudo-inverse computation
                trafo = ray_trafos['ray_trafo_mat'].reshape(-1, np.prod(ray_trafos['space'].shape))
                if cfg.use_double:
                    trafo = trafo.astype(np.float64)
                U_trafo, S_trafo, Vh_trafo = scipy.sparse.linalg.svds(trafo, k=100)
                # (Vh.T S U.T U S Vh)^-1 == (Vh.T S^2 Vh)^-1 == Vh.T S^-2 Vh
                S_inv_Vh_trafo = scipy.sparse.diags(1/S_trafo) @ Vh_trafo
                # trafo_T_trafo_inv_diag = np.diag(S_inv_Vh_trafo.T @ S_inv_Vh_trafo)
                trafo_T_trafo_inv_diag = np.sum(S_inv_Vh_trafo**2, axis=0)
                lik_hess_inv_diag_mean = np.mean(trafo_T_trafo_inv_diag) * np.exp(log_noise_model_variance_obs.item())
            print('noise_x_correction_term:', lik_hess_inv_diag_mean)

            approx_log_prob, block_mask_inds, block_log_probs, block_diags, block_eps_values = predictive_image_log_prob(
                    recon.to(reconstructor.device), example_image.to(reconstructor.device),
                    ray_trafos, bayesianized_model, filtbackproj.to(reconstructor.device), reconstructor.model,
                    fwAD_be_model, fwAD_be_modules, log_noise_model_variance_obs,
                    eps_mode=cfg.density.eps_mode, eps=cfg.density.eps, cov_image_eps=cfg.density.cov_image_eps,
                    block_size=cfg.density.block_size_for_approx,
                    vec_batch_size=cfg.mrglik.impl.vec_batch_size, 
                    cov_obs_mat_chol=torch.linalg.cholesky(cov_obs_mat),
                    noise_x_correction_term=lik_hess_inv_diag_mean)

            torch.save({'approx_log_prob': approx_log_prob, 'block_mask_inds': block_mask_inds, 'block_log_probs': block_log_probs, 'block_diags': block_diags, 'block_eps_values': block_eps_values},
                './predictive_image_log_prob_{}.pt'.format(i))
            
            print('approx log prob ', approx_log_prob / example_image.numel())


if __name__ == '__main__':
    coordinator()
