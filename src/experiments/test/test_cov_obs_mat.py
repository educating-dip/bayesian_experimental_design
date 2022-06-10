import os
import time
import numpy as np
from itertools import islice
import hydra
from omegaconf import DictConfig
from tabulate import tabulate
from copy import deepcopy
from dataset.utils import (
        get_standard_ray_trafos,
        load_testset_MNIST_dataset, load_testset_KMNIST_dataset,
        load_testset_walnut)
from dataset.mnist import simulate
import torch
from hydra.utils import get_original_cwd
import odl
from deep_image_prior import DeepImagePriorReconstructor
from deep_image_prior.utils import PSNR, SSIM
from priors_marglik import BayesianizeModel
from linearized_laplace import compute_jacobian_single_batch
from scalable_linearised_laplace import (
        add_batch_grad_hooks, prior_cov_obs_mat_mul, get_prior_cov_obs_mat, get_diag_prior_cov_obs_mat,
        get_unet_batch_ensemble, get_fwAD_model, vec_weight_prior_cov_mul)

@hydra.main(config_path='../cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

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
    else:
        raise NotImplementedError

    torch.manual_seed(0)
    np.random.seed(0)

    for i, data_sample in enumerate(islice(loader, cfg.num_images)):

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed + i)  # for reproducible noise in simulate

        if cfg.name in ['mnist', 'kmnist']:
            example_image, _ = data_sample
            observation, filtbackproj, example_image = simulate(
                example_image,
                ray_trafos,
                cfg.noise_specs
                )
        elif cfg.name == 'walnut':
            observation, filtbackproj, example_image = data_sample
        else:
            raise NotImplementedError

        reconstructor = DeepImagePriorReconstructor(**ray_trafo, cfg=cfg.net)

        if cfg.name in ['mnist', 'kmnist']:  
            recon, _ = reconstructor.reconstruct(
                observation, fbp=filtbackproj, ground_truth=example_image)
            torch.save(reconstructor.model.state_dict(),
                    './dip_model_{}.pt'.format(i))
        elif cfg.name == 'walnut':
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
        modules = bayesianized_model.get_all_modules_under_prior()
        v = torch.rand(cfg.mrglik.impl.vec_batch_size, 1, *ray_trafos['ray_trafo'].range.shape).to(reconstructor.device)
        log_noise_model_variance_obs = torch.tensor(0.).to(reconstructor.device)
        compare_with_assembled_jac_mat_mul = cfg.name in ['mnist', 'kmnist']

        if compare_with_assembled_jac_mat_mul:
            jac = compute_jacobian_single_batch(
                    filtbackproj.to(reconstructor.device),
                    reconstructor.model,
                    modules, example_image.numel())

        add_batch_grad_hooks(reconstructor.model, modules)

        be_model, be_module_mapping = get_unet_batch_ensemble(reconstructor.model, v.shape[0], return_module_mapping=True)
        be_modules = [be_module_mapping[m] for m in modules]

        fwAD_be_model, fwAD_be_module_mapping = get_fwAD_model(be_model, return_module_mapping=True, share_parameters=True)
        fwAD_be_modules = [fwAD_be_module_mapping[m] for m in be_modules]

        ray_trafos['ray_trafo_module'].to(reconstructor.device)
        ray_trafos['ray_trafo_module_adj'].to(reconstructor.device)

        if cfg.mrglik.impl.use_fwAD_for_jvp:
            print('using forward-mode AD')
            cov_obs_mat = get_prior_cov_obs_mat(ray_trafos, filtbackproj.to(reconstructor.device), bayesianized_model, reconstructor.model, fwAD_be_model, fwAD_be_modules, log_noise_model_variance_obs, cfg.mrglik.impl.vec_batch_size, use_fwAD_for_jvp=True)
        else:
            print('using finite differences')
            cov_obs_mat = get_prior_cov_obs_mat(ray_trafos, filtbackproj.to(reconstructor.device), bayesianized_model, reconstructor.model, be_model, be_modules, log_noise_model_variance_obs, cfg.mrglik.impl.vec_batch_size, use_fwAD_for_jvp=False)

        obs_numel = np.prod(ray_trafos['ray_trafo'].range.shape)
        batch_obs_domain = odl.rn([obs_numel, cfg.mrglik.impl.vec_batch_size], dtype=np.float32)

        class CovObsMulOperator(odl.Operator):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs, linear=True)

            def _call(self, v):  # implement right multiplication, batch dim is second
                v = torch.from_numpy(np.asarray(v)).T.view(self.domain.shape[1], 1, *ray_trafos['ray_trafo'].range.shape).to(reconstructor.device)
                v = prior_cov_obs_mat_mul(ray_trafos, filtbackproj.to(reconstructor.device),
                        bayesianized_model, reconstructor.model,
                        fwAD_be_model if cfg.mrglik.impl.use_fwAD_for_jvp else be_model,
                        fwAD_be_modules if cfg.mrglik.impl.use_fwAD_for_jvp else be_modules,
                        v, log_noise_model_variance_obs, use_fwAD_for_jvp=cfg.mrglik.impl.use_fwAD_for_jvp).detach()
                v = v.view(self.domain.shape[1], self.domain.shape[0]).T.cpu().numpy()
                return v

            @property
            def adjoint(self):
                return self

        cov_obs_mul_op = CovObsMulOperator(domain=batch_obs_domain, range=batch_obs_domain)
        cov_obs_mat_op = odl.operator.tensor_ops.MatrixOperator(cov_obs_mat.cpu().numpy(), domain=batch_obs_domain, range=batch_obs_domain)

        cov_obs_mul_mat_repr = odl.operator.oputils.matrix_representation(cov_obs_mul_op)
        cov_obs_mul_mat_repr = np.array(cov_obs_mul_mat_repr)[:, 0, :, 0].reshape(obs_numel, obs_numel)
        cov_obs_mul_mat_repr_op = odl.operator.tensor_ops.MatrixOperator(cov_obs_mul_mat_repr, domain=batch_obs_domain, range=batch_obs_domain)

        print('opnorms')
        print('opnorm of cov_obs_mul_op', odl.operator.oputils.power_method_opnorm(cov_obs_mul_op, maxiter=100, rtol=1e-05, atol=1e-08))
        print('opnorm of cov_obs_mul_mat_repr_op', odl.operator.oputils.power_method_opnorm(cov_obs_mul_mat_repr_op, maxiter=100, rtol=1e-05, atol=1e-08))
        print('opnorm of cov_obs_mat_op', odl.operator.oputils.power_method_opnorm(cov_obs_mat_op, maxiter=100, rtol=1e-05, atol=1e-08))
        print('differences')
        print('opnorm of cov_obs_mul_mat_repr_op-cov_obs_mul_op', odl.operator.oputils.power_method_opnorm(cov_obs_mul_mat_repr_op-cov_obs_mul_op, maxiter=100, rtol=1e-05, atol=1e-08))
        print('opnorm of cov_obs_mul_op-cov_obs_mat_op', odl.operator.oputils.power_method_opnorm(cov_obs_mul_op-cov_obs_mat_op, maxiter=100, rtol=1e-05, atol=1e-08))
        print('opnorm of cov_obs_mul_mat_repr_op-cov_obs_mat_op', odl.operator.oputils.power_method_opnorm(cov_obs_mul_mat_repr_op-cov_obs_mat_op, maxiter=100, rtol=1e-05, atol=1e-08))

        np.random.seed(0)
        cov_obs_mul_op_results = []
        cov_obs_mul_mat_repr_op_results = []
        cov_obs_mat_op_results = []
        for _ in range(1000):
            test_obs = np.random.random(batch_obs_domain.shape)
            cov_obs_mul_op_results.append(cov_obs_mul_op(test_obs))
            cov_obs_mul_mat_repr_op_results.append(cov_obs_mul_mat_repr_op(test_obs))
            cov_obs_mat_op_results.append(cov_obs_mat_op(test_obs))
            # cov_obs_mat_op_results.append(cov_obs_mat_op.adjoint(test_obs))  # adjoint seems closer to others
            # cov_obs_mat_op_results.append(odl.operator.tensor_ops.MatrixOperator(cov_obs_mat.T.cpu().numpy(), domain=batch_obs_domain, range=batch_obs_domain)(test_obs))

        print('mse of cov_obs_mul_mat_repr_op(v) - cov_obs_mul_op(v)', np.mean(
                np.asarray(cov_obs_mul_mat_repr_op_results) - np.asarray(cov_obs_mul_op_results))**2)
        print('mse of cov_obs_mul_op(v) - cov_obs_mat_op(v)', np.mean(
                np.asarray(cov_obs_mul_op_results) - np.asarray(cov_obs_mat_op_results))**2)
        print('mse of cov_obs_mul_mat_repr_op(v) - cov_obs_mat_op(v)', np.mean(
                np.asarray(cov_obs_mul_mat_repr_op_results) - np.asarray(cov_obs_mat_op_results))**2)

        if compare_with_assembled_jac_mat_mul:
            from priors_marglik import BlocksGPpriors
            block_priors = BlocksGPpriors(
                    reconstructor.model,
                    bayesianized_model,
                    reconstructor.device,
                    cfg.mrglik.priors.lengthscale_init,
                    cfg.mrglik.priors.variance_init,
                    lin_weights=None)

            Kxx = block_priors.matrix_prior_cov_mul(jac) @ jac.transpose(1, 0) # J * Sigma_theta * J.T
            # Kxx = vec_weight_prior_cov_mul(bayesianized_model, jac) @ jac.transpose(1, 0) # J * Sigma_theta * J.T

            # constructing Kyy
            Kyy = ray_trafos['ray_trafo_module'](Kxx.view(example_image.numel(), *example_image.shape[1:]))
            Kyy = Kyy.view(example_image.numel(), -1).T.view(-1, *example_image.shape[1:])
            Kyy = ray_trafos['ray_trafo_module'](Kyy).view(-1, np.prod(v.shape[2:])) + torch.exp(log_noise_model_variance_obs) * torch.eye(np.prod(v.shape[2:]), device=reconstructor.device)

            ray_trafo_mat = torch.from_numpy(ray_trafos['ray_trafo_mat'])
            ray_trafo_mat = ray_trafo_mat.view(ray_trafo_mat.shape[0] * ray_trafo_mat.shape[1], -1).to(reconstructor.device)

            print('opnorm of Kyy', odl.operator.oputils.power_method_opnorm(odl.operator.tensor_ops.MatrixOperator(Kyy.detach().cpu().numpy()), maxiter=100, rtol=1e-05, atol=1e-08))
            print('opnorm of cov_obs_mat-Kyy', odl.operator.oputils.power_method_opnorm(odl.operator.tensor_ops.MatrixOperator((cov_obs_mat-Kyy.detach()).cpu().numpy()), maxiter=100, rtol=1e-05, atol=1e-08))
            print('opnorm of cov_obs_mat.T-Kyy', odl.operator.oputils.power_method_opnorm(odl.operator.tensor_ops.MatrixOperator((cov_obs_mat.T-Kyy.detach()).cpu().numpy()), maxiter=100, rtol=1e-05, atol=1e-08))
 
    print('max GPU memory used:', torch.cuda.max_memory_allocated())

if __name__ == '__main__':
    coordinator()
