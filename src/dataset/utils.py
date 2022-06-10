import os
import numpy as np
import odl
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from odl.contrib.torch import OperatorModule
from torch.utils.data import DataLoader, TensorDataset
from hydra.utils import get_original_cwd
from .rectangles import RectanglesDataset
from .walnut_patches import WalnutPatchesDataset
from .walnut import (
        get_walnut_data, get_walnut_single_slice_matrix_ray_trafos,
        get_walnut_proj_numel)
from .pretraining_ellipses import DiskDistributedEllipsesDataset
from .matrix_ray_trafo_utils import MatrixRayTrafo, get_matrix_ray_trafo_module

def load_testset_MNIST_dataset(path='mnist', batchsize=1,
                               crop=False):
    path = os.path.join(get_original_cwd(), path)
    testset = datasets.MNIST(root=path, train=False, download=True,
                             transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                (0.1307,), (0.3081,))]))
    return DataLoader(testset, batchsize, shuffle=False)

def load_testset_KMNIST_dataset(path='kmnist', batchsize=1,
                               crop=False):
    path = os.path.join(get_original_cwd(), path)
    testset = datasets.KMNIST(root=path, train=False, download=True,
                             transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                (0.1307,), (0.3081,))]))
    return DataLoader(testset, batchsize, shuffle=False)

def load_trainset_MNIST_dataset(path='mnist', batchsize=1,
                               crop=False):
    path = os.path.join(get_original_cwd(), path)
    trainset = datasets.MNIST(root=path, train=True, download=True,
                             transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                (0.1307,), (0.3081,))]))
    return DataLoader(trainset, batchsize, shuffle=False)

def load_trainset_KMNIST_dataset(path='kmnist', batchsize=1,
                               crop=False):
    path = os.path.join(get_original_cwd(), path)
    trainset = datasets.KMNIST(root=path, train=True, download=True,
                             transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                (0.1307,), (0.3081,))]))
    return DataLoader(trainset, batchsize, shuffle=False)

def load_testset_rectangles_dataset(cfg, batchsize=1, fixed_seed=True):
    testset = RectanglesDataset(image_size=cfg.image_specs.size, num_rects=cfg.num_rects, num_angle_modes=cfg.num_angle_modes, angle_modes_sigma=cfg.angle_modes_sigma, fixed_seed=fixed_seed)
    return DataLoader(testset, batchsize, shuffle=False)

def load_trainset_rectangles_dataset(cfg, batchsize=1, fixed_seed=42):
    trainset = RectanglesDataset(image_size=cfg.image_specs.size, num_rects=cfg.num_rects, num_angle_modes=cfg.num_angle_modes, angle_modes_sigma=cfg.angle_modes_sigma, fixed_seed=fixed_seed)
    return DataLoader(trainset, batchsize, shuffle=False)

def load_testset_walnut_patches_dataset(cfg, batchsize=1, fixed_seed=True):

    testset = WalnutPatchesDataset(
        image_size=cfg.image_specs.size,
        data_path=cfg.data_path_test, 
        walnut_id=cfg.walnut_id,
        orbit_id=cfg.orbit_id,
        slice_ind=cfg.slice_ind,
        fixed_seed=fixed_seed)
    return DataLoader(testset, batchsize, shuffle=False)

def load_testset_walnut(cfg):
    observation, filtbackproj, ground_truth = get_walnut_data(cfg)

    testset = TensorDataset(
            torch.from_numpy(observation[None, None]),
            torch.from_numpy(filtbackproj[None, None]),
            torch.from_numpy(ground_truth[None, None]))

    return DataLoader(testset, 1, shuffle=False)

def get_standard_ray_trafos(cfg, return_torch_module=True,
                            return_op_mat=False, override_angular_sub_sampling=None):

    ray_trafo_impl = cfg.get('ray_trafo_impl', None)

    if not ray_trafo_impl:  # default, create parallel beam geometry
        half_size = cfg.size / 2
        space = odl.uniform_discr([-half_size, -half_size], [half_size,
                                half_size], [cfg.size, cfg.size],
                                dtype='float32')
        geometry = odl.tomo.parallel_beam_geometry(space,
                num_angles=cfg.beam_num_angle)
        ray_trafo = odl.tomo.RayTransform(space, geometry)
        angular_sub_sampling = (
                (1 if not cfg.angular_sub_sampling else cfg.angular_sub_sampling)
                if override_angular_sub_sampling is None else override_angular_sub_sampling)
        _ray_trafo_angle_sub_sampling = odl.tomo.RayTransform(space, geometry[::angular_sub_sampling])
        pseudoinverse = odl.tomo.fbp_op(_ray_trafo_angle_sub_sampling)
        ray_trafo_mat = \
            odl.operator.oputils.matrix_representation(ray_trafo)
        if angular_sub_sampling != 1:
            ray_trafo_mat = ray_trafo_mat[::angular_sub_sampling]
        ray_trafo_mat_flat = ray_trafo_mat.reshape(-1, cfg.size**2)
        matrix_ray_trafo = MatrixRayTrafo(ray_trafo_mat_flat,
                im_shape=(cfg.size, cfg.size),
                proj_shape=_ray_trafo_angle_sub_sampling.range.shape)
    #     ray_trafo = matrix_ray_trafo.apply

        class apply_ray_trafo: 
                def __call__(self, x):
                    return matrix_ray_trafo.apply(x)
        ray_trafo_range = _ray_trafo_angle_sub_sampling.range
        ray_trafo = apply_ray_trafo()
        ray_trafo.range = ray_trafo_range
        ray_trafo_dict = {
            'space': space,
            'geometry': geometry[::angular_sub_sampling],
            'ray_trafo': ray_trafo,
            'pseudoinverse': pseudoinverse,
            }

        if return_torch_module:
            ray_trafo_dict['ray_trafo_module'] = (
                    get_matrix_ray_trafo_module(
                            ray_trafo_mat_flat, (cfg.size, cfg.size),
                            ray_trafo_range.shape, sparse=False))
            ray_trafo_dict['ray_trafo_module_adj'] = (
                    get_matrix_ray_trafo_module(
                            ray_trafo_mat_flat, (cfg.size, cfg.size),
                            ray_trafo_range.shape, adjoint=True, sparse=False))
            ray_trafo_dict['pseudoinverse_module'] = OperatorModule(pseudoinverse)
        if return_op_mat:
            ray_trafo_dict['ray_trafo_mat'] = ray_trafo_mat
            ray_trafo_mat_adj = ray_trafo_mat.T
            ray_trafo_dict['ray_trafo_mat_adj'] = ray_trafo_mat_adj

    elif ray_trafo_impl == 'custom':
        assert override_angular_sub_sampling is None
        if cfg.ray_trafo_custom.name == 'walnut_single_slice_matrix':
            ray_trafo_dict = get_walnut_single_slice_matrix_ray_trafos(
                    cfg,
                    return_torch_module=return_torch_module,
                    return_op_mat=return_op_mat)
        else:
            raise ValueError('Unknown custom ray trafo \'{}\''.format(
                    cfg.ray_trafo_custom.name))
    else:
        raise ValueError('Unknown ray trafo implementation \'{}\''.format(
                    ray_trafo_impl))

    return ray_trafo_dict


def extract_trafos_as_matrices(ray_trafos): 

    trafo = torch.from_numpy(ray_trafos['ray_trafo_mat'])
    trafo = trafo.reshape(-1, ray_trafos['space'].shape[0]**2)
    trafo_adj = trafo.T
    trafo_adj_trafo = trafo_adj @ trafo
    trafo_trafo_adj = trafo @ trafo_adj

    return trafo, trafo_adj, trafo_adj_trafo, trafo_trafo_adj


def get_pretraining_dataset(cfg, return_ray_trafo_torch_module=True,
                            return_ray_trafo_op_mat=False):
    ray_trafos = get_standard_ray_trafos(cfg,
            return_torch_module=return_ray_trafo_torch_module,
            return_op_mat=return_ray_trafo_op_mat)

    ray_trafo = ray_trafos['ray_trafo']
    pseudoinverse = ray_trafos['pseudoinverse']

    cfg_p = cfg.pretraining

    if cfg_p.noise_specs.noise_type == 'white':
        specs_kwargs = {
            'stddev': cfg_p.noise_specs.stddev
        }
    elif cfg_p.noise_specs.noise_type == 'poisson':
        specs_kwargs = {
            'mu_water': cfg_p.noise_specs.mu_water,
            'photons_per_pixel': cfg_p.noise_specs.photons_per_pixel
        }
    else:
        raise NotImplementedError

    if cfg.name == 'walnut':
        dataset_specs = {
            'diameter': cfg_p.disk_diameter,
            'image_size': cfg.image_specs.size,
            'train_len': cfg_p.train_len,
            'validation_len': cfg_p.validation_len,
            'test_len': cfg_p.test_len}
        ellipses_dataset = DiskDistributedEllipsesDataset(**dataset_specs)
        space = ellipses_dataset.space
        proj_numel = get_walnut_proj_numel(cfg)
        proj_space = odl.rn((1, proj_numel), dtype=np.float32)
        dataset = ellipses_dataset.create_pair_dataset(ray_trafo=ray_trafo,
                pinv_ray_trafo=pseudoinverse,
                domain=space, proj_space=proj_space,
                noise_type=cfg_p.noise_specs.noise_type,
                specs_kwargs=specs_kwargs,
                noise_seeds={
                    'train': cfg_p.seed,
                    'validation': cfg_p.seed + 1,
                    'test': cfg_p.seed + 2})
    else:
        raise NotImplementedError

    return dataset, ray_trafos
