import os
from functools import partial
from math import ceil
from hydra.utils import get_original_cwd
import numpy as np
import odl
from .walnuts_interface import (
        MAX_NUM_ANGLES, PROJS_COLS, VOL_SZ, VOX_SZ, get_projection_data,
        get_single_slice_ray_trafo, get_single_slice_ray_trafo_matrix,
        get_single_slice_ind, get_ground_truth,
        get_first_proj_col_for_sub_sampling)
from .matrix_ray_trafo_utils import MatrixRayTrafo, get_matrix_ray_trafo_module

def get_walnut_single_slice_matrix_ray_trafos(cfg, return_torch_module=True,
                                              return_op_mat=True):

    custom_cfg = cfg.ray_trafo_custom

    data_path_ray_trafo = os.path.join(get_original_cwd(), custom_cfg.data_path)
    matrix_path = os.path.join(get_original_cwd(), custom_cfg.matrix_path)

    space = odl.uniform_discr(
            [-VOX_SZ*VOL_SZ[1]/2, -VOX_SZ*VOL_SZ[2]/2],
            [VOX_SZ*VOL_SZ[1]/2, VOX_SZ*VOL_SZ[2]/2],
            [VOL_SZ[1], VOL_SZ[2]])

    angular_sub_sampling = cfg.angular_sub_sampling
    proj_col_sub_sampling = cfg.proj_col_sub_sampling

    walnut_ray_trafo = get_single_slice_ray_trafo(
            data_path=data_path_ray_trafo,
            walnut_id=custom_cfg.walnut_id,
            orbit_id=custom_cfg.orbit_id,
            angular_sub_sampling=angular_sub_sampling,
            proj_col_sub_sampling=proj_col_sub_sampling)
    matrix = get_single_slice_ray_trafo_matrix(
            path=matrix_path,
            walnut_id=custom_cfg.walnut_id,
            orbit_id=custom_cfg.orbit_id,
            angular_sub_sampling=angular_sub_sampling,
            proj_col_sub_sampling=proj_col_sub_sampling)
    matrix_ray_trafo = MatrixRayTrafo(matrix,
            im_shape=(cfg.size, cfg.size),
            proj_shape=(1, matrix.shape[0],))
#     ray_trafo = matrix_ray_trafo.apply

    class apply_ray_trafo: 
            def __call__(self, x):
                return matrix_ray_trafo.apply(x)
    ray_trafo = apply_ray_trafo()
    ray_trafo.range = odl.rn((1, matrix.shape[0],), dtype=np.float32)

    pseudoinverse = lambda y: walnut_ray_trafo.apply_fdk(y.squeeze(), squeeze=True)

    ray_trafo_dict = {
            'space': space,
            'ray_trafo': ray_trafo,
            'pseudoinverse': pseudoinverse,
            }

    if return_torch_module:
        ray_trafo_dict['ray_trafo_module'] = (
                get_matrix_ray_trafo_module(
                        matrix, (cfg.size, cfg.size),
                        (1, matrix.shape[0],), sparse=True))
        ray_trafo_dict['ray_trafo_module_adj'] = (
                get_matrix_ray_trafo_module(
                        matrix, (cfg.size, cfg.size),
                        (1, matrix.shape[0],), adjoint=True, sparse=True))
        # ray_trafo_dict['pseudoinverse_module'] not implemented
    if return_op_mat:
        ray_trafo_dict['ray_trafo_mat'] = matrix
        ray_trafo_dict['ray_trafo_mat_adj'] = matrix.T

    return ray_trafo_dict

def get_walnut_proj_numel(cfg):
    num_angles = ceil(MAX_NUM_ANGLES / cfg.angular_sub_sampling)
    first_proj_col = get_first_proj_col_for_sub_sampling(
                factor=cfg.proj_col_sub_sampling)
    num_proj_cols = len(range(
            first_proj_col, PROJS_COLS, cfg.proj_col_sub_sampling))

    return num_angles * num_proj_cols

def get_walnut_data(cfg):

    data_path_test = os.path.join(
            get_original_cwd(), cfg.data_path_test)
    data_path_ray_trafo = os.path.join(
            get_original_cwd(), cfg.ray_trafo_custom.data_path)

    ray_trafo_dict = get_walnut_single_slice_matrix_ray_trafos(
            cfg, return_torch_module=False)
    pseudoinverse = ray_trafo_dict['pseudoinverse']

    angular_sub_sampling = cfg.angular_sub_sampling
    proj_col_sub_sampling = cfg.proj_col_sub_sampling

    observation_full = get_projection_data(
            data_path=data_path_test,
            walnut_id=cfg.walnut_id, orbit_id=cfg.orbit_id,
            angular_sub_sampling=angular_sub_sampling,
            proj_col_sub_sampling=proj_col_sub_sampling)

    # WalnutRayTrafo instance needed for selecting and masking the projections
    walnut_ray_trafo = get_single_slice_ray_trafo(
            data_path_ray_trafo,
            walnut_id=cfg.ray_trafo_custom.walnut_id,
            orbit_id=cfg.ray_trafo_custom.orbit_id,
            angular_sub_sampling=angular_sub_sampling,
            proj_col_sub_sampling=proj_col_sub_sampling)

    observation = walnut_ray_trafo.flat_projs_in_mask(
            walnut_ray_trafo.projs_from_full(observation_full))

    filtbackproj = np.asarray(pseudoinverse(observation)) # TODO 

    slice_ind = get_single_slice_ind(
            data_path=data_path_test,
            walnut_id=cfg.walnut_id, orbit_id=cfg.orbit_id)
    ground_truth = get_ground_truth(
            data_path=data_path_test,
            walnut_id=cfg.walnut_id, orbit_id=cfg.orbit_id,
            slice_ind=slice_ind)

    scaling_factor = cfg.get('scaling_factor', 1.)
    if scaling_factor != 1.:
        observation *= scaling_factor
        filtbackproj *= scaling_factor
        ground_truth *= scaling_factor

    return observation[None], filtbackproj, ground_truth

INNER_PART_START_0 = 72
INNER_PART_START_1 = 72
INNER_PART_END_0 = 424
INNER_PART_END_1 = 424

def get_inner_block_indices(block_size):

    num_blocks_0 = VOL_SZ[1] // block_size
    num_blocks_1 = VOL_SZ[2] // block_size
    start_block_0 = INNER_PART_START_0 // block_size
    start_block_1 = INNER_PART_START_1 // block_size
    end_block_0 = ceil(INNER_PART_END_0 / block_size)
    end_block_1 = ceil(INNER_PART_END_1 / block_size)

    block_idx_list = [
        block_idx for block_idx in range(num_blocks_0 * num_blocks_1)
        if block_idx % num_blocks_0 in range(start_block_0, end_block_0) and
        block_idx // num_blocks_0 in range(start_block_1, end_block_1)]

    return block_idx_list
