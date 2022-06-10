import os
import odl
import torch
import scipy
import datetime
import tensorboardX
import socket
import numpy as np
from deep_image_prior.utils import PSNR
from deep_image_prior import normalize
from odl.contrib.torch import OperatorModule
from dataset.matrix_ray_trafo_utils import get_matrix_ray_trafo_module


class adjust_filtbackproj_module:

    def __init__(self, space, geometry):
        self.geometry = geometry
        self.filt = OperatorModule(odl.tomo.analytic.filtered_back_projection.fbp_filter_op(
            odl.tomo.RayTransform(space, geometry)
            )
        )

    def __call__(self, data, adj_module):
        pad = torch.zeros(
            data.shape[:2] + self.geometry.partition.shape, dtype=data.dtype
            )
        pad[:, :, :data.shape[2], :] = data 
        scale = self.geometry.partition.shape[0] / data.shape[2]
        return adj_module(
            scale * self.filt(pad)[:, :, :data.shape[2], :].to(data.dtype)
        )

def _get_ray_trafo_adj_module(ray_trafo_mat_flat, cur_proj_inds_list, im_shape, num_projs_per_acq, dtype):
    ray_trafo_module_adj = get_matrix_ray_trafo_module(
            # reshaping of matrix rows to (len(cur_proj_inds_list), num_projs_per_acq) is row-major
            ray_trafo_mat_flat[np.concatenate(cur_proj_inds_list)],
            im_shape=im_shape, proj_shape=(len(cur_proj_inds_list), num_projs_per_acq),
            adjoint=True, sparse=scipy.sparse.isspmatrix(ray_trafo_mat_flat)).to(dtype=dtype)
    return ray_trafo_module_adj

def greedy_optimal_angle_search(
    example_image, observation_full, ray_trafos_full, ray_trafo_mat_flat, 
    total_num_acq_projs, 
    proj_inds_per_angle, init_proj_inds_list,
    acq_angle_inds, init_angle_inds, 
    log_path, 
    eqdist_filtbackproj=None
    ):

    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    comment = 'greedy_experimetal_design'
    logdir = os.path.join(
        log_path,
        current_time + '_' + socket.gethostname() + comment
    )
    writer = tensorboardX.SummaryWriter(logdir=logdir)

    num_projs_per_acq = proj_inds_per_angle.shape[-1]
    assert all(len(inds) == num_projs_per_acq for inds in init_proj_inds_list)
    
    im_shape = example_image.shape[2:]
    acq_angle_inds_list = list(acq_angle_inds)
    cur_angle_inds_list = list(init_angle_inds)
    cur_proj_inds_list = list(init_proj_inds_list)

    adjusted_fbp_module = adjust_filtbackproj_module(
        ray_trafos_full['space'], 
        ray_trafos_full['geometry']
    )

    opt_abs_angles_inds = []
    for i in range(total_num_acq_projs):
        psnrs = []
        for ind_angle in acq_angle_inds_list:
            add_cur_angle_inds_list = cur_angle_inds_list + [ind_angle]
            add_cur_proj_inds_list = cur_proj_inds_list + [proj_inds_per_angle[ind_angle]]
            ray_trafo_module_adj = _get_ray_trafo_adj_module(ray_trafo_mat_flat, add_cur_proj_inds_list, im_shape, num_projs_per_acq, dtype=observation_full.dtype)
            observation = observation_full[:, :, add_cur_angle_inds_list, :]
            cur_fbp = adjusted_fbp_module(observation, ray_trafo_module_adj)
            psnr = PSNR(cur_fbp[0, 0].numpy(), example_image[0, 0].cpu().numpy())
            psnrs.append(psnr)

        opt_abs_angle_ind = acq_angle_inds_list[np.argmax(psnrs)]
        cur_angle_inds_list.append(opt_abs_angle_ind)
        cur_proj_inds_list.append(proj_inds_per_angle[opt_abs_angle_ind])
        opt_abs_angles_inds.append(opt_abs_angle_ind)

        del acq_angle_inds_list[np.argmax(psnrs)]

    observation = observation_full[:, :, cur_angle_inds_list, :]
    ray_trafo_module_adj = _get_ray_trafo_adj_module(ray_trafo_mat_flat, cur_proj_inds_list, im_shape, num_projs_per_acq, dtype=observation_full.dtype)
    opt_fbp = adjusted_fbp_module(observation, ray_trafo_module_adj)
    psnr = PSNR(opt_fbp[0, 0].numpy(), example_image[0, 0].cpu().numpy())
    print('greedy reconstruction\nPSNR {:.4f}'.format(psnr))
    
    writer.add_image('example_image', normalize(example_image[0].numpy()), i)
    writer.add_image('fbp', normalize(opt_fbp[0].numpy()), i)
    writer.add_image('SE', normalize(( example_image[0, 0] - opt_fbp[0].numpy()))**2 , i)
    writer.add_image('eqdist_fbp', normalize(eqdist_filtbackproj[0].numpy()), i)
    writer.close()
