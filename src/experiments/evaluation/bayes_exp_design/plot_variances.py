from multiprocessing.sharedctypes import Value
import os
import datetime
from warnings import warn
from math import ceil
from copy import deepcopy
import numpy as np
import hydra
import yaml
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

# python plot_variances.py +bed.plot_variances.image_idx=1 +bed.plot_variances.path=

def walk_dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from walk_dict(v)
        else:
            yield v

prior_title_dict = {
    'mll_prior': 'linear DIP',
    'mll_prior_refinement': 'linear DIP, retrained',
    'g_prior': 'linear DIP (g-prior)',
    'g_prior_refinement': 'linear DIP (g-prior), retrained',
    'linear_model_isotropic': 'isotropic',
    'linear_model_gp': 'Matern-½ process',
    'random': 'random',
}

def coordinator():

    image_idx = 1

    path_dip = '/home/rb876/rds/rds-t2-cs133-hh9aMiOkJqI/dip/script_angle_selection_data/outputs/2022-06-03T15:59:19.184938Z'
    path_iso = '/home/rb876/rds/rds-t2-cs133-hh9aMiOkJqI/dip/script_angle_selection_data/outputs/2022-06-03T19:24:28.367609Z'

    full_angles = np.linspace(0., np.pi, 200, endpoint=False) + np.pi/200 * 0.5

    bayes_exp_design_dict_dip = np.load(os.path.join(path_dip, 'bayes_exp_design_{}.npz'.format(image_idx)), allow_pickle=True)
    bayes_exp_design_dict_iso = np.load(os.path.join(path_iso, 'bayes_exp_design_{}.npz'.format(image_idx)), allow_pickle=True)


    fig, axs = plt.subplots(2, 3, figsize=(10, 5.5))

    min_var, max_var = np.inf, -np.inf
    image_max_var = -np.inf

    # dip
    for i in range(0, 2):
        obj = bayes_exp_design_dict_dip['obj_per_batch'][i]['obj']
        min_var = min(min_var, np.min(obj))
        max_var = max(max_var, np.max(obj))
        acq_angle_inds = bayes_exp_design_dict_dip['obj_per_batch'][i]['acq_angle_inds']
        var_x = full_angles[acq_angle_inds] * 180. / np.pi
        axs[0, i+1].set_title('acquired angle at t = {}'.format(i))
        axs[0, i+1].plot(var_x, obj, '.')
        axs[0, i+1].plot(full_angles[acq_angle_inds[np.argmax(obj)]] * 180. / np.pi, np.max(obj), '*', color='tab:red', markersize=10)
        axs[0, i+1].invert_xaxis()
        axs[0, i+1].grid(alpha=0.3)
        axs[0, i+1].set_xticks([0, 45, 90, 135, 180])
        axs[0, i+1].set_xticklabels([])
        axs[0, i+1].spines.right.set_visible(False)
        axs[0, i+1].spines.top.set_visible(False)


    s_var_images = bayes_exp_design_dict_dip['obj_per_batch'][0]['s_var_images']
    image_max_var = max(image_max_var, np.max(s_var_images))
    im = axs[0, 0].imshow(s_var_images**0.5, cmap='gray', vmax=image_max_var, interpolation='none')
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])

    min_var, max_var = np.inf, -np.inf
    image_max_var = -np.inf

    # iso
    for i in range(0, 2):
        obj = bayes_exp_design_dict_iso['obj_per_batch'][i]['obj']
        min_var = min(min_var, np.min(obj))
        max_var = max(max_var, np.max(obj))
        acq_angle_inds = bayes_exp_design_dict_iso['obj_per_batch'][i]['acq_angle_inds']
        var_x = full_angles[acq_angle_inds] * 180. / np.pi
        axs[1, i+1].plot(var_x, obj, '.')
        axs[1, i+1].plot(full_angles[acq_angle_inds[np.argmax(obj)]] * 180. / np.pi, np.max(obj), '*', color='tab:red', markersize=10)
        axs[1, i+1].set_xlabel('angle')
        axs[1, i+1].invert_xaxis()
        axs[1, i+1].grid(alpha=0.3)
        axs[1, i+1].set_xticks([0, 45, 90, 135, 180])
        axs[1, i+1].spines.right.set_visible(False)
        axs[1, i+1].spines.top.set_visible(False)
    
    s_var_images = bayes_exp_design_dict_iso['obj_per_batch'][0]['s_var_images']
    image_max_var = max(image_max_var, np.max(s_var_images))
    im = axs[1, 0].imshow(np.reshape(s_var_images**0.5, (128, 128)), cmap='gray', vmax=image_max_var*.85, interpolation='none')
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])

    axs[0, 1].set_ylabel('variance')
    axs[1, 1].set_ylabel('variance')

    axs[0, 0].set_ylabel('linear DIP std-dev')
    axs[1, 0].set_ylabel('isotropic std-dev')

    suffix = ''
    fig.savefig('./plot_variances_{}image{}.pdf'.format(suffix, image_idx), bbox_inches='tight', pad_inches=0.)
    fig.savefig('./plot_variances_{}image{}.png'.format(suffix, image_idx), bbox_inches='tight', pad_inches=0., dpi=600)

    # just use the latest dict
    ground_truth = bayes_exp_design_dict_dip['ground_truth']
    fig, ax = plt.subplots()
    ax.imshow(ground_truth, cmap=plt.get_cmap('Greens'), interpolation='none')
    ax.set_title('ground truth')
    ax.set_axis_off()
    fig.savefig('ground_truth_{}.pdf'.format(image_idx), bbox_inches='tight', pad_inches=0.)


if __name__ == '__main__':
    coordinator()
