from multiprocessing.sharedctypes import Value
import os
import datetime
from warnings import warn
from math import ceil
from copy import deepcopy
import numpy as np
import hydra
from hydra.utils import get_original_cwd
import yaml
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from dataset.utils import get_standard_ray_trafos

# python plot_angle_selections.py +bed.plot_angle_selections.image_idx=1 +bed.plot_angle_selections.bed_runs_yaml=rectangles_runs.yaml +bed.plot_angle_selections.noise=0.05 +bed.plot_angle_selections.priors=['mll_prior','mll_prior_refinement','g_prior','g_prior_refinement','linear_model_isotropic','linear_model_gp'] +bed.plot_angle_selections.criterion=var

def plot_angle_selection(ax, init_angle_inds, full_angles, selected_angle_inds, batch_size=5, max_num_acq_angles=20, colors_per_batch=('tab:blue', 'tab:orange', 'tab:green', 'tab:red'), use_legend=True, legend_kwargs=None):
    full_angles = full_angles
    handles = []
    num_angles_list = range(len(init_angle_inds), len(init_angle_inds) + max_num_acq_angles, batch_size)
    for j, num_angles in enumerate(num_angles_list):
        color = colors_per_batch[j]
        if (len(full_angles) % (num_angles + batch_size) == 0
                and (num_angles + batch_size) % len(init_angle_inds) == 0):
            baseline_step = len(full_angles) // (num_angles + batch_size)
            baseline_angle_inds = np.arange(0, len(full_angles), baseline_step)
        else:
            baseline_angle_inds = None
        # for theta in full_angles[init_angle_inds]:
        #     ax.plot([theta, theta], [0.1, 1.], color='gray')
        for theta in full_angles[np.setdiff1d(list(range(len(full_angles))), init_angle_inds)]:
            ax.plot([theta, theta], [0., 1.], color='gray', linewidth=0.25, alpha=0.025)
        if baseline_angle_inds is not None:
            for theta in full_angles[baseline_angle_inds]:
                baseline_h, = ax.plot([theta, theta], [0., 1.], color='gray', linewidth=0.5, alpha=0.5, zorder=1.5)
        for theta in full_angles[selected_angle_inds[num_angles-len(init_angle_inds):num_angles-len(init_angle_inds)+batch_size]]:
            h, = ax.plot([theta, theta], [0.1, .9 - (.9-.7) * j / (len(num_angles_list) - 1)], color=color, zorder=2.+0.1*(len(num_angles_list)-j))
        handles.append(h)
    ax.set_yticks([])
    ax.set_ylim((0., 1.))
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_thetagrids(full_angles[init_angle_inds]/np.pi*180.)
    if use_legend:
        legend_kwargs = legend_kwargs or {}
        labels = ['angles {:d}-{:d}'.format(num_angles+1, num_angles+batch_size) for num_angles in num_angles_list]
        handles.append(Line2D([0], [0], color='black', lw=1.5))
        labels.append('initial angles $\mathcal{B}^{(0)}$')
        handles.append(baseline_h)
        labels.append('equidistant selection')
        legend_kwargs.setdefault('ncol', len(handles))
        ax.legend(handles, labels, **legend_kwargs)
    ax.grid(linewidth=1.5, color='black')

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

@hydra.main(config_path='../../../cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    if cfg.use_double:
        torch.set_default_tensor_type(torch.DoubleTensor)

    runs_yaml = os.path.join(get_original_cwd(), cfg.bed.plot_angle_selections.bed_runs_yaml)
    with open(runs_yaml, 'r') as f:
        runs = yaml.safe_load(f)

    for run_path_list in walk_dict(runs):
        if run_path_list:
            for j, p in enumerate(run_path_list):
                assert isinstance(p, str)
                if p.find(':') != -1 and p.find(':') < p.find('/'):
                    # remove server spec
                    _, p = p.split(':', maxsplit=1)
                    run_path_list[j] = p

    noise = cfg.bed.plot_angle_selections.noise
    # 0.05, 0.1

    priors = cfg.bed.plot_angle_selections.priors

    image_idx = cfg.bed.plot_angle_selections.image_idx

    criterion = cfg.bed.plot_angle_selections.criterion
    # 'EIG', 'diagonal_EIG', 'var'

    fig, axs = plt.subplots(2, ceil(len(priors) / 2), subplot_kw={'projection': 'polar'}, figsize=(12, 6))

    num_batches = 3

    legend_idx = 3
    legend_kwargs = {'loc': 'upper center', 'bbox_to_anchor': (0.5, 0.15)}

    ray_trafos_full = None

    legend = None

    for i, (ax, prior) in enumerate(zip(axs.T.flat, priors)):
        try:
            paths = runs[noise][prior][criterion]['dip']
        except KeyError:
            paths = runs[noise][prior][criterion]['tv']
        cfgs = [OmegaConf.load(os.path.join(p, '.hydra', 'config.yaml')) for p in paths]
        path = next(p for p, cfg in zip(paths, cfgs) if image_idx in range(cfg.get('skip_first_images', 0), cfg.num_images))
        bayes_exp_design_dict = np.load(os.path.join(
                path,
                'bayes_exp_design_{}.npz'.format(image_idx)))
        selected_angle_inds = np.concatenate(bayes_exp_design_dict['best_inds_per_batch'])

        full_num_angles = cfgs[0].beam_num_angle
        init_angle_inds = np.arange(0, full_num_angles, cfgs[0].angular_sub_sampling)

        full_angles = np.linspace(0., np.pi, full_num_angles, endpoint=False) + 0.5 * np.pi / full_num_angles

        bed_cfg = OmegaConf.load(os.path.join(path, '.hydra', 'config.yaml'))
        if ray_trafos_full is None:  # should be the same for all, so do not reload
            ray_trafos_full = get_standard_ray_trafos(bed_cfg, return_torch_module=True, return_op_mat=True, override_angular_sub_sampling=1)

        plot_angle_selection(
                ax, init_angle_inds, full_angles, selected_angle_inds,
                batch_size=cfgs[0].bed.reconstruct_every_k_step,
                max_num_acq_angles=cfgs[0].bed.reconstruct_every_k_step * num_batches,
                use_legend=(i == legend_idx), legend_kwargs=legend_kwargs)
        ax.set_title(prior_title_dict[prior], pad=-15)

        if ax.get_legend() is not None:
            legend = ax.get_legend()

    suffix = ''
    fig.savefig('./plot_angle_selections_{}_{}{}_image{}.pdf'.format(noise, criterion, suffix, image_idx), bbox_inches='tight', pad_inches=0., bbox_extra_artists=(legend,))
    fig.savefig('./plot_angle_selections_{}_{}{}_image{}.png'.format(noise, criterion, suffix, image_idx), bbox_inches='tight', pad_inches=0., bbox_extra_artists=(legend,), dpi=600)

    # just use the latest dict
    ground_truth = bayes_exp_design_dict['ground_truth']
    fig, ax = plt.subplots()
    ax.imshow(ground_truth, cmap='gray', interpolation='none')
    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.set_axis_off()
    fig.savefig('ground_truth_{}.pdf'.format(image_idx), bbox_inches='tight', pad_inches=0.)

if __name__ == '__main__':
    coordinator()
