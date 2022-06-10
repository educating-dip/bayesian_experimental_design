from importlib.resources import path
import sys
sys.path.append('../')

import os
import numpy as np 
import numpy as np
import torch
import tensorly as tl
tl.set_backend('pytorch')
import matplotlib
import matplotlib.pyplot as plt
from deep_image_prior.utils import PSNR, SSIM

DIRPATH='/home/rb876/rds/rds-t2-cs133-hh9aMiOkJqI/dip/dip_bayesian_ext/src/experiments/evaluation/'

set_xlim_dict = {
    (5, 0.05): 0.90, 
    (5, 0.1): 0.95, 
    (10, 0.05): 0.45, 
    (10, 0.1): 0.6,
    (20, 0.05): 0.75,
    (20, 0.1): 0.35, 
    (30, 0.05): 0.25,
    (30, 0.1): 0.25
}

dic = {'images':
        {   
            'num': 1, 
            'n_rows': 1,
            'n_cols': 7,
            'figsize': (10, 3),
            'idx_to_norm': [],
            'idx_add_cbar': [4, 5],
            'idx_add_test_loglik': [4, 5],
            'idx_test_log_lik': {4:0, 5:1},
            'idx_hist_insert': [6],
        },
        'hist': 
        {
            'num_bins': 25,
            'num_k_retained': 5, 
            'opacity': [0.3, 0.3, 0.3], 
            'zorder': [10, 5, 0],
            'color': {6: ['#e63946', '#35DCDC', '#5A6C17']}, 
            'linewidth': 0.75, 
            },            
}


def kmnist_image_fig_subplots(data, loglik, filename, titles):

    fs_m1 = 8  # for figure ticks
    fs = 10  # for regular figure text
    fs_p1 = 24 #  figure titles

    matplotlib.rc('font', size=fs)          # controls default text sizes
    matplotlib.rc('axes', titlesize=fs)     # fontsize of the axes title
    matplotlib.rc('axes', labelsize=fs)     # fontsize of the x and y labels
    matplotlib.rc('xtick', labelsize=fs_m1)    # fontsize of the tick labels
    matplotlib.rc('ytick', labelsize=fs_m1)    # fontsize of the tick labels
    matplotlib.rc('legend', fontsize=fs_m1)    # legend fontsize
    matplotlib.rc('figure', titlesize=fs_p1)   # fontsize of the figure title

    matplotlib.rc('font', **{'family':'serif', 'serif': ['Palatino']})
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble']=r'\usepackage{amsmath}'

    fig, axs = plt.subplots(dic['images']['n_rows'], dic['images']['n_cols'], figsize=dic['images']['figsize'],
          facecolor='w', edgecolor='k', constrained_layout=True)
    
    for i, (el, ax, title) in enumerate(zip(data, axs.flatten(), titles)):
        if i in dic['images']['idx_hist_insert']:
            kws = dict(histtype= "stepfilled", linewidth = dic['hist']['linewidth'], ls='dashed', density=True)
            for (el, alpha, zorder, color, label) in zip(el[0], dic['hist']['opacity'], 
                dic['hist']['zorder'], dic['hist']['color'][i], el[1]):
                    ax.hist(el.flatten(), bins=dic['hist']['num_bins'], zorder=zorder,
                        facecolor=hex_to_rgb(color, alpha), edgecolor=hex_to_rgb(color, alpha=1), label=label, **kws)
            ax.set_title(title, y=1.01)
            ax.set_xlim([0, set_xlim_dict[(num_angles, stddev)]])
            ax.set_ylim([0.09, 90])
            ax.set_yscale('log')
            ax.set_ylabel('density')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_aspect( (ax.get_xlim()[1]-ax.get_xlim()[0]) / ( np.log10(ax.get_ylim()[1]) - np.log10(ax.get_ylim()[0])) )

            ax.legend()
            ax.grid(alpha=0.3)
        else:
            im = ax.imshow(el, cmap='gray', )
            ax.set_title(title)
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            if i in dic['images']['idx_add_cbar']: 
                fig.colorbar(im, ax=ax, shrink=0.35)

            if i in dic['images']['idx_add_test_loglik']:
                ax.set_xlabel('log-likelihood: ${:.4f}$'.format(loglik[dic['images']['idx_test_log_lik'][i]]))

    fig.savefig(filename + '.png', dpi=100)
    fig.savefig(filename + '.pdf')

def hex_to_rgb(value, alpha):
    value = value.lstrip('#')
    lv = len(value)
    out = tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    out = [el / 255 for el in out] + [alpha]
    return tuple(out) 

def gather_data_from_bayes_dip(idx, path_to_data):
    
    recon_data = np.load(os.path.join(path_to_data, 'recon_info_{}.npz'.format(idx)),  allow_pickle=True)
    log_lik_data = np.load(os.path.join(path_to_data, 'test_log_lik_info_{}.npz'.format(idx)),  allow_pickle=True)['test_log_lik'].item()

    example_image = recon_data['image'].squeeze()
    filtbackproj = recon_data['filtbackproj'].squeeze()
    observation = recon_data['observation'].squeeze()
    recon = recon_data['recon'].squeeze()
    abs_error = np.abs(example_image - recon)
    pred_cov_matrix_mll = recon_data['model_post_cov_no_predcp']
    std_pred_mll = np.sqrt( np.diag(pred_cov_matrix_mll) - 0.000326).reshape(28, 28)
    
    return (example_image, filtbackproj, observation, recon, abs_error, std_pred_mll, log_lik_data['test_loglik_MLL'])

def gather_data_from_bayes_dip_lin(idx, path_to_data): 

    recon_data = np.load(os.path.join(path_to_data, 'recon_info_{}.npz'.format(idx)),  allow_pickle=True)
    test_log_lik_mll_lin = np.load(os.path.join(path_to_data, 'test_log_lik_info_{}.npz'.format(idx)),  allow_pickle=True)['test_log_lik'].item()
    # recon_data_lin = torch.load(os.path.join(path_to_data,'./linearized_weights_{}.pt'.format(idx)))
    # recon_lin = recon_data_lin['linearized_prediction'].squeeze().reshape(28, 28).cpu().numpy()
    pred_cov_matrix_mll = recon_data['model_post_cov_no_predcp']
    std_pred_mll_lin = np.sqrt(np.diag(pred_cov_matrix_mll) - 0.000326).reshape(28, 28)
    
    return std_pred_mll_lin, test_log_lik_mll_lin['test_loglik_MLL']

def normalized_error_for_qq_plot(recon, image, std):
    normalized_error = (recon - image) / std
    return normalized_error

if __name__ == "__main__":


    for angles in [20]:
        for noise in [0.05]: 
            global num_angles
            num_angles = angles
            global stddev
            stddev = noise

            for idx in range(dic['images']['num']):

                (example_image, filtbackproj, observation, recon, abs_error, std_pred_mll, test_log_lik_mll) = gather_data_from_bayes_dip(idx, '/home/rb876/rds/rds-t2-cs133-hh9aMiOkJqI/dip/scripts/outputs/2022-01-25T18:39:01.923124Z')

                (std_pred_mll_lin, test_log_lik_mll_lin) = gather_data_from_bayes_dip_lin(idx, '/home/rb876/rds/rds-t2-cs133-hh9aMiOkJqI/dip/scripts/outputs/2022-01-25T18:39:01.924045Z')
                folder_name = 'kmnist_lin_num_angles_{}_stddev_{}'.format(num_angles, stddev)
                dir_path = os.path.join('./', 'images', folder_name)
                breakpoint()
                if not os.path.isdir(dir_path):
                    os.makedirs(dir_path)

                kmnist_image_fig_subplots(
                    (
                    example_image, np.transpose(observation), recon, abs_error, std_pred_mll, std_pred_mll_lin,
                    (
                        (abs_error, std_pred_mll, std_pred_mll_lin),
                        ['$error$', 'std-dev  ($\\tilde \\theta$)', 'std-dev ($\\theta^{\star}$)']
                    ), 
                    ),
                    (test_log_lik_mll, test_log_lik_mll_lin),
                    dir_path + '/lin_main_{}'.format(idx), 
                    ['${\mathbf{x}}$', '$\mathbf{y}_{\delta}$', '${\mathbf{x}^* (\\tilde \\theta)}$', 'error', 'std-dev ($\\tilde \\theta$)', 'std-dev ($\\theta^{\star}$)',  'marginal std-dev']
                    )
