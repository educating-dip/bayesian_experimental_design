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
try:
    import tensorflow as tf
    from tensorflow.core.util import event_pb2
    from tensorflow.python.lib.io import tf_record
    from tensorflow.errors import DataLossError
    TF_AVAILABLE = True
except ModuleNotFoundError:
    TF_AVAILABLE = False

# for paper

# figure 1
# dip
# python psnr_eval_from_runs_yaml.py num_images=30 +bed.psnr_eval.bed_runs_yaml=rectangles_runs.yaml +bed.psnr_eval.mode=max +bed.psnr_eval.noise=0.05 +bed.psnr_eval.recon_method='dip' +bed.psnr_eval.priors=['mll_prior','mll_prior_refinement','g_prior','g_prior_refinement','random'] +bed.psnr_eval.criterions=['var'] +bed.psnr_eval.show_stddev=False
# python psnr_eval_from_runs_yaml.py num_images=30 +bed.psnr_eval.bed_runs_yaml=rectangles_runs.yaml +bed.psnr_eval.mode=max +bed.psnr_eval.noise=0.1 +bed.psnr_eval.recon_method='dip' +bed.psnr_eval.priors=['mll_prior','mll_prior_refinement','g_prior','g_prior_refinement','random'] +bed.psnr_eval.criterions=['var'] +bed.psnr_eval.show_stddev=False
# python psnr_eval_from_runs_yaml.py num_images=30 +bed.psnr_eval.bed_runs_yaml=rectangles_runs.yaml +bed.psnr_eval.mode=max +bed.psnr_eval.noise=0.05 +bed.psnr_eval.recon_method='dip' +bed.psnr_eval.priors=['mll_prior','mll_prior_refinement','g_prior','g_prior_refinement','random'] +bed.psnr_eval.criterions=['var']
# python psnr_eval_from_runs_yaml.py num_images=30 +bed.psnr_eval.bed_runs_yaml=rectangles_runs.yaml +bed.psnr_eval.mode=max +bed.psnr_eval.noise=0.1 +bed.psnr_eval.recon_method='dip' +bed.psnr_eval.priors=['mll_prior','mll_prior_refinement','g_prior','g_prior_refinement','random'] +bed.psnr_eval.criterions=['var']

# figure 2
# tv
# python psnr_eval_from_runs_yaml.py num_images=30 +bed.psnr_eval.bed_runs_yaml=rectangles_runs.yaml +bed.psnr_eval.mode=max +bed.psnr_eval.noise=0.05 +bed.psnr_eval.recon_method='tv' +bed.psnr_eval.priors=['mll_prior','mll_prior_refinement','g_prior','g_prior_refinement','random'] +bed.psnr_eval.criterions=['var'] +bed.psnr_eval.show_stddev=False
# python psnr_eval_from_runs_yaml.py num_images=30 +bed.psnr_eval.bed_runs_yaml=rectangles_runs.yaml +bed.psnr_eval.mode=max +bed.psnr_eval.noise=0.1 +bed.psnr_eval.recon_method='tv' +bed.psnr_eval.priors=['mll_prior','mll_prior_refinement','g_prior','g_prior_refinement','random'] +bed.psnr_eval.criterions=['var'] +bed.psnr_eval.show_stddev=False
# python psnr_eval_from_runs_yaml.py num_images=30 +bed.psnr_eval.bed_runs_yaml=rectangles_runs.yaml +bed.psnr_eval.mode=max +bed.psnr_eval.noise=0.05 +bed.psnr_eval.recon_method='tv' +bed.psnr_eval.priors=['mll_prior','mll_prior_refinement','g_prior','g_prior_refinement','random'] +bed.psnr_eval.criterions=['var']
# python psnr_eval_from_runs_yaml.py num_images=30 +bed.psnr_eval.bed_runs_yaml=rectangles_runs.yaml +bed.psnr_eval.mode=max +bed.psnr_eval.noise=0.1 +bed.psnr_eval.recon_method='tv' +bed.psnr_eval.priors=['mll_prior','mll_prior_refinement','g_prior','g_prior_refinement','random'] +bed.psnr_eval.criterions=['var']

# figure 3
# overall tv
# python psnr_eval_from_runs_yaml.py num_images=30 +bed.psnr_eval.bed_runs_yaml=rectangles_runs.yaml +bed.psnr_eval.mode=max +bed.psnr_eval.noise=0.05 +bed.psnr_eval.recon_method='tv' +bed.psnr_eval.priors=['mll_prior','g_prior','linear_model_isotropic','linear_model_gp','random'] +bed.psnr_eval.criterions=['var','EIG'] +bed.psnr_eval.symbol_legend_kind=criterion +bed.psnr_eval.show_stddev=False
# python psnr_eval_from_runs_yaml.py num_images=30 +bed.psnr_eval.bed_runs_yaml=rectangles_runs.yaml +bed.psnr_eval.mode=max +bed.psnr_eval.noise=0.1 +bed.psnr_eval.recon_method='tv' +bed.psnr_eval.priors=['mll_prior','g_prior','linear_model_isotropic','linear_model_gp','random'] +bed.psnr_eval.criterions=['var','EIG'] +bed.psnr_eval.symbol_legend_kind=criterion +bed.psnr_eval.show_stddev=False
# python psnr_eval_from_runs_yaml.py num_images=30 +bed.psnr_eval.bed_runs_yaml=rectangles_runs.yaml +bed.psnr_eval.mode=max +bed.psnr_eval.noise=0.05 +bed.psnr_eval.recon_method='tv' +bed.psnr_eval.priors=['mll_prior','g_prior','linear_model_isotropic','linear_model_gp','random'] +bed.psnr_eval.criterions=['var','EIG'] +bed.psnr_eval.symbol_legend_kind=criterion
# python psnr_eval_from_runs_yaml.py num_images=30 +bed.psnr_eval.bed_runs_yaml=rectangles_runs.yaml +bed.psnr_eval.mode=max +bed.psnr_eval.noise=0.1 +bed.psnr_eval.recon_method='tv' +bed.psnr_eval.priors=['mll_prior','g_prior','linear_model_isotropic','linear_model_gp','random'] +bed.psnr_eval.criterions=['var','EIG'] +bed.psnr_eval.symbol_legend_kind=criterion


# python psnr_eval_from_runs_yaml.py num_images=30 +bed.psnr_eval.bed_runs_yaml=rectangles_runs.yaml +bed.psnr_eval.mode=max +bed.psnr_eval.noise=0.05 +bed.psnr_eval.recon_method='dip' +bed.psnr_eval.priors=['mll_prior','mll_prior_refinement','g_prior','g_prior_refinement','random'] +bed.psnr_eval.criterions=['var']
# python psnr_eval_from_runs_yaml.py num_images=30 +bed.psnr_eval.bed_runs_yaml=rectangles_runs.yaml +bed.psnr_eval.mode=max +bed.psnr_eval.noise=0.1 +bed.psnr_eval.recon_method='dip' +bed.psnr_eval.priors=['mll_prior','mll_prior_refinement','g_prior','g_prior_refinement','random'] +bed.psnr_eval.criterions=['var']
# python psnr_eval_from_runs_yaml.py num_images=30 +bed.psnr_eval.bed_runs_yaml=rectangles_runs.yaml +bed.psnr_eval.mode=max +bed.psnr_eval.noise=0.05 +bed.psnr_eval.recon_method='tv' +bed.psnr_eval.priors=['mll_prior','mll_prior_refinement','g_prior','g_prior_refinement','random'] +bed.psnr_eval.criterions=['var']
# python psnr_eval_from_runs_yaml.py num_images=30 +bed.psnr_eval.bed_runs_yaml=rectangles_runs.yaml +bed.psnr_eval.mode=max +bed.psnr_eval.noise=0.1 +bed.psnr_eval.recon_method='tv' +bed.psnr_eval.priors=['mll_prior','mll_prior_refinement','g_prior','g_prior_refinement','random'] +bed.psnr_eval.criterions=['var']

# python psnr_eval_from_runs_yaml.py num_images=30 +bed.psnr_eval.bed_runs_yaml=rectangles_runs.yaml +bed.psnr_eval.mode=max +bed.psnr_eval.noise=0.05 +bed.psnr_eval.recon_method='dip' +bed.psnr_eval.priors=['mll_prior','mll_prior_refinement','g_prior','g_prior_refinement','random'] +bed.psnr_eval.criterions=['EIG']
# python psnr_eval_from_runs_yaml.py num_images=30 +bed.psnr_eval.bed_runs_yaml=rectangles_runs.yaml +bed.psnr_eval.mode=max +bed.psnr_eval.noise=0.1 +bed.psnr_eval.recon_method='dip' +bed.psnr_eval.priors=['mll_prior','mll_prior_refinement','g_prior','g_prior_refinement','random'] +bed.psnr_eval.criterions=['EIG']
# python psnr_eval_from_runs_yaml.py num_images=30 +bed.psnr_eval.bed_runs_yaml=rectangles_runs.yaml +bed.psnr_eval.mode=max +bed.psnr_eval.noise=0.05 +bed.psnr_eval.recon_method='tv' +bed.psnr_eval.priors=['mll_prior','mll_prior_refinement','g_prior','g_prior_refinement','random'] +bed.psnr_eval.criterions=['EIG']
# python psnr_eval_from_runs_yaml.py num_images=30 +bed.psnr_eval.bed_runs_yaml=rectangles_runs.yaml +bed.psnr_eval.mode=max +bed.psnr_eval.noise=0.1 +bed.psnr_eval.recon_method='tv' +bed.psnr_eval.priors=['mll_prior','mll_prior_refinement','g_prior','g_prior_refinement','random'] +bed.psnr_eval.criterions=['EIG']


# python psnr_eval_from_runs_yaml.py num_images=30 +bed.psnr_eval.bed_runs_yaml=rectangles_runs.yaml +bed.psnr_eval.mode=max +bed.psnr_eval.noise=0.05 +bed.psnr_eval.recon_method='tv' +bed.psnr_eval.priors=['mll_prior','g_prior','linear_model_isotropic','linear_model_gp','random'] +bed.psnr_eval.criterions=['var']
# python psnr_eval_from_runs_yaml.py num_images=30 +bed.psnr_eval.bed_runs_yaml=rectangles_runs.yaml +bed.psnr_eval.mode=max +bed.psnr_eval.noise=0.1 +bed.psnr_eval.recon_method='tv' +bed.psnr_eval.priors=['mll_prior','g_prior','linear_model_isotropic','linear_model_gp','random'] +bed.psnr_eval.criterions=['var']

# python psnr_eval_from_runs_yaml.py num_images=30 +bed.psnr_eval.bed_runs_yaml=rectangles_runs.yaml +bed.psnr_eval.mode=max +bed.psnr_eval.noise=0.05 +bed.psnr_eval.recon_method='tv' +bed.psnr_eval.priors=['mll_prior','g_prior','linear_model_isotropic','linear_model_gp','random'] +bed.psnr_eval.criterions=['EIG']
# python psnr_eval_from_runs_yaml.py num_images=30 +bed.psnr_eval.bed_runs_yaml=rectangles_runs.yaml +bed.psnr_eval.mode=max +bed.psnr_eval.noise=0.1 +bed.psnr_eval.recon_method='tv' +bed.psnr_eval.priors=['mll_prior','g_prior','linear_model_isotropic','linear_model_gp','random'] +bed.psnr_eval.criterions=['EIG']

# python psnr_eval_from_runs_yaml.py num_images=30 +bed.psnr_eval.bed_runs_yaml=rectangles_runs.yaml +bed.psnr_eval.mode=max +bed.psnr_eval.noise=0.05 +bed.psnr_eval.recon_method='tv' +bed.psnr_eval.priors=['mll_prior','g_prior','linear_model_isotropic','linear_model_gp','random'] +bed.psnr_eval.criterions=['diagonal_EIG']
# python psnr_eval_from_runs_yaml.py num_images=30 +bed.psnr_eval.bed_runs_yaml=rectangles_runs.yaml +bed.psnr_eval.mode=max +bed.psnr_eval.noise=0.1 +bed.psnr_eval.recon_method='tv' +bed.psnr_eval.priors=['mll_prior','g_prior','linear_model_isotropic','linear_model_gp','random'] +bed.psnr_eval.criterions=['diagonal_EIG']


# python psnr_eval_from_runs_yaml.py num_images=30 +bed.psnr_eval.bed_runs_yaml=rectangles_runs.yaml +bed.psnr_eval.mode=max +bed.psnr_eval.noise=0.05 +bed.psnr_eval.recon_method='dip' +bed.psnr_eval.priors=['mll_prior','g_prior','linear_model_isotropic','linear_model_gp','random'] +bed.psnr_eval.criterions=['var','EIG','diagonal_EIG'] +bed.psnr_eval.symbol_legend_kind=criterion
# python psnr_eval_from_runs_yaml.py num_images=30 +bed.psnr_eval.bed_runs_yaml=rectangles_runs.yaml +bed.psnr_eval.mode=max +bed.psnr_eval.noise=0.1 +bed.psnr_eval.recon_method='dip' +bed.psnr_eval.priors=['mll_prior','g_prior','linear_model_isotropic','linear_model_gp','random'] +bed.psnr_eval.criterions=['var','EIG','diagonal_EIG'] +bed.psnr_eval.symbol_legend_kind=criterion
# python psnr_eval_from_runs_yaml.py num_images=30 +bed.psnr_eval.bed_runs_yaml=rectangles_runs.yaml +bed.psnr_eval.mode=max +bed.psnr_eval.noise=0.05 +bed.psnr_eval.recon_method='tv' +bed.psnr_eval.priors=['mll_prior','g_prior','linear_model_isotropic','linear_model_gp','random'] +bed.psnr_eval.criterions=['var','EIG','diagonal_EIG'] +bed.psnr_eval.symbol_legend_kind=criterion
# python psnr_eval_from_runs_yaml.py num_images=30 +bed.psnr_eval.bed_runs_yaml=rectangles_runs.yaml +bed.psnr_eval.mode=max +bed.psnr_eval.noise=0.1 +bed.psnr_eval.recon_method='tv' +bed.psnr_eval.priors=['mll_prior','g_prior','linear_model_isotropic','linear_model_gp','random'] +bed.psnr_eval.criterions=['var','EIG','diagonal_EIG'] +bed.psnr_eval.symbol_legend_kind=criterion


def extract_tensorboard_scalars(log_file=None, save_as_npz=None, tags=None):
    if not TF_AVAILABLE:
        raise RuntimeError('Tensorflow could not be imported, which is '
                           'required by `extract_tensorboard_scalars`')

    def my_summary_iterator(path):
        try:
            for r in tf_record.tf_record_iterator(path):
                yield event_pb2.Event.FromString(r)
        except DataLossError:
            warn('DataLossError occured, terminated reading file')

    if tags is not None:
        tags = [t.replace('/', '_').lower() for t in tags]
    values = {}
    try:
        for event in tqdm(my_summary_iterator(log_file)):
            if event.WhichOneof('what') != 'summary':
                continue
            step = event.step
            for value in event.summary.value:
                use_value = True
                if hasattr(value, 'simple_value'):
                    v = value.simple_value
                elif value.tensor.ByteSize():
                    v = tf.make_ndarray(value.tensor)
                else:
                    use_value = False
                if use_value:
                    tag = value.tag.replace('/', '_').lower()
                    if tags is None or tag in tags:
                        values.setdefault(tag, []).append((step, v))
    except DataLossError as e:
        warn('stopping for log_file "{}" due to DataLossError: {}'.format(
            log_file, e))
    scalars = {}
    for k in values.keys():
        v = np.asarray(values[k])
        steps, steps_counts = np.unique(v[:, 0], return_counts=True)
        scalars[k + '_steps'] = steps
        scalars[k + '_scalars'] = v[np.cumsum(steps_counts)-1, 1]

    if save_as_npz is not None:
        np.savez(save_as_npz, **scalars)

    return scalars

def load_or_convert_logs(path, require_num_steps=None, max_logs=None, log_dirs_end_with='DIP+TV'):
    log_files = []
    def filter_child(d):
        if not (d.endswith(log_dirs_end_with) and os.path.isdir(os.path.join(path, d))):
            return False
        try:
            _ = datetime.datetime.strptime(d[:14], '%b%d_%H-%M-%S')
            return True
        except:
            return False
    path_children = [d for d in os.listdir(path) if filter_child(d)]
    for d in sorted(path_children, key=lambda s: datetime.datetime.strptime(s[:14], '%b%d_%H-%M-%S')):
        path_d = os.path.join(path, d)
        logs_in_d = [f for f in os.listdir(path_d) if f.startswith('events.') and not f.endswith('.npz')]
        assert len(logs_in_d) == 1
        if os.path.getsize(os.path.join(path_d, logs_in_d[0])) > 40:
            log_files.append(os.path.join(path_d, logs_in_d[0]))
            if max_logs is not None and len(log_files) >= max_logs:
                break
    psnrs = []
    best_psnrs = []
    for log_file in log_files:
        npz_cache_filepath = log_file + '.npz'
        log = None
        if os.path.isfile(npz_cache_filepath):
            log = np.load(npz_cache_filepath)
            if require_num_steps is not None and len(log['output_psnr_steps']) != require_num_steps:
                log = None
        if log is None:
            os.makedirs(os.path.dirname(npz_cache_filepath), exist_ok=True)
            extract_tensorboard_scalars(log_file=log_file, save_as_npz=npz_cache_filepath)
            log = np.load(npz_cache_filepath)
            if require_num_steps is not None and len(log['output_psnr_steps']) != require_num_steps:
                log = None
        if log is not None:
            psnrs.append(log['output_psnr_scalars'])
            best_psnrs.append(log['best_output_psnr_scalars'])

    return psnrs, best_psnrs

def ffinf_success_exps(bed_psnrs, baseline_psnrs, condition, thresh=0.1, angle=10):

    def find_indices(lst, condition):
        return [i for i, elem in enumerate(lst) if condition(elem)]

    diff = np.asarray(bed_psnrs[angle]) - np.asarray(baseline_psnrs[angle])
    return find_indices(list(diff), condition)

def select_iters(psnrs, mode):
    if mode == 'max':
        selected_iters = [np.argmax(p) for p in psnrs]
    elif mode == 'last':
        selected_iters = [(len(p) - 1) for p in psnrs]
    else:
        raise ValueError
    return selected_iters

def get_initial_iters_and_psnrs(path, mode, skip_first_images, num_images, num_angles, require_num_iters=None, log_dirs_end_with='DIP+TV'):
    _, best_psnrs = load_or_convert_logs(
            path,
            max_logs=num_images,
            require_num_steps=require_num_iters,
            log_dirs_end_with=log_dirs_end_with,
            )
    # psnrs = psnrs[skip_first_images:num_images]
    best_psnrs = best_psnrs[skip_first_images:num_images]

    selected_iters = select_iters(best_psnrs, mode)
    selected_psnrs = [p[m] for p, m in zip(best_psnrs, selected_iters)]

    initial_iters = {}
    initial_psnrs = {}
    for i in range(num_images - skip_first_images):
        initial_iters.setdefault((num_angles), []).append(selected_iters[i])
        initial_psnrs.setdefault((num_angles), []).append(selected_psnrs[i])

    return initial_iters, initial_psnrs

def get_baseline_iters_and_psnrs(path, mode, skip_first_images, num_images, beam_num_angle, init_num_angles, total_num_acq_projs, acq_projs_batch_size, reconstruct_every_k_step, log_dirs_end_with='DIP+TV'):
    baseline_iters = {}
    baseline_psnrs = {}    
    num_batches = ceil(total_num_acq_projs / acq_projs_batch_size)
    num_angles_list = list(range(
            init_num_angles + acq_projs_batch_size * reconstruct_every_k_step,
            init_num_angles + num_batches * acq_projs_batch_size + 1,
            acq_projs_batch_size * reconstruct_every_k_step))
    num_angles_list = [angle for angle in num_angles_list if (
        beam_num_angle % angle == 0 
            and angle % init_num_angles == 0)
        ]
    psnrs, best_psnrs = load_or_convert_logs(
        path,
        max_logs = num_images * len(num_angles_list),
        log_dirs_end_with=log_dirs_end_with,
    )
    psnrs = psnrs[skip_first_images * len(num_angles_list):num_images * len(num_angles_list)]
    best_psnrs = best_psnrs[skip_first_images * len(num_angles_list):num_images * len(num_angles_list)]

    selected_iters = select_iters(best_psnrs, mode)
    selected_psnrs = [p[m] for p, m in zip(best_psnrs, selected_iters)]
    k = 0
    for _ in range(num_images - skip_first_images): 
        for total_num_angles in num_angles_list:
            baseline_iters.setdefault((total_num_angles), []).append(selected_iters[k])
            baseline_psnrs.setdefault((total_num_angles), []).append(selected_psnrs[k])
            k += 1

    return baseline_iters, baseline_psnrs

def get_bed_iters_and_psnrs_partial_run(path, mode, init_num_angles, total_num_acq_projs, acq_projs_batch_size, reconstruct_every_k_step, log_dirs_end_with='DIP+TV'):
    # returns a list of all (completed) images found in this run, without checking cfg.skip_first_images,
    # so element 0 of the returned list will be the image with index `bed_cfg.get('skip_first_images', 0)`
    bed_iters = {}
    bed_psnrs = {}
    num_batches = ceil(total_num_acq_projs / acq_projs_batch_size)
    num_angles_list = list(range(
            init_num_angles + acq_projs_batch_size * reconstruct_every_k_step,
            init_num_angles + num_batches * acq_projs_batch_size + 1,
            acq_projs_batch_size * reconstruct_every_k_step))

    psnrs, best_psnrs = load_or_convert_logs(path, log_dirs_end_with=log_dirs_end_with)

    completed_images = len(psnrs) // len(num_angles_list)
    psnrs = psnrs[:completed_images*len(num_angles_list)]
    best_psnrs = best_psnrs[:completed_images*len(num_angles_list)]
    assert len(psnrs) % len(num_angles_list) == 0
    psnrs_per_sample = []
    best_psnrs_per_sample = []
    for j in range(0, len(psnrs), len(num_angles_list)):
        psnrs_per_sample.append(psnrs[j:j+len(num_angles_list)])
        best_psnrs_per_sample.append(best_psnrs[j:j+len(num_angles_list)])
    assert len(psnrs_per_sample[-1]) == len(num_angles_list)

    for k, num_angles in enumerate(num_angles_list):
        best_psnrs_k = [best_psnrs_j[k] for best_psnrs_j in best_psnrs_per_sample]
        selected_iters = select_iters(best_psnrs_k, mode)
        selected_psnrs = [best_psnrs_j[k][selected_iters_j] for best_psnrs_j, selected_iters_j in zip(best_psnrs_per_sample, selected_iters)]
        bed_iters[(num_angles)] = selected_iters
        bed_psnrs[(num_angles)] = selected_psnrs

    return bed_iters, bed_psnrs

def get_bed_iters_and_psnrs(paths, bed_cfgs, mode, skip_first_images, num_images, init_num_angles, log_dirs_end_with='DIP+TV'):

    total_num_acq_projs = bed_cfgs[0].bed.total_num_acq_projs
    acq_projs_batch_size = bed_cfgs[0].bed.acq_projs_batch_size
    reconstruct_every_k_step = bed_cfgs[0].bed.reconstruct_every_k_step

    for bed_cfg in bed_cfgs:
        # consistency across multiple runs
        assert bed_cfg.bed.total_num_acq_projs == bed_cfgs[0].bed.total_num_acq_projs
        assert bed_cfg.bed.acq_projs_batch_size == bed_cfgs[0].bed.acq_projs_batch_size
        assert bed_cfg.bed.reconstruct_every_k_step == bed_cfgs[0].bed.reconstruct_every_k_step

    num_batches = ceil(total_num_acq_projs / acq_projs_batch_size)
    num_angles_list = list(range(
            init_num_angles + acq_projs_batch_size * reconstruct_every_k_step,
            init_num_angles + num_batches * acq_projs_batch_size + 1,
            acq_projs_batch_size * reconstruct_every_k_step))
    assert len(paths) == len(bed_cfgs)

    bed_iters = {num_angles: [None for _ in range(skip_first_images, num_images)] for num_angles in num_angles_list}
    bed_psnrs = {num_angles: [None for _ in range(skip_first_images, num_images)] for num_angles in num_angles_list}

    for path, bed_cfg in zip(paths, bed_cfgs):
        bed_iters_partial_run, bed_psnrs_partial_run = get_bed_iters_and_psnrs_partial_run(
                path, mode, init_num_angles, total_num_acq_projs, acq_projs_batch_size, reconstruct_every_k_step,
                log_dirs_end_with=log_dirs_end_with)
        start_point_of_partial_run = bed_cfg.get('skip_first_images', 0)
        for num_angles in num_angles_list:
            for i_in_partial_run, (bed_iters_partial_run_k_i, bed_psnrs_partial_run_k_i) in enumerate(zip(bed_iters_partial_run[num_angles], bed_psnrs_partial_run[num_angles])):
                i_abs = start_point_of_partial_run + i_in_partial_run
                if (i_abs in range(skip_first_images, num_images) and
                        bed_iters[num_angles][i_abs-skip_first_images] is None):  # first path takes precedence if there are multiple
                    bed_iters[num_angles][i_abs-skip_first_images] = bed_iters_partial_run_k_i
                    bed_psnrs[num_angles][i_abs-skip_first_images] = bed_psnrs_partial_run_k_i
    for num_angles in num_angles_list:
        if None in bed_psnrs[num_angles]:
            raise RuntimeError('Did not find image {:d} in any of the paths:\n  {}'.format(bed_psnrs[num_angles].index(None), paths))

    return bed_iters, bed_psnrs

def walk_dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from walk_dict(v)
        else:
            yield v

@hydra.main(config_path='../../../cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    if cfg.use_double:
        torch.set_default_tensor_type(torch.DoubleTensor)

    runs_yaml = os.path.join(get_original_cwd(), cfg.bed.psnr_eval.bed_runs_yaml)
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

    noise = cfg.bed.psnr_eval.noise
    # 0.05, 0.1

    recon_method = cfg.bed.psnr_eval.recon_method
    # 'dip', 'tv'

    priors = cfg.bed.psnr_eval.priors
    # 'mll_prior', 'mll_prior_refinement', 'g_prior', 'g_prior_refinement', 'linear_model_isotropic', 'linear_model_gp', 'random'
    # 'baseline'

    criterions = cfg.bed.psnr_eval.criterions
    # 'EIG', 'diagonal_EIG', 'var'

    if 'random' in priors:
        criterions.append('random')

    mode = cfg.bed.psnr_eval.mode
    assert mode in ['max', 'last']

    show_initial = False

    show_stddev = cfg.bed.psnr_eval.get('show_stddev', True)

    symbol_legend_kind = cfg.bed.psnr_eval.get('symbol_legend_kind', 'refinement')

    baseline_paths = runs[noise]['baseline'][recon_method]

    baseline_cfg_0 = OmegaConf.load(os.path.join(baseline_paths[0], '.hydra', 'config.yaml'))

    runs_init_yaml = os.path.join(get_original_cwd(), cfg.bed.psnr_eval.get('bed_runs_init_yaml', os.path.splitext(cfg.bed.psnr_eval.bed_runs_yaml)[0] + '_init' + os.path.splitext(cfg.bed.psnr_eval.bed_runs_yaml)[1]))
    with open(runs_init_yaml, 'r') as f:
        runs_init = yaml.safe_load(f)

    for run_path_list in walk_dict(runs_init):
        if run_path_list:
            for j, p in enumerate(run_path_list):
                assert isinstance(p, str)
                if p.find(':') != -1 and p.find(':') < p.find('/'):
                    # remove server spec
                    _, p = p.split(':', maxsplit=1)
                    run_path_list[j] = p

    # only support initial runs to stem from a single run that starts with sample 0
    assert len(runs_init[noise][recon_method]) == 1

    load_path = runs_init[noise][recon_method][0]

    # baseline's load_path should be consistent with the path from the initial run yaml
    assert len(runs_init[noise]['dip']) == 1
    assert os.path.basename(baseline_cfg_0.density.compute_single_predictive_cov_block.load_path.rstrip('/')) == os.path.basename(runs_init[noise]['dip'][0].rstrip('/'))

    initial_cfg = OmegaConf.load(os.path.join(load_path, '.hydra', 'config.yaml'))

    init_num_angles = ceil(initial_cfg.beam_num_angle / initial_cfg.angular_sub_sampling)
    beam_num_angle = initial_cfg.beam_num_angle
    angular_sub_sampling = initial_cfg.angular_sub_sampling

    assert cfg.num_images > 1  # prevent accidental usage of default (1 image only)

    log_dirs_end_with = {'dip': 'DIP+TV', 'tv': 'TVAdam'}[recon_method]

    if show_initial:
        require_num_iters = None
        if recon_method == 'dip':
            require_num_iters = initial_cfg.net.optim.iterations
        elif recon_method == 'tv':
            require_num_iters = initial_cfg.bed.tvadam.iterations
        initial_iters, initial_psnrs = get_initial_iters_and_psnrs(
                load_path, mode, cfg.get('skip_first_images', 0), cfg.num_images, init_num_angles,
                require_num_iters=require_num_iters,
                log_dirs_end_with=log_dirs_end_with)

    # only support baseline to stem from a single run that starts with sample 0
    assert len(baseline_paths) == 1
    assert baseline_cfg_0.get('skip_first_images', 0) == 0

    # consistency with initial_cfg
    assert baseline_cfg_0.beam_num_angle == beam_num_angle
    assert baseline_cfg_0.angular_sub_sampling == angular_sub_sampling

    baseline_iters, baseline_psnrs = get_baseline_iters_and_psnrs(
            baseline_paths[0], mode, cfg.get('skip_first_images', 0), cfg.num_images, beam_num_angle, init_num_angles,
            baseline_cfg_0.bed.total_num_acq_projs, baseline_cfg_0.bed.acq_projs_batch_size, baseline_cfg_0.bed.reconstruct_every_k_step,
            log_dirs_end_with=log_dirs_end_with)

    bed_iters_dict = {}
    bed_psnrs_dict = {}

    for prior in priors:
        bed_iters_dict[prior] = {}
        bed_psnrs_dict[prior] = {}
        for crit in criterions:
            if runs[noise][prior].get(crit, {}).get(recon_method, None) is not None:
                paths = runs[noise][prior][crit][recon_method]
                bed_cfgs = [OmegaConf.load(os.path.join(path, '.hydra', 'config.yaml')) for path in paths]

                for bed_cfg in bed_cfgs:
                    # consistency with initial_cfg
                    assert bed_cfg.noise_specs.stddev == noise
                    assert bed_cfg.beam_num_angle == beam_num_angle
                    assert bed_cfg.angular_sub_sampling == angular_sub_sampling
                    # consistency with baseline
                    assert os.path.basename(bed_cfg.density.compute_single_predictive_cov_block.load_path.rstrip('/')) == os.path.basename(baseline_cfg_0.density.compute_single_predictive_cov_block.load_path.rstrip('/'))

                bed_iters, bed_psnrs = get_bed_iters_and_psnrs(
                        paths, bed_cfgs, mode, cfg.get('skip_first_images', 0), cfg.num_images, init_num_angles,
                        log_dirs_end_with=log_dirs_end_with)
            else:
                bed_iters = None
                bed_psnrs = None

            bed_iters_dict[prior][crit] = bed_iters
            bed_psnrs_dict[prior][crit] = bed_psnrs

    mean_initial_psnrs = {
                (num_angles): np.mean(psnrs)
                for (num_angles), psnrs in initial_psnrs.items()} if show_initial else {}
    std_initial_psnrs = {
                (num_angles): np.std(psnrs) / np.sqrt(cfg.num_images - cfg.get('skip_first_images', 0))
                for (num_angles), psnrs in initial_psnrs.items()} if show_initial else {}
    mean_baseline_psnrs = {
            (num_angles): np.mean(psnrs)
            for (num_angles), psnrs in baseline_psnrs.items()}
    std_baseline_psnrs = {
            (num_angles): np.std(psnrs) / np.sqrt(cfg.num_images - cfg.get('skip_first_images', 0))
            for (num_angles), psnrs in baseline_psnrs.items()}
    mean_baseline_psnrs = { **mean_initial_psnrs, **mean_baseline_psnrs}
    std_baseline_psnrs = { **std_initial_psnrs, **std_baseline_psnrs}

    prior_title_dict = {
        'mll_prior': 'linear DIP',
        'mll_prior_refinement': 'linear DIP' if symbol_legend_kind == 'refinement' else 'linear DIP w/ refinement',
        'g_prior': 'linear DIP (g-prior)',
        'g_prior_refinement': 'linear DIP (g-prior)' if symbol_legend_kind == 'refinement' else 'linear DIP (g-prior) w/ refinement',
        'linear_model_isotropic': 'isotropic',
        'linear_model_gp': 'Matern-½ process',
        'random': 'random',
    }

    criterion_title_dict = {
        'EIG': 'EIG',
        'diagonal_EIG': 'diagonal EIG',
        'var': 'ESE',
        'random': 'random baseline',
        'baseline': 'equidistant baseline',
    }

    prior_color_dict = {
        'mll_prior': 'tab:orange',
        'mll_prior_refinement': 'tab:orange',  # 'tab:red',
        'g_prior': 'tab:blue',
        'g_prior_refinement': 'tab:blue',  # 'tab:cyan',
        'linear_model_isotropic': 'tab:green',
        'linear_model_gp': 'tab:olive',
        'random': 'black',
    }

    bed_psnrs_list = []
    bed_prior_list = []
    bed_criterion_list = []
    bed_plot_kwargs_list = []
    for prior in priors:
        for crit in criterions:
            if bed_psnrs_dict[prior][crit] is not None:
                bed_psnrs_list.append(bed_psnrs_dict[prior][crit])
                bed_prior_list.append(prior)
                bed_criterion_list.append(crit)
                bed_plot_kwargs_list.append(
                    {
                        'linestyle': (
                                (('dashed' if 'refinement' in prior else 'solid') if crit != 'random' else 'dotted')
                                if symbol_legend_kind == 'refinement' else
                                {'EIG': 'dashed', 'diagonal_EIG': 'dashdot', 'var': 'solid', 'random': 'dotted'}[crit]),
                        'color': prior_color_dict[prior],
                        'label': prior_title_dict[prior] + ' ' + criterion_title_dict[crit],
                    })
    

    mean_bed_psnrs_list = []
    std_bed_psnrs_list = []
    for bed_psnrs in bed_psnrs_list:
        mean_bed_psnrs = {
                (num_angles): np.mean(psnrs)
                for (num_angles), psnrs in bed_psnrs.items()}
        std_bed_psnrs = {
                (num_angles): np.std(psnrs) / np.sqrt(cfg.num_images - cfg.get('skip_first_images', 0))
                for (num_angles), psnrs in bed_psnrs.items()}
        mean_bed_psnrs_list.append(mean_bed_psnrs)
        std_bed_psnrs_list.append(std_bed_psnrs)

    for j, bed_psnrs in enumerate(bed_psnrs_list):
        print('Evaluation for BED path {:d} ({}):'.format(j, bed_prior_list[j] + ' ' + bed_criterion_list[j]))
        if cfg.get('skip_first_images', 0) == 0:
            thresh = 0.5
            angle = 10
            idx = ffinf_success_exps(bed_psnrs, baseline_psnrs,  condition = lambda e: e > thresh, thresh=thresh, angle=angle)
            print('samples idx {} reporting {} dB gain in PSNR using {} angles'.format(idx, thresh, angle))
            idx = ffinf_success_exps(bed_psnrs, baseline_psnrs, condition = lambda e: e < -thresh, thresh=thresh, angle=angle)
            print('unsuccessful acquisitions idx {}'.format(idx))

    fig, ax = plt.subplots(figsize=(6, 4))
    baseline_x = np.asarray(list(mean_baseline_psnrs.keys()))
    baseline_y_mean = np.asarray(list(mean_baseline_psnrs.values()))
    baseline_y_std = np.asarray(list(std_baseline_psnrs.values()))
    handle_baseline, = ax.plot(baseline_x, baseline_y_mean, color='black', linewidth=1.5)
    if show_stddev:
        ax.fill_between(baseline_x, baseline_y_mean - baseline_y_std, baseline_y_mean + baseline_y_std, color='black', alpha=0.1)
    handles_list = []
    for j, mean_bed_psnrs in enumerate(mean_bed_psnrs_list):
        bed_x = np.asarray(list(mean_bed_psnrs.keys()))
        bed_y_mean = np.asarray(list(mean_bed_psnrs.values()))
        bed_y_std = np.asarray(list(std_bed_psnrs.values()))
        h, = ax.plot(bed_x, bed_y_mean, linewidth=1.5, **bed_plot_kwargs_list[j])
        if show_stddev:
            ax.fill_between(bed_x, bed_y_mean - bed_y_std, bed_y_mean + bed_y_std, **bed_plot_kwargs_list[j], alpha=0.1)
        handles_list.append(h)
    ax.set_ylabel('PSNR [dB]')
    ax.set_xlabel('#total acquired angles')
    prior_handles = []
    prior_labels = []
    for prior in priors:
        if 'refinement' in prior and prior.replace('_refinement', '') in priors:
            continue
        if prior in bed_prior_list and prior != 'random':
            idx_in_list = bed_prior_list.index(prior)
            h = deepcopy(handles_list[idx_in_list])
            h.set_linestyle('solid')
            prior_handles.append(h)
            prior_labels.append(prior_title_dict[prior])
    prior_handles.append(handle_baseline)
    prior_labels.append(criterion_title_dict['baseline'])
    if 'random' in bed_criterion_list:
        idx_in_list = bed_criterion_list.index('random')
        prior_handles.append(handles_list[idx_in_list])
        prior_labels.append(criterion_title_dict['random'])
    prior_legend_kwargs = {}
    # if not show_initial:
    prior_legend_kwargs['loc'] = 'upper center'
    prior_legend_kwargs['bbox_to_anchor'] = (0.5, -0.15)
    prior_legend_kwargs['ncol'] = 6  # 3 if len(prior_handles) != 4 else 2
    prior_legend = ax.legend(prior_handles, prior_labels, **prior_legend_kwargs)
    ax.add_artist(prior_legend)
    symbol_handles = []
    symbol_labels = []
    if symbol_legend_kind == 'refinement':
        for refinement in [False, True]:
            idx_in_list = next((i for i, prior in enumerate(bed_prior_list) if ('refinement' in prior) == refinement), None)
            if idx_in_list is not None:
                h = deepcopy(handles_list[idx_in_list])
                h.set_color('gray')
                symbol_handles.append(h)
                symbol_labels.append('DIP retrained every 5 angles' if refinement else 'DIP trained on pilot scan ($\mathcal{B}^{(0)}$)')
    elif symbol_legend_kind == 'criterion':
        for crit in criterions:
            if crit in bed_criterion_list:
                if crit == 'random':
                    continue
                idx_in_list = bed_criterion_list.index(crit)
                h = deepcopy(handles_list[idx_in_list])
                h.set_color('gray')
                symbol_handles.append(h)
                symbol_labels.append(criterion_title_dict[crit])
    symbol_legend_kwargs = {}
    # if not show_initial:
    symbol_legend_kwargs['loc'] = 'lower right'
    symbol_legend_kwargs['bbox_to_anchor'] = (0.95, 0.05)
    contains_dip_prior = any('refinement' in prior for prior in bed_prior_list)
    if not (symbol_legend_kind == 'refinement' and not contains_dip_prior):
        symbol_legend = ax.legend(symbol_handles, symbol_labels, **symbol_legend_kwargs)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3)
    title = ''
    title += {'dip': 'DIP reconstructions', 'tv': 'TV reconstructions'}[recon_method]
    is_all_EIG = all(crit == 'EIG' or crit == 'random' for crit in bed_criterion_list)
    is_all_diagonal_EIG = all(crit == 'diagonal_EIG' or crit == 'random' for crit in bed_criterion_list)
    is_all_var = all(crit == 'var' or crit == 'random' for crit in bed_criterion_list)
    if is_all_EIG:
        title += ' — angle selection by EIG'
    if is_all_diagonal_EIG:
        title += ' — angle selection by diagonal EIG'
    if is_all_var:
        title += ' — angle selection by ESE'
    ax.set_title(title)
    suffix = ''
    if any('refinement' in prior for prior in bed_prior_list):
        suffix += '_refinement'
    suffix += '_EIG' if is_all_EIG else ('_diagonal_EIG' if is_all_diagonal_EIG else ('_var' if is_all_var else '_multi_crit'))
    suffix += '_no_stderr' if not show_stddev else ''
    fig.savefig('./psnr_eval_{}_{}{}.pdf'.format(noise, recon_method, suffix), bbox_inches='tight', pad_inches=0.)
    fig.savefig('./psnr_eval_{}_{}{}.png'.format(noise, recon_method, suffix), bbox_inches='tight', pad_inches=0., dpi=600)

if __name__ == '__main__':
    coordinator()
