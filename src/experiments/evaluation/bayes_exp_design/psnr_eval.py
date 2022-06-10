import os
from warnings import warn
from math import ceil
import datetime
import numpy as np
import hydra
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
from dataset.mnist import simulate
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

def load_or_convert_logs(path, require_num_steps=None, max_logs=None):
    log_files = []
    def filter_child(d):
        if not (d.endswith('DIP+TV') and os.path.isdir(os.path.join(path, d))):
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

@hydra.main(config_path='../../../cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    if cfg.use_double:
        torch.set_default_tensor_type(torch.DoubleTensor)

    baseline_path = cfg.bed.psnr_eval.baseline_path

    bed_path_list = cfg.bed.psnr_eval.bed_path
    if isinstance(bed_path_list, str):
        bed_path_list = [bed_path_list]

    mode = cfg.bed.psnr_eval.mode
    assert mode in ['max', 'last']

    bed_title_list = cfg.bed.psnr_eval.get('bed_title', ['' for _ in bed_path_list])
    if isinstance(bed_title_list, str):
        bed_title_list = [bed_title_list]

    baseline_cfg = OmegaConf.load(os.path.join(baseline_path, '.hydra', 'config.yaml'))

    initial_cfg = OmegaConf.load(os.path.join(
            baseline_cfg.density.compute_single_predictive_cov_block.load_path, '.hydra', 'config.yaml'))

    # assert baseline_cfg.num_images == initial_cfg.num_images
    assert baseline_cfg.noise_specs.stddev == initial_cfg.noise_specs.stddev 
    assert baseline_cfg.beam_num_angle == initial_cfg.beam_num_angle
    assert baseline_cfg.angular_sub_sampling == initial_cfg.angular_sub_sampling

    psnrs, best_psnrs = load_or_convert_logs(
            baseline_cfg.density.compute_single_predictive_cov_block.load_path,
            max_logs = cfg.num_images,
            require_num_steps=initial_cfg.net.optim.iterations
            )
    psnrs = psnrs[cfg.get('skip_first_images', 0):cfg.num_images]
    best_psnrs = best_psnrs[cfg.get('skip_first_images', 0):cfg.num_images]

    if mode == 'max':
        selected_iters = [np.argmax(p) for p in best_psnrs]
    elif mode == 'last':
        selected_iters = [(len(p) - 1) for p in best_psnrs]
    else:
        raise ValueError
    selected_psnrs = [p[m] for p, m in zip(best_psnrs, selected_iters)]
    
    initial_iters = {}
    initial_psnrs = {}
    for i in range(cfg.num_images - cfg.get('skip_first_images', 0)):
        initial_iters.setdefault((
            ceil(initial_cfg.beam_num_angle / initial_cfg.angular_sub_sampling)), []).append(selected_iters[i])
        initial_psnrs.setdefault((
            ceil(initial_cfg.beam_num_angle / initial_cfg.angular_sub_sampling)), []).append(selected_psnrs[i])

    baseline_iters = {}
    baseline_psnrs = {}    
    init_num_angles = ceil(baseline_cfg.beam_num_angle / baseline_cfg.angular_sub_sampling)
    num_batches = ceil(baseline_cfg.bed.total_num_acq_projs / baseline_cfg.bed.acq_projs_batch_size)
    num_angles_list = list(range(
            init_num_angles + baseline_cfg.bed.acq_projs_batch_size * baseline_cfg.bed.reconstruct_every_k_step,
            init_num_angles + num_batches * baseline_cfg.bed.acq_projs_batch_size + 1,
            baseline_cfg.bed.acq_projs_batch_size * baseline_cfg.bed.reconstruct_every_k_step))
    num_angles_list = [angle for angle in num_angles_list if (
        baseline_cfg.beam_num_angle % angle == 0 
            and angle % ceil(baseline_cfg.beam_num_angle / baseline_cfg.angular_sub_sampling) == 0)
        ]
    psnrs, best_psnrs = load_or_convert_logs(
        cfg.bed.psnr_eval.baseline_path,
        max_logs = cfg.num_images * len(num_angles_list),
    )
    psnrs = psnrs[cfg.get('skip_first_images', 0) * len(num_angles_list):cfg.num_images * len(num_angles_list)]
    best_psnrs = best_psnrs[cfg.get('skip_first_images', 0) * len(num_angles_list):cfg.num_images * len(num_angles_list)]

    if mode == 'max':
        selected_iters = [np.argmax(p) for p in best_psnrs]
    elif mode == 'last':
        selected_iters = [(len(p) - 1) for p in best_psnrs]
    else:
        raise ValueError
    selected_psnrs = [p[m] for p, m in zip(best_psnrs, selected_iters)]
    k = 0
    for _ in range(cfg.num_images - cfg.get('skip_first_images', 0)): 
        for total_num_angles in num_angles_list:
            baseline_iters.setdefault((total_num_angles), []).append(selected_iters[k])
            baseline_psnrs.setdefault((total_num_angles), []).append(selected_psnrs[k])
            k += 1

    bed_iters_list = []
    bed_psnrs_list = []

    bed_plot_kwargs_list = []

    for bed_path, bed_title in zip(bed_path_list, bed_title_list):
        bed_cfg = OmegaConf.load(os.path.join(bed_path, '.hydra', 'config.yaml'))

        # assert baseline_cfg.density.compute_single_predictive_cov_block.load_path == bed_cfg.density.compute_single_predictive_cov_block.load_path
        # assert cfg.get('skip_first_images', 0) < cfg.num_images
        
        # assert baseline_cfg.num_images == bed_cfg.num_images == initial_cfg.num_images
        assert baseline_cfg.noise_specs.stddev == bed_cfg.noise_specs.stddev
        assert baseline_cfg.beam_num_angle == bed_cfg.beam_num_angle
        assert baseline_cfg.angular_sub_sampling == bed_cfg.angular_sub_sampling

        bed_iters = {}
        bed_psnrs = {}
        init_num_angles = ceil(bed_cfg.beam_num_angle / bed_cfg.angular_sub_sampling)
        num_batches = ceil(bed_cfg.bed.total_num_acq_projs / bed_cfg.bed.acq_projs_batch_size)
        num_angles_list = list(range(
                init_num_angles + bed_cfg.bed.acq_projs_batch_size * bed_cfg.bed.reconstruct_every_k_step,
                init_num_angles + num_batches * bed_cfg.bed.acq_projs_batch_size + 1,
                bed_cfg.bed.acq_projs_batch_size * bed_cfg.bed.reconstruct_every_k_step))

        psnrs, best_psnrs = load_or_convert_logs(bed_path, max_logs=cfg.num_images*len(num_angles_list))

        assert len(psnrs) % len(num_angles_list) == 0
        psnrs_per_sample = []
        best_psnrs_per_sample = []
        for j in range(0, len(psnrs), len(num_angles_list)):
            psnrs_per_sample.append(psnrs[j:j+len(num_angles_list)])
            best_psnrs_per_sample.append(best_psnrs[j:j+len(num_angles_list)])
        assert len(psnrs_per_sample[-1]) == len(num_angles_list)

        psnrs_per_sample = psnrs_per_sample[cfg.get('skip_first_images', 0):cfg.num_images]
        best_psnrs_per_sample = best_psnrs_per_sample[cfg.get('skip_first_images', 0):cfg.num_images]
        
        for k, num_angles in enumerate(num_angles_list):
            if mode == 'max':
                selected_iters = [np.argmax(best_psnrs_j[k]) for best_psnrs_j in best_psnrs_per_sample]
            elif mode == 'last':
                selected_iters = [(len(best_psnrs_j[k]) - 1) for best_psnrs_j in best_psnrs_per_sample]
            else:
                raise ValueError
            selected_psnrs = [best_psnrs_j[k][selected_iters_j] for best_psnrs_j, selected_iters_j in zip(best_psnrs_per_sample, selected_iters)]
            bed_iters[(num_angles)] = selected_iters
            bed_psnrs[(num_angles)] = selected_psnrs

        # adapt if needed
        bed_plot_kwargs = {}
        if bed_title:
            bed_plot_kwargs['label'] = bed_title
        if bed_cfg.bed.use_objective_prior:
            if bed_cfg.bed.update_network_params:
                bed_plot_kwargs.setdefault('label', 'G prior w/ refinement')
                bed_plot_kwargs['color'] = 'tab:blue'
                bed_plot_kwargs['linestyle'] = 'dashdot'
            else:
                bed_plot_kwargs.setdefault('label', 'G prior')
                bed_plot_kwargs['color'] = 'tab:blue'
                bed_plot_kwargs['linestyle'] = 'solid'
        else:
            if bed_cfg.bed.update_network_params:
                bed_plot_kwargs.setdefault('label', 'MLL prior w/ refinement')
                bed_plot_kwargs['color'] = 'tab:orange'
                bed_plot_kwargs['linestyle'] = 'dashdot'
            else:
                bed_plot_kwargs.setdefault('label', 'MLL prior')
                bed_plot_kwargs['color'] = 'tab:orange'
                bed_plot_kwargs['linestyle'] = 'solid'

        bed_iters_list.append(bed_iters)
        bed_psnrs_list.append(bed_psnrs)

        bed_plot_kwargs_list.append(bed_plot_kwargs)

    mean_initial_psnrs = {
                (num_angles): np.mean(psnrs)
                for (num_angles), psnrs in initial_psnrs.items()}
    mean_baseline_psnrs = {
            (num_angles): np.mean(psnrs)
            for (num_angles), psnrs in baseline_psnrs.items()}
    mean_baseline_psnrs = { **mean_initial_psnrs, **mean_baseline_psnrs}

    mean_bed_psnrs_list = []

    for bed_psnrs in bed_psnrs_list:
        mean_bed_psnrs = {
                (num_angles): np.mean(psnrs)
                for (num_angles), psnrs in bed_psnrs.items()}
        mean_bed_psnrs_list.append(mean_bed_psnrs)

    for j, bed_psnrs in enumerate(bed_psnrs_list):
        print('Evaluation for BED path {:d} ({}):'.format(j, bed_title_list[j] + ', ' + bed_path_list[j] if bed_title_list[j] else bed_path_list[j]))
        if cfg.get('skip_first_images', 0) == 0:
            thresh = 0.5
            angle = 10
            idx = ffinf_success_exps(bed_psnrs, baseline_psnrs,  condition = lambda e: e > thresh, thresh=thresh, angle=angle)
            print('samples idx {} reporting {} dB gain in PSNR using {} angles'.format(idx, thresh, angle))
            idx = ffinf_success_exps(bed_psnrs, baseline_psnrs, condition = lambda e: e < -thresh, thresh=thresh, angle=angle)
            print('unsuccessful acquisitions idx {}'.format(idx))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(mean_baseline_psnrs.keys(), mean_baseline_psnrs.values(), color='gray')
    for j, mean_bed_psnrs in enumerate(mean_bed_psnrs_list):
        ax.plot(mean_bed_psnrs.keys(), mean_bed_psnrs.values(), **bed_plot_kwargs_list[j])
    ax.set_ylabel('PSNR [dB]')
    ax.set_xlabel('#directions')
    ax.legend()
    fig.savefig('./psnr_eval_{}.pdf'.format(initial_cfg.noise_specs.stddev), bbox_inches='tight', pad_inches=0.)
    fig.savefig('./psnr_eval_{}.png'.format(initial_cfg.noise_specs.stddev), bbox_inches='tight', pad_inches=0., dpi=600)

if __name__ == '__main__':
    coordinator()
