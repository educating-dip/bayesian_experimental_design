import os
from math import ceil
import numpy as np
from tqdm import tqdm
from warnings import warn
import hydra
from omegaconf import OmegaConf
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

def load_or_convert_logs(path, require_num_steps=None):
    log_files = []
    for d in sorted(os.listdir(path)):
        path_d = os.path.join(path, d)
        if os.path.isdir(path_d):
            logs_in_d = [f for f in os.listdir(path_d) if f.startswith('events.') and not f.endswith('.npz')]
            assert len(logs_in_d) == 1
            if os.path.getsize(os.path.join(path_d, logs_in_d[0])) > 40:
                log_files.append(os.path.join(path_d, logs_in_d[0]))
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


### rectangles
# {
#   (5, 0.05): (1e-2, 60000),
#   (5, 0.1): (1e-2, 60000),
#   (10, 0.05): (3e-3, 30000),
#   (10, 0.1): (1e-2, 30000),
#   (20, 0.05): (3e-3, 10000),
#   (20, 0.1): (1e-2, 10000),
#   (40, 0.05): (3e-3, 10000),
#   (40, 0.1): (3e-3, 10000),
# }

if __name__ == '__main__':

    cfg_list = []
    psnrs_list = []
    best_psnrs_list = []

    name = 'rectangles'
    angles = 40
    suffix = '_{}'.format(angles) if name == 'rectangles' else ''
    noise = 0.05

    title = '{}: angles={}, noise={}'.format(name, angles, noise)

    with open('/localdata/jleuschn/experiments/dip_bayesian_ext/{}_tvadam{}_hyper_param_search_experiments.txt'.format(name, suffix)) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            path = line.split('uni-bremen.de:22')[1]
            print(path)
            try:
                cfg = OmegaConf.load(os.path.join(path, '.hydra', 'config.yaml'))
            except FileNotFoundError:
                print('skipping', path)
                continue
            if cfg.name == name and ceil(cfg.beam_num_angle / cfg.angular_sub_sampling) == angles and cfg.noise_specs.stddev == noise:
                psnrs, best_psnrs = load_or_convert_logs(os.path.join(path, 'runs'), require_num_steps=cfg.bed.tvadam.iterations)
                if len(psnrs) < 10:
                    print('skipping', path)
                    continue
                if len(psnrs[-1]) < len(psnrs[0]):
                    psnrs = psnrs[:-1]
                    best_psnrs = best_psnrs[:-1]
                    cfgs = cfgs[:-1]
                    assert all(len(p) == len(psnrs[0]) for p in psnrs)
                    assert all(len(p) == len(psnrs[0]) for p in psnrs)
                if len(psnrs) < cfg.num_images:
                    print('using only {} samples for gamma={}'.format(len(psnrs), cfg.bed.tvadam.gamma))
                print(cfg.bed.tvadam.gamma)
                cfg_list.append(cfg)
                psnrs_list.append(psnrs)
                best_psnrs_list.append(best_psnrs)

    cfg_list, psnrs_list, best_psnrs_list = map(list, zip(*sorted(zip(cfg_list, psnrs_list, best_psnrs_list), key=lambda x: x[0].bed.tvadam.gamma)))
    gammas = [cfg.bed.tvadam.gamma for cfg in cfg_list]

    # for x in psnrs_list + best_psnrs_list:
    #     del x[41:]

    max_mean_psnr_list = [np.max(np.mean(psnrs, axis=0)) for psnrs in psnrs_list]
    max_idx = np.argmax(max_mean_psnr_list)
    print(title)
    for i, (cfg, max_mean_psnr) in enumerate(zip(cfg_list, max_mean_psnr_list)):
        print('gamma {}: max mean psnr {:.2f}'.format(cfg.bed.tvadam.gamma, max_mean_psnr) + (' [max at {} iterations]'.format(np.argmax(np.mean(psnrs_list[i], axis=0))) if i == max_idx else '') + (' [incomplete {}/{}]'.format(len(psnrs_list[i]), cfg.num_images) if len(psnrs_list[i]) < cfg.num_images else ''))

    for cfg, psnrs in zip(cfg_list, psnrs_list):
        print('maximum for gamma {} at {} iterations'.format(cfg.bed.tvadam.gamma, np.argmax(np.mean(psnrs, axis=0))))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, ax = plt.subplots()
    ax.set_title(title)
    for i, (cfg, psnrs, best_psnrs) in enumerate(zip(cfg_list, psnrs_list, best_psnrs_list)):
        gamma = cfg.bed.tvadam.gamma
        color = colors[gammas.index(gamma)]
        label = 'gamma={}'.format(gamma) + (' [max]' if i == max_idx else '') + (' [incomplete {}/{}]'.format(len(psnrs), cfg.num_images) if len(psnrs) < cfg.num_images else '')
        ax.plot(np.mean(psnrs, axis=0), color=color, label=label)
        # for psnr_history in psnrs:
        #     ax.plot(psnr_history, color=color, label=label)
    plt.legend()
    plt.show()

    # fig, ax = plt.subplots()
    # for j, psnr in enumerate(psnrs):
    #     ax.plot(psnr, label=str(j))
    # plt.legend()
    # plt.show()
