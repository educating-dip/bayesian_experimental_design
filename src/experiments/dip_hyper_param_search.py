import os
from itertools import islice
import hydra
from omegaconf import DictConfig
from dataset.utils import (
        get_standard_ray_trafos,
        load_trainset_MNIST_dataset, load_trainset_KMNIST_dataset,
        load_trainset_rectangles_dataset,
        )
from dataset.mnist import simulate
import torch
from deep_image_prior import DeepImagePriorReconstructor
from deep_image_prior.utils import PSNR, SSIM

@hydra.main(config_path='../cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    ray_trafos = get_standard_ray_trafos(cfg, return_torch_module=True)

    ray_trafo = {'ray_trafo_module': ray_trafos['ray_trafo_module'],
                 'reco_space': ray_trafos['space'],
                 }

    # use training images from (k)mnist for hyper parameter search
    # data: observation, filtbackproj, example_image
    if cfg.name == 'mnist':
        loader = load_trainset_MNIST_dataset()
    elif cfg.name == 'kmnist':
        loader = load_trainset_KMNIST_dataset()
    elif cfg.name == 'rectangles':
        loader = load_trainset_rectangles_dataset(cfg)
    else:
        raise NotImplementedError

    for i, data_sample in enumerate(islice(loader, cfg.num_images)):
        if i < cfg.get('skip_first_images', 0):
            continue

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed + i + 10000)  # for reproducible noise in simulate

        if cfg.name in ['mnist', 'kmnist', 'rectangles']:
            example_image = data_sample[0] if cfg.name in ['mnist', 'kmnist'] else data_sample
            ray_trafos['ray_trafo_module'].to(example_image.device)
            ray_trafos['ray_trafo_module_adj'].to(example_image.device)
            observation, filtbackproj, example_image = simulate(
                example_image,
                ray_trafos,
                cfg.noise_specs
                )
        else:
            raise NotImplementedError

        reconstructor = DeepImagePriorReconstructor(**ray_trafo, cfg=cfg.net)

        ray_trafos['ray_trafo_module'].to(reconstructor.device)
        ray_trafos['ray_trafo_module_adj'].to(reconstructor.device)

        recon, _ = reconstructor.reconstruct(
                observation, fbp=filtbackproj.to(reconstructor.device), ground_truth=example_image.to(reconstructor.device))

        torch.save(reconstructor.model.state_dict(),
                './dip_model_{}.pt'.format(i))

        print('DIP reconstruction of sample {:d}'.format(i))
        print('PSNR:', PSNR(recon, example_image[0, 0].cpu().numpy()))
        print('SSIM:', SSIM(recon, example_image[0, 0].cpu().numpy()))

if __name__ == '__main__':
    coordinator()
