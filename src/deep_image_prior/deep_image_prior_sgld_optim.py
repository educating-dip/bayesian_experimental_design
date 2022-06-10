import os
import torch
import numpy as np
from math import ceil
from torch.nn import MSELoss
from hydra.utils import get_original_cwd
from tqdm import tqdm
from warnings import warn 
from .utils import tv_loss, PSNR, normalize
from copy import deepcopy
from .deep_image_prior import DeepImagePriorReconstructor
from torch.optim.optimizer import Optimizer, required

class SGLD(Optimizer):
    """
    SGLD optimiser based on pytorch's SGD.
    Note that the weight decay is specified in terms of the gaussian prior sigma.
    """

    def __init__(self, params, lr=required, weight_decay=0, noisy_lr=1e-8):

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, weight_decay=weight_decay, noisy_lr=noisy_lr)

        super(SGLD, self).__init__(params, defaults)

    def step(self, add_noise=False):
        """
        Performs a single optimization step.
        """
        loss = None

        for group in self.param_groups:

            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                if add_noise:
                    langevin_noise = p.data.new(p.data.size()).normal_(mean=0, std=1) / np.sqrt(group['lr'])
                    p.data.add_(-group['noisy_lr'],
                                0.5 * d_p + langevin_noise)
                else:
                    p.data.add_(-group['lr'], 0.5 * d_p)

        return loss

class DeepImagePriorReconstructorSGLD(DeepImagePriorReconstructor):
   
    def reconstruct(self, noisy_observation, fbp=None, ground_truth=None, use_init_model=True, use_tv_loss=True, num_burn_in_steps=10000):

        if self.cfg.torch_manual_seed:
            torch.random.manual_seed(self.cfg.torch_manual_seed)

        if use_init_model: 
            self.init_model()

        if self.cfg.load_pretrain_model:
            path = os.path.join(
                get_original_cwd(),
                self.cfg.learned_params_path if self.cfg.learned_params_path.endswith('.pt') \
                    else self.cfg.learned_params_path + '.pt')
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        else:
            self.model.to(self.device)

        self.model.train()

        if self.cfg.recon_from_randn:
            self.net_input = 0.1 * \
                torch.randn(1, *self.reco_space.shape)[None].to(self.device)
        else:
            self.net_input = fbp.to(self.device)

        self.init_optimizer()
        self.optimizer = SGLD(self.model.parameters(), lr=self.cfg.optim.lr,  weight_decay=1e-6, noisy_lr=1e-7)

        y_delta = noisy_observation.to(self.device)

        if self.cfg.optim.loss_function == 'mse':
            criterion = MSELoss(reduction='sum')
        else:
            warn('Unknown loss function, falling back to MSE')
            criterion = MSELoss(reduction='sum')

        best_loss = np.inf
        model_out = self.model(self.net_input)
        best_output, pre_activation_best_output = model_out[0].detach(), model_out[1].detach()
        best_params_state_dict = deepcopy(self.model.state_dict())

        sample_recon = []
        with tqdm(range(self.cfg.optim.iterations), desc='DIP', disable= not self.cfg.show_pbar) as pbar:
            for i in pbar:
                self.optimizer.zero_grad()
                model_out = self.model(self.net_input)
                output, pre_activation_output = model_out[0], model_out[1].detach()
                loss = criterion(self.ray_trafo_module(output), y_delta)
                if use_tv_loss: 
                    loss = loss + self.cfg.optim.gamma * tv_loss(output)
                loss.backward()

                if loss.item() < best_loss:
                    best_params_state_dict = deepcopy(self.model.state_dict())
                self.optimizer.step(add_noise = False if i < num_burn_in_steps else True)

                for p in self.model.parameters():
                    p.data.clamp_(-1000, 1000) # MIN,MAX

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_output = output.detach()
                    pre_activation_best_output = pre_activation_output

                if ground_truth is not None:
                    best_output_psnr = PSNR(best_output.detach().cpu(), ground_truth.cpu())
                    output_psnr = PSNR(output.detach().cpu(), ground_truth.cpu())
                    pbar.set_description('DIP output_psnr={:.1f}'.format(output_psnr), refresh=False)
                    self.writer.add_scalar('best_output_psnr', best_output_psnr, i)
                    self.writer.add_scalar('output_psnr', output_psnr, i)

                self.writer.add_scalar('loss', loss.item(),  i)
                if i % 1000 == 0:
                    self.writer.add_image('reco', normalize(best_output[0, ...]).cpu().numpy(), i)
            
                if (i > num_burn_in_steps):
                    sample_recon.append(output.detach().cpu())
        
        self.model.load_state_dict(best_params_state_dict)
        self.writer.close()

        return best_output[0, 0, ...].cpu().numpy(), pre_activation_best_output[0, 0, ...].cpu().numpy(), torch.cat(sample_recon, dim=0)