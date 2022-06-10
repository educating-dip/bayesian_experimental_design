import torch

def simulate(x, ray_trafos, cfg, return_numpy=False):

    x = (x - x.min()) / (x.max() - x.min())
    observation = ray_trafos['ray_trafo_module'](x)
    relative_stddev = torch.mean(torch.abs(observation))
    noisy_observation = observation \
        + torch.zeros(*observation.shape).normal_(0, 1) \
        * relative_stddev * cfg.stddev
    filtbackproj = ray_trafos['pseudoinverse_module'](noisy_observation)
    return (noisy_observation, filtbackproj,
            x) if not return_numpy else (noisy_observation.squeeze().numpy(),
            filtbackproj.squeeze().numpy(), x.squeeze().numpy())
