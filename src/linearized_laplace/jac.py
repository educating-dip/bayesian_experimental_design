import torch
from collections.abc import Iterable

def compute_jacobian_single_batch(
    input,
    model,
    modules,
    out_dim,
    return_on_cpu=False,
    ):

    jac = []
    model.eval()
    f = model(input)[0].view(-1)
    for o in range(out_dim):
        f_o = f[o]
        model.zero_grad()
        f_o.backward(retain_graph=True)
        jacs_o = agregate_flatten_weight_grad(modules).detach()
        jac.append(jacs_o)
    return (torch.stack(jac,
            dim=0) if not return_on_cpu else torch.stack(jac,
            dim=0).cpu())

def agregate_flatten_weight_grad(modules):
    grads_o = []
    for layer in modules:
        assert isinstance(layer, torch.nn.Conv2d)
        grads_o.append(layer.weight.grad.flatten())
    return torch.cat(grads_o)
