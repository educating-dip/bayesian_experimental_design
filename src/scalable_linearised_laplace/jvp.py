import numpy as np
import torch
import torch.autograd.forward_ad as fwAD
from tqdm import tqdm
from linearized_laplace import compute_jacobian_single_batch, agregate_flatten_weight_grad
from scalable_linearised_laplace.batch_ensemble import Conv2dBatchEnsemble
from scalable_linearised_laplace.fwAD import get_fwAD_model
from scalable_linearised_laplace.conv2d_fwAD import Conv2dFwAD

class FwAD_JvP_PreserveAndRevertWeightsToParameters(object):
    def __init__(self, modules):
        self.modules = modules
    def __enter__(self):
        preserve_all_weights_block_standard_parameters(self.modules)
    def __exit__(self, type, value, traceback):
        revert_all_weights_block_to_standard_parameters(self.modules)

# note: after calling fwAD_JvP or fwAD_JvP_batch_ensemble, the weights are no
# longer stored as nn.Parameters of the respective modules; to keep the original
# parameters, wrap the code in above context manager
# (``with FwAD_JvP_PreserveAndRevertWeightsToParameters(modules): ...``), or
# manually call ``preserve_all_weights_block_standard_parameters(modules)``
# before and ``revert_all_weights_block_to_standard_parameters(modules)`` after

def fwAD_JvP(x, model, vec, modules, pre_activation=False, saturation_safety=True):

    assert len(vec.shape) == 1
    assert len(x.shape) == 4

    model.eval()

    with torch.no_grad(), fwAD.dual_level():
        set_all_weights_block_tangents(modules, vec)
        out = model(x, saturation_safety=saturation_safety)[1 if pre_activation else 0]
        JvP = fwAD.unpack_dual(out).tangent

    return JvP

# note: after calling fwAD_JvP_batch_ensemble the weights nn.Parameters of the module are
# renamed and a dual tensor attribute is stored instead under the original name;
# to obtain the original parameters, call
# ``preserve_all_weights_block_standard_parameters(modules)`` before and
# ``revert_all_weights_block_to_standard_parameters(modules)`` after
def fwAD_JvP_batch_ensemble(x, model, vec, modules, pre_activation=False, saturation_safety=True):

    assert len(vec.shape) == 2
    assert len(x.shape) in (4, 5)

    if len(x.shape) == 4:
        x = torch.broadcast_to(x, (vec.shape[0],) + x.shape)  # insert instance dim

    model.eval()

    with torch.no_grad(), fwAD.dual_level():
        set_all_weights_block_tangents_batch_ensemble(modules, vec)
        out = model(x, saturation_safety=saturation_safety)[1 if pre_activation else 0]
        JvP = fwAD.unpack_dual(out).tangent

    return JvP

def set_all_weights_block_tangents(modules, tangents):  # TODO set weight tensors from backward model (in order to not copy)?
    n_weights_all = 0
    for layer in modules:
        assert isinstance(layer, Conv2dFwAD)
        n_weights = layer.weight.numel()
        weight = layer.weight
        del layer.weight
        layer.weight = fwAD.make_dual(weight, tangents[n_weights_all:n_weights_all+n_weights].view_as(weight))
        n_weights_all += n_weights

def set_all_weights_block_tangents_batch_ensemble(modules, tangents):
    n_weights_all = 0
    for layer in modules:
        assert isinstance(layer, Conv2dBatchEnsemble)
        n_weights = np.prod(layer.weight.shape[1:])  # dim 0 is instance dim
        weight = layer.weight
        del layer.weight
        layer.weight = fwAD.make_dual(weight, tangents[:, n_weights_all:n_weights_all+n_weights].view_as(weight))
        n_weights_all += n_weights

def preserve_all_weights_block_standard_parameters(modules):
    for layer in modules:
        assert isinstance(layer, (Conv2dFwAD, Conv2dBatchEnsemble))
        assert isinstance(layer.weight, torch.nn.Parameter)
        layer._weight_primal = layer.weight

def revert_all_weights_block_to_standard_parameters(modules):
    for layer in modules:
        assert isinstance(layer, (Conv2dFwAD, Conv2dBatchEnsemble))
        del layer.weight
        layer.weight = layer._weight_primal
        del layer._weight_primal
        # remove and re-add bias parameter to obtain original parameter order
        bias = layer.bias
        del layer.bias
        layer.bias = bias

# jacobian vector product w.r.t. the `weight` parameters of `modules`
def finite_diff_JvP(x, model, vec, modules, eps=None, pre_activation=False, saturation_safety=True):

    assert len(vec.shape) == 1
    model.eval()
    with torch.no_grad():
        map_weights = get_weight_block_vec(modules)

        if eps is None:
            torch_eps = torch.finfo(vec.dtype).eps
            w_map_max = map_weights.abs().max().clamp(min=torch_eps)
            v_max = vec.abs().max().clamp(min=torch_eps)
            eps = np.sqrt(torch_eps) * (1 + w_map_max) / (2 * v_max)

        w_plus = map_weights.clone().detach() + vec * eps
        set_all_weights_block(modules, w_plus)
        f_plus = model(x, saturation_safety=saturation_safety)[1 if pre_activation else 0]

        w_minus = map_weights.clone().detach() - vec * eps
        set_all_weights_block(modules, w_minus)
        f_minus = model(x, saturation_safety=saturation_safety)[1 if pre_activation else 0]

        JvP = (f_plus - f_minus) / (2 * eps)
        set_all_weights_block(modules, map_weights)
        return JvP

def finite_diff_JvP_batch_ensemble(x, model, vec, modules, eps=None, pre_activation=False, saturation_safety=True):

    assert len(vec.shape) == 2
    assert len(x.shape) in (4, 5)

    if len(x.shape) == 4:
        x = torch.broadcast_to(x, (vec.shape[0],) + x.shape)  # insert instance dim

    model.eval()
    with torch.no_grad():
        map_weights = get_weight_block_vec_batch_ensemble(modules)

        if eps is None:
            torch_eps = torch.finfo(vec.dtype).eps
            w_map_max = map_weights.abs().max().clamp(min=torch_eps)
            v_max = vec.abs().max().clamp(min=torch_eps)
            eps = np.sqrt(torch_eps) * (1 + w_map_max) / (2 * v_max)

        w_plus = map_weights.clone().detach() + vec * eps
        set_all_weights_block_batch_ensemble(modules, w_plus)
        f_plus = model(x, saturation_safety=saturation_safety)[1 if pre_activation else 0]

        w_minus = map_weights.clone().detach() - vec * eps
        set_all_weights_block_batch_ensemble(modules, w_minus)
        f_minus = model(x, saturation_safety=saturation_safety)[1 if pre_activation else 0]

        JvP = (f_plus - f_minus) / (2 * eps)
        set_all_weights_block_batch_ensemble(modules, map_weights)
        return JvP

def set_all_weights_block(modules, weights):
    n_weights_all = 0
    for layer in modules:
        assert isinstance(layer, torch.nn.Conv2d)
        n_weights = layer.weight.numel()
        layer.weight.copy_(weights[n_weights_all:n_weights_all+n_weights].view_as(layer.weight))
        n_weights_all += n_weights

def get_weight_block_vec(modules):
    ws = []
    for layer in modules:
        assert isinstance(layer, torch.nn.Conv2d)
        ws.append(layer.weight.flatten())
    return torch.cat(ws)

def set_all_weights_block_batch_ensemble(modules, weights):
    n_weights_all = 0
    for layer in modules:
        assert isinstance(layer, Conv2dBatchEnsemble)
        n_weights = np.prod(layer.weight.shape[1:])  # dim 0 is instance dim
        layer.weight.copy_(weights[:, n_weights_all:n_weights_all+n_weights].view_as(layer.weight))
        n_weights_all += n_weights

def get_weight_block_vec_batch_ensemble(modules):
    ws = []
    for layer in modules:
        assert isinstance(layer, Conv2dBatchEnsemble)
        ws.append(layer.weight.view(layer.num_instances, -1))
    return torch.cat(ws, dim=1)

def get_jac_fwAD(
    input,
    fwAD_model,
    fwAD_modules,
    return_on_cpu=False,
    ):

    jac = []
    ws = get_weight_block_vec(fwAD_modules).clone().detach()
    ws[:] = 0.
    for i in range(ws.shape[0]):
        ws[i] = 1.
        jacs_i = fwAD_JvP(input, fwAD_model, ws, fwAD_modules).flatten()
        jac.append(jacs_i.cpu() if return_on_cpu else jacs_i)
        ws[i] = 0.  # reset
    return torch.stack(jac, dim=1)

def get_jac_fwAD_batch_ensemble(
    input,
    fwAD_be_model,
    fwAD_be_modules,
    return_on_cpu=False,
    ):

    jac = []
    ws = get_weight_block_vec_batch_ensemble(fwAD_be_modules).clone().detach()
    vec_batch_size, param_numel = ws.shape
    ws[:] = 0.
    for i in tqdm(range(0, param_numel, vec_batch_size), desc='get_jac_fwAD_batch_ensemble', miniters=param_numel//vec_batch_size//100):
        # set ws.view(vec_batch_size, -1) to be a subset of rows of torch.eye(param_numel); in last batch, it may contain some additional (zero) rows
        ws.view(vec_batch_size, -1)[:, i:i+vec_batch_size].fill_diagonal_(1.)

        jacs_batch = fwAD_JvP_batch_ensemble(input, fwAD_be_model, ws, fwAD_be_modules).view(vec_batch_size, -1)

        if i+vec_batch_size > param_numel:  # last batch
            jacs_batch = jacs_batch[:param_numel%vec_batch_size]

        jac.append(jacs_batch.T.cpu() if return_on_cpu else jacs_batch.T)

        ws.view(vec_batch_size, -1)[:, i:i+vec_batch_size].fill_diagonal_(0.)  # reset

    return torch.cat(jac, dim=1)
