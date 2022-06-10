# adapted from fannypack 
import torch 
import torch.nn as nn
from typing import List
from opt_einsum import contract

_supported_layers = ['Linear', 'Conv2d']  # Supported layer class types
_hooks_disabled: bool = False           # work-around for https://github.com/pytorch/pytorch/issues/25723
_preserve_graph: bool = False
_grad_list = {}

def get_grad_list():
    return _grad_list

def clear_grads() -> None:
    """Delete external dictionary containing gradients"""
    global _grad_list
    _grad_list = {}

def disable_batch_grad_hooks() -> None:
    """
    Globally disable all hooks installed by this library.
    """
    global _hooks_disabled
    _hooks_disabled = True

def enable_batch_grad_hooks() -> None:
    """the opposite of disable_hooks()"""
    global _hooks_disabled
    _hooks_disabled = False

def disable_preserve_graph() -> None:
    """
    Globally disable all hooks installed by this library.
    """
    global _preserve_graph
    _preserve_graph = False

def enable_preserve_graph() -> None:
    """the opposite of disable_hooks()"""
    global _preserve_graph
    _preserve_graph = True

def remove_batch_grad_hooks(model: nn.Module) -> None:
    """
    Remove hooks added by add_hooks(model)
    """
    assert not isinstance(model, nn.DataParallel)
    if 'batch_grad_hooks' not in model.__dict__.keys():
        print("Warning, asked to remove hooks, but no hooks found")
    else:
        for handle in model.__dict__['batch_grad_hooks']:
            handle.remove()
        del model.__dict__['batch_grad_hooks']
        clear_grads()

def is_supported(layer: nn.Module) -> bool:
    """Check if this layer is supported"""
    return _layer_type(layer) in _supported_layers

def _layer_type(layer: nn.Module) -> str:
    return layer.__class__.__name__

def _capture_activations(layer: nn.Module, input: List[torch.Tensor], output: torch.Tensor):
    """Save activations into layer.activations in forward pass"""

    if _hooks_disabled:
        return
    assert _layer_type(layer) in _supported_layers, "Hook installed on unsupported layer, this shouldn't happen"
    
    if _preserve_graph:
        setattr(layer, "activations", input[0])
    else:
        setattr(layer, "activations", input[0].detach())

def clear_layer_activations(layer: nn.Module) -> None:
    """Delete layer.activations in every layer."""
    if hasattr(layer, 'activations'):
        del layer.activations

def add_batch_grad_hooks(model: nn.Module, modules: list, include_grad_bias: bool = False, loss_type: str = 'sum') -> None:
    """
    Adds hooks to model to save activations and backprop values.
    The hooks will
    1. save activations into param.activations during forward pass
    Call "remove_batch_grad_hooks(model)" to disable this.
    Args:
        model:
    """

    global _hooks_disabled
    _hooks_disabled = False
    global inc_grad_bias
    inc_grad_bias = include_grad_bias

    assert not isinstance(model, nn.DataParallel)

    if 'batch_grad_hooks' in model.__dict__.keys():
        print('Warning: hooks already installed, deleting them')
        remove_batch_grad_hooks(model)

    clear_grads()
    handles = []
    for layer in modules:
        if _layer_type(layer) in _supported_layers:
            handles.append(layer.register_forward_hook(_capture_activations))
            handles.append(layer.register_backward_hook(generate_batch_grad_hook(loss_type)))

    model.__dict__.setdefault('batch_grad_hooks', []).extend(handles)

def generate_batch_grad_hook(loss_type: str = 'sum'):
    assert loss_type in ('sum', 'mean')
    if loss_type == 'mean':
        raise RuntimeError('MultiGPU (external dict saving) is not currently supported with mean loss reduction. \
            This is due to the batch size being ambiguous to a single GPU.')
        
    def _compute_batch_grad(layer: nn.Module, _input, output):
        """Append backprop to layer.backprops_list in backward pass."""

        if _hooks_disabled:
            return
        
        layer_type = _layer_type(layer)
        if layer_type not in _supported_layers:
            return  
        assert hasattr(layer, 'activations'), "No activations detected, run forward after add_hooks(model)"
                
        acts = layer.activations
        n = acts.shape[0]
        
        if _preserve_graph:
            grads = output[0]
        else:
            grads = output[0].detach()

        if loss_type == 'mean':
            grads = grads * n
        else:
            pass
        
        if layer_type == 'Linear':
            
            _grad_list[layer.weight] = linear_batchgrad(acts, grads)
            if inc_grad_bias: 
                _grad_list[layer.bias] = linear_batchgrad_bias(grads)
                        
        elif layer_type == 'Conv2d':
            acts = torch.nn.functional.unfold(acts, layer.kernel_size, padding=layer.padding, stride=layer.stride)
            grads = grads.reshape(n, -1, acts.shape[-1])
            grad1 = conv2d_batchgrad(acts, grads)
            shape = [n] + list(layer.weight.shape)
            
            _grad_list[layer.weight] = grad1.reshape(shape)
            if inc_grad_bias: 
                _grad_list[layer.bias] = conv2d_batchgrad_bias(acts, grads)
                
        clear_layer_activations(layer)
        
    return _compute_batch_grad

def linear_batchgrad(acts, grads):
    return contract('ni,nj->nij', grads, acts)

def linear_batchgrad_bias(grads):
    return grads

def conv2d_batchgrad(acts, grads):
    return contract('ijk,ilk->ijl', grads, acts)

def conv2d_batchgrad_bias(acts, grads):
    return torch.sum(grads, dim=2)

def aggregate_flatten_weight_batch_grad(batch_size, modules, store_device):
    # adapted version from linearized_laplace/jac.py/agregate_flatten_weight_grad
    
    grad_list = get_grad_list()
    aggregate_grads = []
    for module in modules:
        aggregate_grads.append(grad_list[module.weight].reshape(batch_size, -1).to(store_device)) # hard-coded disregarding biases 
    return torch.cat(aggregate_grads, dim=1)
