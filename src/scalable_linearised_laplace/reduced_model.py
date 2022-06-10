import torch
from copy import deepcopy
from functools import reduce

class Inactive(torch.nn.Module):
    def __init__(self, im_shape):
        super().__init__()
        # output a tensor indicating the image shape (but not the batch or channel dimensions),
        # useful for dynamic shape inference like in UpBlock.forward
        self.register_buffer('dummy_out', torch.empty(0, 0, *im_shape))

    def forward(self, *args, **kwargs):
        return self.dummy_out

class Leaf(torch.nn.Module):
    def __init__(self, out):
        super().__init__()
        self.register_buffer('out', out.clone())

    def forward(self, *args, **kwargs):
        return self.out

def get_reduced_model(model, x_input, replace_inactive, replace_leaf, return_module_mapping=False, share_parameters=True):

    shared_parameters_from_model = model if share_parameters else None

    reduced_model = deepcopy(model)
    if share_parameters:
        for param_name, param in model.named_parameters(recurse=True):
            *module_names, attr_name = param_name.split('.')
            new_parent_module = reduce(getattr, module_names, reduced_model)
            delattr(new_parent_module, attr_name)
            setattr(new_parent_module, attr_name, param)

    ## capture leaf values (we can use reduced_model here because it is just a clone of model at this point)
    module_orig_to_copies_mapping = {old: copy for copy, old in zip(reduced_model.modules(), model.modules())}
    # translate specified inactive and leaf modules to copied modules
    replace_inactive = [module_orig_to_copies_mapping[old] for old in replace_inactive]
    replace_leaf = [module_orig_to_copies_mapping[old] for old in replace_leaf]

    inactive_im_shapes = {}
    leaf_values = {}
    def capture_inactive_im_shape(module, input, output):
        assert module not in inactive_im_shapes  # only support models for which each module is called once during model.forward
        inactive_im_shapes[module] = tuple(output.shape[2:])
    def capture_leaf_value(module, input, output):
        assert module not in leaf_values  # only support models for which each module is called once during model.forward
        leaf_values[module] = output.detach()

    hook_handles = []
    for inactive in replace_inactive:
        h = inactive.register_forward_hook(capture_inactive_im_shape)
        hook_handles.append(h)
    for leaf in replace_leaf:
        h = leaf.register_forward_hook(capture_leaf_value)
        hook_handles.append(h)
    reduced_model.eval()
    reduced_model.forward(x_input)
    for h in hook_handles:
        h.remove()

    ## replace inactive and leaf layers
    if return_module_mapping:

        module_copies_to_orig_mapping = {copy: old for copy, old in zip(reduced_model.modules(), model.modules())}

        replaced_module_mapping = {}
        replace_with_inactive_and_leaf_layers(
            reduced_model, replace_inactive, replace_leaf, inactive_im_shapes, leaf_values,
            out_module_mapping=replaced_module_mapping)

        # translate "copy -> new" to original modules "old -> new"
        replaced_module_mapping = {module_copies_to_orig_mapping[copy]: new for copy, new in replaced_module_mapping.items()}

        module_mapping = module_orig_to_copies_mapping.copy()
        module_mapping.update(replaced_module_mapping)

        return reduced_model, module_mapping

    else:
        replace_with_inactive_and_leaf_layers(
            reduced_model, replace_inactive, replace_leaf, inactive_im_shapes, leaf_values)
        return reduced_model

# inspired by https://discuss.pytorch.org/t/how-to-modify-a-pretrained-model/60509/13
def replace_with_inactive_and_leaf_layers(
        model, replace_inactive, replace_leaf, inactive_im_shapes, leaf_values, out_module_mapping=None):

    for n, module in model.named_children():
        new_module = None
        if module in replace_leaf:
            new_module = Leaf(leaf_values[module])
        elif module in replace_inactive:
            new_module = Inactive(inactive_im_shapes[module])
        if new_module is not None:
            setattr(model, n, new_module)
            if out_module_mapping is not None:
                out_module_mapping[module] = new_module

        # replace potential children recursively
        replace_with_inactive_and_leaf_layers(
                module, replace_inactive, replace_leaf, inactive_im_shapes, leaf_values,
                out_module_mapping=out_module_mapping)
