from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F

# replacing torch.nn.Conv2d
class Conv2dBatchEnsemble(nn.Module):
    def __init__(self, module, num_instances, conv2d_fun=None):
        super().__init__()
        assert isinstance(module, nn.Conv2d)
        self.num_instances = num_instances
        self.conv2d_fun = F.conv2d if conv2d_fun is None else conv2d_fun

        self.in_channels = module.in_channels
        self.out_channels = module.out_channels
        self.kernel_size = module.kernel_size
        self.stride = module.stride
        self.padding = module.padding
        self.dilation = module.dilation
        self.output_padding = module.output_padding
        assert module.groups == 1,  'not supported'  # groups are used for num_instances, implementing groups > 1 probably would not be too hard, but not needed for now
        self.padding_mode = module.padding_mode
        self._reversed_padding_repeated_twice = module._reversed_padding_repeated_twice

        self.weight = nn.Parameter(torch.stack([module.weight] * self.num_instances))
        if module.bias is not None:
            self.bias = nn.Parameter(torch.stack([module.bias] * self.num_instances))
        else:
            self.register_parameter('bias', None)

    def _conv_forward(self, input, weight, bias):

        # grouped conv adapted from https://discuss.pytorch.org/t/how-to-apply-different-kernels-to-each-example-in-a-batch-when-using-convolution/84848/4

        # original input.shape: (self.num_instances, N, C_in, ...)
        _, batch_size, in_channels, *in_spatial_dims = input.shape
        input = input.transpose(0, 1)  # (N, self.num_instances, C_in, ...)
        input = input.view(batch_size, self.num_instances * in_channels, *in_spatial_dims)

        # original weight.shape: (self.num_instances, C_out, ...)
        out_channels = weight.shape[1]
        weight = weight.view(-1, *weight.shape[2:])  # (self.num_instances * C_out, ...)
        # original bias.shape: (self.num_instances, C_out)
        bias = bias.view(-1)  # (self.num_instances * C_out)

        if self.padding_mode != 'zeros':
            out = self.conv2d_fun(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                    weight, bias, self.stride,
                    (0, 0), self.dilation, self.num_instances)
        out = self.conv2d_fun(input, weight, bias, self.stride,
                self.padding, self.dilation, self.num_instances)

        out = out.view(batch_size, self.num_instances, out_channels, *out.shape[2:])
        out = out.transpose(0, 1)  # (self.num_instances, N, C_out, ...)

        return out

    def forward(self, input):
        return self._conv_forward(input, self.weight, self.bias)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        conv2d_s = s.format(**self.__dict__)
        s = 'Conv2d({}), num_instances={}'.format(conv2d_s, self.num_instances)
        if self.conv2d_fun is not F.conv2d:
            s += ', conv2d_fun={}'.format(self.conv2d_fun)
        return s

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

# replacing any module which already supports a batch dimension and does not
# need to differentiate between instances, e.g. a parameter-free module or
# a module for which the same parameters should be used for all instances
# (this module just temporarily collapses batch and instance dim)
class SharedBatchEnsemble(nn.Module):
    def __init__(self, module, num_instances):
        super().__init__()
        self.module = deepcopy(module)
        self.num_instances = num_instances

    def forward(self, input):
        # combine num_instances and batch dim
        input = input.view(-1, *input.shape[2:])
        out = self.module(input)
        out = out.view(self.num_instances, -1, *out.shape[1:])
        return out

    def extra_repr(self):
        return 'num_instances={}'.format(self.num_instances)

class FallbackBatchEnsemble(nn.Module):
    def __init__(self, module, num_instances):
        super().__init__()
        self.ensemble_modules = nn.ModuleList(
                [deepcopy(module) for _ in range(num_instances)])

    def forward(self, x):
        out = torch.stack([m(x_i) for m, x_i in zip(self.ensemble_modules, x)])
        return out

BATCH_ENSEMBLE_REPLACE = (
    (nn.Conv2d, Conv2dBatchEnsemble),
)

BATCH_ENSEMBLE_KEEP = (
    nn.ModuleList,
    nn.Sequential,
    nn.LeakyReLU,
)

# model is considered a container that can handle multi-instance input and
# output, only its submodules are replaced
def get_batch_ensemble_model(model, num_instances, return_module_mapping=False, **kwargs):
    be_model = deepcopy(model)

    if return_module_mapping:

        module_orig_to_copies_mapping = {old: copy for copy, old in zip(be_model.modules(), model.modules())}
        module_copies_to_orig_mapping = {copy: old for copy, old in zip(be_model.modules(), model.modules())}

        replaced_module_mapping = {}
        replace_with_batch_ensemble_layers(be_model, num_instances, out_module_mapping=replaced_module_mapping, **kwargs)

        # translate module_copies_mapping ("copy -> new") to original modules "old -> new"
        replaced_module_mapping = {module_copies_to_orig_mapping[copy]: new for copy, new in replaced_module_mapping.items()}

        module_mapping = module_orig_to_copies_mapping.copy()
        module_mapping.update(replaced_module_mapping)

        return be_model, module_mapping

    else:
        replace_with_batch_ensemble_layers(be_model, num_instances, **kwargs)
        return be_model

# inspired by https://discuss.pytorch.org/t/how-to-modify-a-pretrained-model/60509/13
def replace_with_batch_ensemble_layers(
        model, num_instances,
        add_replace=(), add_keep=(), remove_replace=(), remove_keep=(),
        out_module_mapping=None):

    replace = tuple(t for t in BATCH_ENSEMBLE_REPLACE + tuple(add_replace) if t not in remove_replace)
    keep = tuple(t for t in BATCH_ENSEMBLE_KEEP + tuple(add_keep) if t not in remove_keep)

    for n, module in model.named_children():
        if not isinstance(module, keep):
            new_constructor = None
            for old, new in replace:
                if isinstance(module, old):
                    new_constructor = new
            if new_constructor is None:
                new_constructor = FallbackBatchEnsemble
            new_module = new_constructor(module, num_instances)
            setattr(model, n, new_module)
            if out_module_mapping is not None:
                out_module_mapping[module] = new_module

        # replace potential children recursively (but not inside FallbackBatchEnsemble)
        if not isinstance(getattr(model, n), FallbackBatchEnsemble):
            replace_with_batch_ensemble_layers(
                    module, num_instances,
                    add_replace=add_replace, add_keep=add_keep, remove_replace=remove_replace, remove_keep=remove_keep,
                    out_module_mapping=out_module_mapping)

if __name__ == '__main__':

    def test_fallback():
        torch.manual_seed(0)

        NUM_INSTANCES = 4

        model = nn.Conv2d(1, 1, 3)
        be_model = FallbackBatchEnsemble(model, NUM_INSTANCES)

        print(be_model)

        test_samples = torch.rand(NUM_INSTANCES, 1, 1, 6, 6)


        out = [model(test_sample) for test_sample in test_samples]
        out_be = be_model(test_samples)

        print('output differences')
        for out_i, out_be_i in zip(out, out_be):
            print(torch.max(torch.abs(out_be_i - out_i)))


        def _modify_layer(layer, seed):
            torch.manual_seed(seed)
            layer.weight.data += torch.rand_like(layer.weight.data)

        out_mod = []
        for i, test_sample in enumerate(test_samples):
            model_copy = deepcopy(model)
            _modify_layer(model_copy, seed=i)
            out_mod.append(model_copy(test_sample))
        be_model_copy = deepcopy(be_model)
        for i, test_sample in enumerate(test_samples):
            _modify_layer(be_model_copy.ensemble_modules[i], seed=i)
        out_be_mod = be_model_copy(test_samples)

        print('output differences after modifications')
        for out_mod_i, out_be_mod_i in zip(out_mod, out_be_mod):
            print(torch.max(torch.abs(out_be_mod_i - out_mod_i)))

    test_fallback()

    def test():
        torch.manual_seed(0)

        NUM_INSTANCES = 4

        model = nn.Sequential(nn.Conv2d(1, 1, 3))  # wrap in a sequential since
                                                # get_batch_ensemble_model
                                                # expects a container
        # compare with fallback batch ensemble (which is assumed to work correctly,
        # tested above separately for a simple case)
        fallback_be_model = FallbackBatchEnsemble(model, NUM_INSTANCES)
        be_model = get_batch_ensemble_model(model, NUM_INSTANCES)

        print(be_model)

        test_samples = torch.rand(NUM_INSTANCES, 1, 1, 6, 6)

        out = [model(test_sample) for test_sample in test_samples]
        out_be = be_model(test_samples)

        print('output differences')
        for out_i, out_be_i in zip(out, out_be):
            print(torch.max(torch.abs(out_be_i - out_i)))


        fallback_be_model_copy = deepcopy(fallback_be_model)
        be_model_copy = deepcopy(be_model)
        for i, test_sample in enumerate(test_samples):
            mod_value = torch.rand_like(model[0].weight.data)
            fallback_be_model_copy.ensemble_modules[i][0].weight.data += mod_value
            be_model_copy[0].weight.data[i] += mod_value
        out_fallback_be_mod = fallback_be_model_copy(test_samples)
        out_be_mod = be_model_copy(test_samples)

        print('output differences after modifications')
        for out_fallback_be_mod_i, out_be_mod_i in zip(out_fallback_be_mod, out_be_mod):
            print(torch.max(torch.abs(out_be_mod_i - out_fallback_be_mod_i)))

    test()
