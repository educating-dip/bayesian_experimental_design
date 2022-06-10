from scalable_linearised_laplace.batch_ensemble import get_batch_ensemble_model, SharedBatchEnsemble
from deep_image_prior.network.unet import InBlock, DownBlock, UpBlock, OutBlock, Concat
from copy import deepcopy
from torch import nn

class ConcatBatchEnsemble(nn.Module):
    def __init__(self, module, num_instances):
        super().__init__()
        assert isinstance(module, Concat)
        self.concat = deepcopy(module)
        self.num_instances = num_instances

    def forward(self, *inputs):
        # combine num_instances and batch dim
        inputs = [input.view(-1, *input.shape[2:]) for input in inputs]
        out = self.concat(*inputs)
        out = out.view(self.num_instances, -1, *out.shape[1:])
        return out

    def extra_repr(self):
        return 'num_instances={}'.format(self.num_instances)

def get_unet_batch_ensemble(model, num_instances, return_module_mapping=False):
    unet_batch_ensemble_model, *module_map = get_batch_ensemble_model(
            model, num_instances,
            add_keep=(InBlock, DownBlock, UpBlock, OutBlock,),
            add_replace=(
                    (Concat, ConcatBatchEnsemble),
                    (nn.Upsample, SharedBatchEnsemble),
                    (nn.GroupNorm, SharedBatchEnsemble),
                ),
            return_module_mapping=return_module_mapping)
    return (unet_batch_ensemble_model, module_map[0]) if return_module_mapping else unet_batch_ensemble_model
