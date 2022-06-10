import torch
from deep_image_prior.network.unet import UNet, InBlock, DownBlock

def _keep_num_blocks_suffices(model, keep_num_blocks, keep_modules):
    assert isinstance(model, UNet)

    blocks = [model.inc, *model.down, *model.up, model.outc]
    total_num_blocks = len(blocks)  # 2 + 2 * (model.scales - 1)
    suffices = all(
            any(m in b.modules() for b in blocks[total_num_blocks-keep_num_blocks:])
            for m in keep_modules)
    return suffices

def get_inactive_and_leaf_modules_unet(model, keep_modules=None, keep_num_blocks=None):
    assert isinstance(model, UNet)

    blocks = [model.inc, *model.down, *model.up, model.outc]
    total_num_blocks = len(blocks)  # 2 + 2 * (model.scales - 1)
    if keep_num_blocks is None:
        assert keep_modules is not None
        keep_num_blocks = next(
                n for n in range(total_num_blocks+1)
                if _keep_num_blocks_suffices(model, n, keep_modules))
    assert keep_num_blocks in range(total_num_blocks+1)

    # blocks to replace
    inactive_or_leaf_modules = blocks[:total_num_blocks-keep_num_blocks]

    # determine leaf blocks (for which the output value is needed)
    leaf_modules = []
    if keep_num_blocks < total_num_blocks:
        # output of the block preceeding the first one being kept is needed
        leaf_modules.append(blocks[-1-keep_num_blocks])

    if model.scales > 1 and keep_num_blocks >= 2:
        # at least one up block is kept
        reverse_ups, skip_sources = model.up[::-1], [model.inc, *model.down[:-1]]
        for up, skip_src in zip(reverse_ups[:keep_num_blocks-1], skip_sources[:keep_num_blocks-1]):
            # up is kept, so output of the source block before the skip connection is needed
            if (skip_src in inactive_or_leaf_modules and  # source block is being replaced
                    up.skip and  # skip connection is active
                    skip_src not in leaf_modules  # not already marked as a leaf
                    ):
                leaf_modules.append(skip_src)

    # remaining blocks are inactive
    inactive_modules = [b for b in inactive_or_leaf_modules if b not in leaf_modules]

    return inactive_modules, leaf_modules
