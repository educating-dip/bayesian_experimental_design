import itertools
import torch
import torch.autograd.forward_ad as fwAD

# class GroupNormFwADFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, num_groups, weight=None, bias=None, eps=1e-5):
#         kwargs = {'weight': weight, 'bias': bias, 'eps': eps}
#         result = torch.nn.functional.group_norm(input, num_groups, **kwargs)
#         ctx.num_groups = num_groups
#         ctx.kwargs = kwargs
#         return result

#     @staticmethod
#     def jvp(ctx, g_input, num_groups, weight=None, bias=None, eps=1e-5):
#         g_out =  # TODO
#         del ctx.num_groups
#         del ctx.kwargs
#         return g_out

# TODO use torch.nn.functional.group_norm for forward pass

# group_norm_fwAD = GroupNormFwADFunction.apply
def group_norm_fwAD(input, num_groups, weight=None, bias=None, eps=1e-5):
    assert input.shape[1] % num_groups == 0
    channels_per_group = input.shape[1] // num_groups
    normed_groups = []
    for i in range(0, input.shape[1], channels_per_group):
        x = input[:, i:i+channels_per_group]
        x = (x - torch.mean(x)) / torch.sqrt(torch.var(x, unbiased=False) + eps)
        normed_groups.append(x)
    out = torch.cat(normed_groups, dim=1)
    if weight is not None:
        out = out * weight[:, None, None]
    if bias is not None:
        out = out + bias[:, None, None]
    return out

class GroupNormFwAD(torch.nn.GroupNorm):
    def forward(self, input):
        return group_norm_fwAD(
            input, self.num_groups, self.weight, self.bias, self.eps)


if __name__ == '__main__':
    # primal_input = torch.randn(1, 7, 10, 10, dtype=torch.double, requires_grad=True)
    # tangent_input = torch.randn(1, 7, 10, 10)
    # with fwAD.dual_level():
    #     dual_input = fwAD.make_dual(primal_input, tangent_input)
    #     dual_output = interpolate_fwAD(dual_input, None, 2)
    #     # jvp = fwAD.unpack_dual(dual_output).tangent

    # It is important to use ``autograd.gradcheck`` to verify that your
    # custom autograd Function computes the gradients correctly. By default,
    # gradcheck only checks the backward-mode (reverse-mode) AD gradients. Specify
    # ``check_forward_ad=True`` to also check forward grads. If you did not
    # implement the backward formula for your function, you can also tell gradcheck
    # to skip the tests that require backward-mode AD by specifying
    # ``check_backward_ad=False``, ``check_undefined_grad=False``, and
    # ``check_batched_grad=False``.
    for num_groups, weight, bias, eps in itertools.product(
            [1, 2, 3, 4], [None, torch.randn(12, dtype=torch.double)], [None, torch.randn(12, dtype=torch.double)], [1e-5, 1e-2]):
        f = lambda input: group_norm_fwAD(input, num_groups, weight, bias, eps)
        primal_input = torch.randn(1, 12, 10, 10, dtype=torch.double, requires_grad=True)
        tangent_input = torch.randn(1, 12, 10, 10)
        print(num_groups, weight, bias, eps)
        x = f(primal_input)
        f_reference = lambda input: torch.nn.functional.group_norm(input, num_groups, weight, bias, eps)
        x_reference = f_reference(primal_input)
        print(torch.max(torch.abs(x - x_reference)))
        torch.autograd.gradcheck(f, (primal_input,), check_forward_ad=True,
                                 check_backward_ad=False, check_undefined_grad=False,
                                 check_batched_grad=False)
