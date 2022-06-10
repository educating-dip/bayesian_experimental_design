import itertools
import torch
import torch.autograd.forward_ad as fwAD

# assumption: F.interpolate is linear w.r.t. the input
class InterpolateFwADFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None):
        kwargs = {'size': size, 'scale_factor': scale_factor, 'mode': mode, 'align_corners': align_corners, 'recompute_scale_factor': recompute_scale_factor}
        result = torch.nn.functional.interpolate(input, **kwargs)
        ctx.kwargs = kwargs
        return result

    @staticmethod
    def jvp(ctx, g_input, size=None, scale_factor=None, mode=None, align_corners=None, recompute_scale_factor=None):
        g_out = torch.nn.functional.interpolate(g_input, **ctx.kwargs)
        del ctx.kwargs
        return g_out

interpolate_fwAD = InterpolateFwADFunction.apply

class UpsampleFwAD(torch.nn.Upsample):
    def forward(self, input):
        return interpolate_fwAD(input, self.size, self.scale_factor, self.mode, self.align_corners)


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
    for size, scale_factor, mode, align_corners, recompute_scale_factor in itertools.chain(
            itertools.product([None], [2, (2.3, 5.5)], ['bilinear', 'bicubic'], [False, True], [False, True]),
            itertools.product([None], [2, (2.3, 5.5)], ['nearest', 'area'], [None], [False, True])):
        f = lambda input: interpolate_fwAD(input, size, scale_factor, mode, align_corners, recompute_scale_factor)
        # f = lambda input: torch.nn.functional.interpolate(input, size, scale_factor, mode, align_corners, recompute_scale_factor)
        primal_input = torch.randn(1, 7, 10, 10, dtype=torch.double, requires_grad=True)
        tangent_input = torch.randn(1, 7, 10, 10)
        print(size, scale_factor, mode, align_corners, recompute_scale_factor)
        torch.autograd.gradcheck(f, (primal_input,), check_forward_ad=True,
                                 check_backward_ad=False, check_undefined_grad=False,
                                 check_batched_grad=False)
