import itertools
import torch
import torch.autograd.forward_ad as fwAD
from torch.nn.grad import conv2d_input, conv2d_weight

class Conv2dFwADFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride=1, padding=0, dilation=1, groups=1):
        kwargs = {'stride': stride, 'padding': padding, 'dilation': dilation, 'groups': groups}
        result = torch.nn.functional.conv2d(input, weight, bias, **kwargs)
        ctx.input = input
        ctx.weight = weight
        # ctx.result = result
        ctx.kwargs = kwargs
        return result

    @staticmethod
    def jvp(ctx, g_input, g_weight, g_bias, stride=None, padding=None, dilation=None, groups=None):
        g_out_input = torch.nn.functional.conv2d(g_input.float(), ctx.weight.detach().float(), **ctx.kwargs)
        g_out_weight = torch.nn.functional.conv2d(ctx.input.detach().float(), g_weight.float(), **ctx.kwargs)
        del ctx.input
        del ctx.weight
        del ctx.kwargs
        g_out_bias = g_bias
        g_out = g_out_input + g_out_weight + g_out_bias[None, :, None, None]
        return g_out

    # # caution: does not work for groups > 1 (see https://github.com/pytorch/pytorch/issues/51430)
    # @staticmethod
    # def backward(ctx, grad_output):
    #     # g_in_input, g_in_weight = torch.autograd.grad(ctx.result, (ctx.input, ctx.weight), grad_output)
    #     # breakpoint()
    #     g_in_input = conv2d_input(ctx.input.size(), ctx.weight, grad_output, **ctx.kwargs)
    #     g_in_weight = conv2d_weight(ctx.input, ctx.weight.size(), grad_output, **ctx.kwargs)
    #     return (g_in_input, g_in_weight, None) + (None,) * len(ctx.kwargs)


conv2d_fwAD = Conv2dFwADFunction.apply

class Conv2dFwAD(torch.nn.Conv2d):
    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != 'zeros':
            return conv2d_fwAD(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return conv2d_fwAD(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)


if __name__ == '__main__':
    # primal_input = torch.randn(1, 7 * 2, 10, 10, dtype=torch.double, requires_grad=True)
    # tangent_input = torch.randn(1, 7 * 2, 10, 10)
    # primal_weight = torch.randn(8, 7, 3, 3, dtype=torch.double, requires_grad=True)
    # tangent_weight = torch.randn(8, 7, 3, 3)
    # primal_bias = torch.randn(8, dtype=torch.double, requires_grad=True)
    # tangent_bias = torch.randn(8)
    # bias_no_grad = torch.randn(8, dtype=torch.double)  # also test with non-dual bias
    # with fwAD.dual_level():
    #     dual_input = fwAD.make_dual(primal_input, tangent_input)
    #     dual_weight = fwAD.make_dual(primal_weight, tangent_weight)
    #     dual_bias = fwAD.make_dual(primal_bias, tangent_bias)
    #     # dual_output = conv2d_fwAD(dual_input, dual_weight, dual_bias)
    #     dual_output_bias_no_grad = conv2d_fwAD(dual_input, dual_weight, bias_no_grad, 1, 0, 1, 2)
    #     # jvp = fwAD.unpack_dual(dual_output).tangent

    # It is important to use ``autograd.gradcheck`` to verify that your
    # custom autograd Function computes the gradients correctly. By default,
    # gradcheck only checks the backward-mode (reverse-mode) AD gradients. Specify
    # ``check_forward_ad=True`` to also check forward grads. If you did not
    # implement the backward formula for your function, you can also tell gradcheck
    # to skip the tests that require backward-mode AD by specifying
    # ``check_backward_ad=False``, ``check_undefined_grad=False``, and
    # ``check_batched_grad=False``.
    for stride, padding, dilation, groups in itertools.product([1, 2], [0, 1], [1, 2], [1, 2]):
        f = lambda input, weight, bias: conv2d_fwAD(input, weight, bias, stride, padding, dilation, groups)
        # f = lambda input, weight, bias: torch.nn.functional.conv2d(input, weight, bias, stride, padding, dilation, groups)
        primal_input = torch.randn(1, 7 * groups, 10, 10, dtype=torch.double, requires_grad=True)
        tangent_input = torch.randn(1, 7 * groups, 10, 10)
        primal_weight = torch.randn(8, 7, 3, 3, dtype=torch.double, requires_grad=True)
        tangent_weight = torch.randn(8, 7, 3, 3)
        primal_bias = torch.randn(8, dtype=torch.double, requires_grad=True)
        tangent_bias = torch.randn(8)
        bias_no_grad = torch.randn(8, dtype=torch.double)  # also test with non-dual bias
        print(stride, padding, dilation, groups)
        torch.autograd.gradcheck(f, (primal_input, primal_weight, primal_bias), check_forward_ad=True,
                                 check_backward_ad=False, check_undefined_grad=False,
                                 check_batched_grad=False)
        torch.autograd.gradcheck(f, (primal_input, primal_weight, bias_no_grad), check_forward_ad=True,
                                check_backward_ad=False, check_undefined_grad=False,
                                check_batched_grad=False)
