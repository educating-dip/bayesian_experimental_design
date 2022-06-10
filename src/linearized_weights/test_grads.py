import torch
from dataset.matrix_ray_trafo_utils import MatrixModule
from linearized_weights import log_homoGauss_grad, tv_loss_grad
from deep_image_prior import tv_loss

def test_log_homoGauss_grad():
    im_shape = (128, 128)
    proj_shape = (10, 150)
    x = torch.normal(0.2, 1., (1, 1, *im_shape), requires_grad=True)
    y = torch.normal(0.2, 1., (1, 1, *proj_shape))
    matrix = torch.normal(0.2, 1., (proj_shape[0] * proj_shape[1], im_shape[0] * im_shape[1]))

    trafo_module = MatrixModule(matrix, proj_shape)
    trafo_module_adj = MatrixModule(matrix.T, im_shape)

    loss = torch.nn.functional.mse_loss(trafo_module(x), y)

    x_grad = 2 / y.numel() * log_homoGauss_grad(trafo_module(x), y, trafo_module_adj)

    x_grad_via_mse_loss_autograd = torch.autograd.grad(loss, x)[0]

    print('dist(2 / y.numel() * log_homoGauss_grad, x_grad_via_mse_loss_autograd):', torch.dist(x_grad, x_grad_via_mse_loss_autograd))
    print('mean(abs(x_grad_via_mse_loss_autograd)):', x_grad_via_mse_loss_autograd.abs().mean())
    assert torch.allclose(x_grad, x_grad_via_mse_loss_autograd)

def test_tv_loss_grad():
    im_shape = (128, 128)
    x = torch.normal(0.2, 1., (1, 1, *im_shape), requires_grad=True)

    loss = tv_loss(x)
#     loss = torch.abs(x[..., :, 1:] - x[..., :, :-1]).sum()

    x_grad = tv_loss_grad(x)
        # breakpoint()
#     sign_diff_x = torch.sign(torch.diff(-x[..., :-1, :], n=1, dim=-1))
#     pad_x = torch.zeros((1, 1, x.shape[-2], 1), device = x.device)
#     pad_y = torch.zeros((1, 1, 1, x.shape[-1]), device = x.device)
#     diff_x_pad = torch.cat([pad_x, torch.cat([sign_diff_x, pad_y[:, :, :, :-1]], dim=-2), pad_x], dim=-1)
#     x_grad = torch.diff(diff_x_pad, n=1, dim=-1)

    x_grad_via_tv_loss_autograd = torch.autograd.grad(loss, x)[0]

    print('dist(2 / y.numel() * log_homoGauss_grad, x_grad_via_tv_loss_autograd):', torch.dist(x_grad, x_grad_via_tv_loss_autograd))
    print('mean(abs(x_grad_via_tv_loss_autograd)):', x_grad_via_tv_loss_autograd.abs().mean())
    assert torch.allclose(x_grad, x_grad_via_tv_loss_autograd)


if __name__ == '__main__':
    test_log_homoGauss_grad()
    test_tv_loss_grad()
