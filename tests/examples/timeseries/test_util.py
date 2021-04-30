import torch
import cmws.examples.timeseries.util as util
import cmws


def test_kernel_shape():
    device = cmws.util.get_device()
    batch_size, num_timesteps_1, num_timesteps_2 = 2, 3, 4
    x_1 = torch.linspace(0, 1, steps=num_timesteps_1, device=device)[None, :, None].expand(
        batch_size, num_timesteps_1, 1
    )
    x_2 = torch.linspace(0, 1, steps=num_timesteps_2, device=device)[None, None, :].expand(
        batch_size, 1, num_timesteps_2
    )
    expression = "W+R*E"

    kernel = util.Kernel(expression, torch.randn(3, 5, device=device))
    cov = kernel(x_1, x_2)
    assert list(cov.shape) == [batch_size, num_timesteps_1, num_timesteps_2]


def test_conversions():
    device = cmws.util.get_device()
    expression = "W+R*E"
    assert expression == util.get_expression(util.get_numeric(expression, device=device))


def test_count_base_kernels():
    expression = "W+R*E"
    assert util.count_base_kernels(expression) == 3
