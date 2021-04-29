import torch
from cmws.examples.timeseries.util import Kernel


def test_kernel_shape():
    device = torch.device("cpu")
    batch_size, num_timesteps_1, num_timesteps_2 = 2, 3, 4
    x_1 = torch.linspace(0, 1, steps=num_timesteps_1, device=device)[None, :, None].expand(
        batch_size, num_timesteps_1, 1
    )
    x_2 = torch.linspace(0, 1, steps=num_timesteps_2, device=device)[None, None, :].expand(
        batch_size, 1, num_timesteps_2
    )

    kernel = Kernel("W+R*E", torch.randn(3, 5, device=device))
    cov = kernel(x_1, x_2)
    assert list(cov.shape) == [batch_size, num_timesteps_1, num_timesteps_2]
