import cmws.examples.stacking.models.stacking
import torch
from cmws.losses import get_log_marginal_joint


def test_get_log_marginal_joint_dims():
    num_channels, im_size = 3, 32
    shape = [2, 3]
    discrete_shape = [7, 8]
    num_particles = 12

    generative_model = cmws.examples.stacking.models.stacking.GenerativeModel(im_size=im_size)
    guide = cmws.examples.stacking.models.stacking.Guide(im_size=im_size)

    obs = torch.rand(*[*shape, num_channels, im_size, im_size])
    num_blocks = torch.randint(2, size=discrete_shape + shape) + 1
    stacking_program = torch.randint(2, size=discrete_shape + shape + [guide.max_num_blocks])
    discrete_latent = (num_blocks, stacking_program)

    log_marginal_joint = get_log_marginal_joint(
        generative_model, guide, discrete_latent, obs, num_particles
    )

    assert list(log_marginal_joint.shape) == discrete_shape + shape
