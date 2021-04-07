import torch
from cmws.examples.stacking.models.stacking import Guide


def test_guide_sample_discrete_dims():
    num_channels, im_size = 3, 32
    shape = [2, 3]
    sample_shape = [4, 5, 6]

    guide = Guide(im_size=im_size)
    obs = torch.rand(*[*shape, num_channels, im_size, im_size])

    num_blocks, stacking_program = guide.sample_discrete(obs, sample_shape)

    assert list(num_blocks.shape) == sample_shape + shape
    assert list(stacking_program.shape) == sample_shape + shape + [guide.max_num_blocks]


def test_guide_sample_continuous_dims():
    num_channels, im_size = 3, 32
    shape = [2, 3]
    sample_shape = [4, 5, 6]
    discrete_shape = [7, 8]

    guide = Guide(im_size=im_size)
    obs = torch.rand(*[*shape, num_channels, im_size, im_size])
    num_blocks = torch.randint(2, size=discrete_shape + shape) + 1
    stacking_program = torch.randint(2, size=discrete_shape + shape + [guide.max_num_blocks])
    discrete_latent = (num_blocks, stacking_program)

    raw_locations = guide.sample_continuous(obs, discrete_latent, sample_shape)

    assert list(raw_locations.shape) == sample_shape + discrete_shape + shape + [
        guide.max_num_blocks
    ]


def test_guide_log_prob_discrete_dims():
    num_channels, im_size = 3, 32
    shape = [2, 3]
    discrete_shape = [4, 5]

    guide = Guide(im_size=im_size)
    obs = torch.rand(*[*shape, num_channels, im_size, im_size])
    num_blocks = torch.randint(2, size=discrete_shape + shape) + 1
    stacking_program = torch.randint(2, size=discrete_shape + shape + [guide.max_num_blocks])
    discrete_latent = (num_blocks, stacking_program)

    log_prob = guide.log_prob_discrete(obs, discrete_latent)

    assert list(log_prob.shape) == discrete_shape + shape


def test_guide_log_prob_continuous_dims():
    num_channels, im_size = 3, 32
    shape = [2, 3]
    discrete_shape = [7, 8]
    continuous_shape = [9, 10, 11]

    guide = Guide(im_size=im_size)
    obs = torch.rand(*[*shape, num_channels, im_size, im_size])
    num_blocks = torch.randint(2, size=discrete_shape + shape) + 1
    stacking_program = torch.randint(2, size=discrete_shape + shape + [guide.max_num_blocks])
    discrete_latent = (num_blocks, stacking_program)
    continuous_latent = torch.randn(
        *[*continuous_shape, *discrete_shape, *shape, guide.max_num_blocks]
    )

    log_prob = guide.log_prob_continuous(obs, discrete_latent, continuous_latent)

    assert list(log_prob.shape) == continuous_shape + discrete_shape + shape
