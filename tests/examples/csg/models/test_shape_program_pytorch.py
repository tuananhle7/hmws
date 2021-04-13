import torch
from cmws.examples.csg.models.shape_program_pytorch import GenerativeModel, Guide


def test_generative_model_log_prob_dims():
    im_size = 64
    shape = [2, 3]
    sample_shape = [4, 5]

    generative_model = GenerativeModel(im_size=im_size)
    obs = torch.rand(*[*shape, im_size, im_size])
    program_id = torch.randint(3, size=sample_shape + shape)
    shape_ids = torch.randint(2, size=sample_shape + shape + [generative_model.max_num_shapes])
    raw_positions = torch.randn(*[*sample_shape, *shape, generative_model.max_num_shapes, 2])

    log_prob = generative_model.log_prob((program_id, shape_ids, raw_positions), obs)

    assert list(log_prob.shape) == sample_shape + shape


def test_generative_model_sample_dims():
    im_size = 64
    sample_shape = [4, 5]

    generative_model = GenerativeModel(im_size=im_size)
    (program_id, shape_ids, raw_positions), obs = generative_model.sample(sample_shape)

    assert list(program_id.shape) == sample_shape
    assert list(shape_ids.shape) == sample_shape + [generative_model.max_num_shapes]
    assert list(raw_positions.shape) == sample_shape + [generative_model.max_num_shapes, 2]
    assert list(obs.shape) == sample_shape + [im_size, im_size]


def test_generative_model_log_prob_discrete_continuous_dims():
    im_size = 64
    shape = [2]
    discrete_shape = [3, 4]
    continuous_shape = [5, 6]

    generative_model = GenerativeModel(im_size=im_size)
    obs = torch.rand(*[*shape, im_size, im_size])
    program_id = torch.randint(3, size=discrete_shape + shape)
    shape_ids = torch.randint(2, size=discrete_shape + shape + [generative_model.max_num_shapes])
    raw_positions = torch.randn(
        *[*continuous_shape, *discrete_shape, *shape, generative_model.max_num_shapes, 2]
    )

    log_prob = generative_model.log_prob_discrete_continuous(
        (program_id, shape_ids), raw_positions, obs
    )

    assert list(log_prob.shape) == continuous_shape + discrete_shape + shape


def test_guide_sample_dims():
    im_size = 64
    shape = [2, 3]
    sample_shape = [4, 5]

    guide = Guide(im_size=im_size)
    obs = torch.rand(*[*shape, im_size, im_size])

    program_id, shape_ids, raw_positions = guide.sample(obs, sample_shape)

    assert list(program_id.shape) == sample_shape + shape
    assert list(shape_ids.shape) == sample_shape + shape + [guide.max_num_shapes]
    assert list(raw_positions.shape) == sample_shape + shape + [guide.max_num_shapes, 2]


def test_guide_log_prob_dims():
    im_size = 64
    shape = [2, 3]
    sample_shape = [4, 5]

    guide = Guide(im_size=im_size)
    obs = torch.rand(*[*shape, im_size, im_size])
    program_id = torch.randint(3, size=sample_shape + shape)
    shape_ids = torch.randint(2, size=sample_shape + shape + [guide.max_num_shapes])
    raw_positions = torch.randn(*[*sample_shape, *shape, guide.max_num_shapes, 2])

    log_prob = guide.log_prob(obs, (program_id, shape_ids, raw_positions))

    assert list(log_prob.shape) == sample_shape + shape


def test_guide_sample_discrete_dims():
    im_size = 64
    shape = [2, 3]
    sample_shape = [4, 5]

    guide = Guide(im_size=im_size)
    obs = torch.rand(*[*shape, im_size, im_size])

    program_id, shape_ids = guide.sample_discrete(obs, sample_shape)

    assert list(program_id.shape) == sample_shape + shape
    assert list(shape_ids.shape) == sample_shape + shape + [guide.max_num_shapes]
