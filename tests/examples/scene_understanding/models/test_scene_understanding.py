import torch
import cmws
from cmws.examples.scene_understanding.models.scene_understanding import GenerativeModel


def test_generative_model_latent_log_prob_dims():
    num_grid_rows, num_grid_cols, num_primitives, max_num_blocks = 3, 3, 5, 3
    shape = [2, 3]

    generative_model = GenerativeModel(num_grid_rows, num_grid_cols, num_primitives, max_num_blocks)

    num_blocks = torch.randint(0, max_num_blocks, shape + [num_grid_rows, num_grid_cols])
    stacking_program = torch.randint(
        0, num_primitives, shape + [num_grid_rows, num_grid_cols, max_num_blocks]
    )
    raw_locations = torch.randn(shape + [num_grid_rows, num_grid_cols, max_num_blocks])
    latent = num_blocks, stacking_program, raw_locations
    log_prob = generative_model.latent_log_prob(latent)

    assert list(log_prob.shape) == shape


def test_generative_model_latent_sample_dims():
    num_grid_rows, num_grid_cols, num_primitives, max_num_blocks = 3, 3, 5, 3
    sample_shape = [2, 3]

    generative_model = GenerativeModel(num_grid_rows, num_grid_cols, num_primitives, max_num_blocks)
    num_blocks, stacking_program, raw_locations = generative_model.latent_sample(sample_shape)

    assert list(num_blocks.shape) == sample_shape + [num_grid_rows, num_grid_cols]
    assert list(stacking_program.shape) == sample_shape + [
        num_grid_rows,
        num_grid_cols,
        max_num_blocks,
    ]
    assert list(raw_locations.shape) == sample_shape + [
        num_grid_rows,
        num_grid_cols,
        max_num_blocks,
    ]


def test_generative_model_discrete_latent_sample_dims():
    num_grid_rows, num_grid_cols, num_primitives, max_num_blocks = 3, 3, 5, 3
    sample_shape = [2, 3]

    generative_model = GenerativeModel(num_grid_rows, num_grid_cols, num_primitives, max_num_blocks)
    num_blocks, stacking_program = generative_model.discrete_latent_sample(sample_shape)

    assert list(num_blocks.shape) == sample_shape + [num_grid_rows, num_grid_cols]
    assert list(stacking_program.shape) == sample_shape + [
        num_grid_rows,
        num_grid_cols,
        max_num_blocks,
    ]


def test_generative_model_log_prob_dims():
    device = cmws.util.get_device()
    # num_grid_rows, num_grid_cols, num_primitives, max_num_blocks = 3, 3, 5, 3
    num_grid_rows, num_grid_cols, num_primitives, max_num_blocks = 2,2, 5, 3
    sample_shape = [2, 3]
    shape = [4]

    generative_model = GenerativeModel(
        num_grid_rows, num_grid_cols, num_primitives, max_num_blocks
    ).to(device)

    num_blocks = torch.randint(
        0, max_num_blocks, sample_shape + shape + [num_grid_rows, num_grid_cols], device=device
    )
    stacking_program = torch.randint(
        0,
        num_primitives,
        sample_shape + shape + [num_grid_rows, num_grid_cols, max_num_blocks],
        device=device,
    )
    raw_locations = torch.randn(
        sample_shape + shape + [num_grid_rows, num_grid_cols, max_num_blocks], device=device
    )
    latent = num_blocks, stacking_program, raw_locations
    obs = torch.rand(shape + [3, 32, 32], device=device)
    log_prob = generative_model.log_prob(latent, obs)

    assert list(log_prob.shape) == sample_shape + shape
