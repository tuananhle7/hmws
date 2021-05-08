import torch
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
