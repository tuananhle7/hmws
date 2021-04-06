import torch
from cmws import util
from cmws.examples.csg.models import (
    heartangles,
    hearts,
    ldif_representation,
    no_rectangle,
    shape_program,
)


@torch.no_grad()
def generate_obs(run_args, num_obs, device, seed=None):
    """
    Returns [num_obs, ...]
    """
    if seed is not None:
        # Fix seed
        util.set_seed(seed)

    # Initialize true generative model
    if run_args.model_type == "hearts" or run_args.model_type == "hearts_pyro":
        true_generative_model = hearts.TrueGenerativeModel().to(device)
    elif run_args.model_type == "heartangles":
        true_generative_model = heartangles.TrueGenerativeModel().to(device)
    elif run_args.model_type == "shape_program":
        true_generative_model = shape_program.TrueGenerativeModel().to(device)
    elif (
        run_args.model_type == "no_rectangle"
        or run_args.model_type == "neural_boundary"
        or run_args.model_type == "neural_boundary_pyro"
    ):
        true_generative_model = no_rectangle.TrueGenerativeModel(
            has_shape_scale=run_args.data_has_shape_scale
        ).to(device)
    elif (
        run_args.model_type == "ldif_representation"
        or run_args.model_type == "ldif_representation_pyro"
    ):
        true_generative_model = ldif_representation.TrueGenerativeModel().to(device)
    elif run_args.model_type == "shape_program_pyro":
        true_generative_model = shape_program.TrueGenerativeModelFixedScale().to(device)

    # Generate data
    _, obs = true_generative_model.sample((num_obs,))

    return obs


@torch.no_grad()
def generate_test_obs(run_args, device):
    return generate_obs(run_args, 10, device, seed=1)
