from models import stacking_pyro
import util


def generate_obs(run_args, num_obs, device, seed=None):
    if seed is not None:
        # Fix seed
        util.set_seed(seed)

    if run_args.model_type == "stacking_pyro":
        obs = stacking_pyro.generate_from_true_generative_model(
            num_obs, num_primitives=run_args.data_num_primitives, device=device
        )
    elif run_args.model_type == "one_primitive":
        obs = stacking_pyro.generate_from_true_generative_model(
            num_obs, num_primitives=1, device=device
        )
    elif run_args.model_type == "two_primitives":
        obs = stacking_pyro.generate_from_true_generative_model(
            num_obs, num_primitives=2, device=device, fixed_num_blocks=True
        )
    elif run_args.model_type == "stacking":
        obs = stacking_pyro.generate_from_true_generative_model(
            num_obs, num_primitives=3, device=device
        )

    return obs


def generate_test_obs(run_args, device):
    return generate_obs(run_args, 10, device, seed=1)