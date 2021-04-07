import pathlib

import torch
from cmws import util
from cmws.examples.stacking.models import stacking_pyro, stacking_with_attachment


@torch.no_grad()
def generate_obs(run_args, num_obs, device, seed=None):
    """
    Returns [num_obs, num_channels, im_size, im_size]
    """
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
    elif run_args.model_type == "stacking_top_down":
        obs = stacking_pyro.generate_from_true_generative_model_top_down(
            num_obs, num_primitives=3, device=device
        )
    elif run_args.model_type == "stacking_with_attachment":
        # Generative model with true primitives
        true_generative_model = stacking_with_attachment.GenerativeModel(
            num_primitives=3, max_num_blocks=run_args.max_num_blocks, true_primitives=True,
        ).to(device)
        _, obs = true_generative_model.sample((num_obs,))

    return obs


@torch.no_grad()
def generate_test_obs(run_args, device):
    return generate_obs(run_args, 10, device, seed=1)


class StackingDataset(torch.utils.data.Dataset):
    """Loads or generates a dataset
    (Takes ~1.2GB of GPU memory)

    Args
        device
        test (bool; default: False)
        force_regenerate (bool; default: False): if False, the dataset is loaded if it exists
            if True, the dataset is regenerated regardless
        seed (int): only used for generation
    """

    def __init__(self, device, test=False, force_regenerate=False, seed=1):
        self.device = device
        self.test = test
        self.num_train_data = 100000
        self.num_test_data = 1000
        if self.test:
            self.num_data = self.num_test_data
        else:
            self.num_data = self.num_train_data

        path = pathlib.Path(__file__).parent.absolute().joinpath("data", "stacking.pt")
        if force_regenerate or not path.exists():
            util.logging.info("Generating dataset...")

            # Make path
            pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

            # Set seed
            util.set_seed(seed)

            # Generate new dataset
            # --Train
            train_obs = stacking_pyro.generate_from_true_generative_model(
                self.num_train_data, num_primitives=3, device=device
            )
            train_obs_id = torch.arange(self.num_train_data, device=device)

            # --Test
            test_obs = stacking_pyro.generate_from_true_generative_model(
                self.num_test_data, num_primitives=3, device=device
            )
            test_obs_id = torch.arange(self.num_test_data, device=device)

            # Save dataset
            torch.save([train_obs, train_obs_id, test_obs, test_obs_id], path)
            util.logging.info(f"Dataset generated and saved to {path}")
        else:
            util.logging.info("Loading dataset...")

            # Load dataset
            train_obs, train_obs_id, test_obs, test_obs_id = torch.load(path, map_location=device)
            util.logging.info(f"Dataset loaded {path}")

        if self.test:
            self.obs, self.obs_id = test_obs, test_obs_id
            train_obs, train_obs_id = None, None
        else:
            self.obs, self.obs_id = train_obs, train_obs_id
            test_obs, test_obs_id = None, None

    def __getitem__(self, idx):
        return self.obs[idx], self.obs_id[idx]

    def __len__(self):
        return self.num_data
