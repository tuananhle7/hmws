import pathlib

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
    elif (
        run_args.model_type == "shape_program_pyro"
        or run_args.model_type == "shape_program_pytorch"
    ):
        true_generative_model = shape_program.TrueGenerativeModelFixedScale().to(device)

    # Generate data
    _, obs = true_generative_model.sample((num_obs,))

    return obs


@torch.no_grad()
def generate_test_obs(run_args, device):
    return generate_obs(run_args, 10, device, seed=1)


class CSGDataset(torch.utils.data.Dataset):
    """Loads or generates a dataset
    Uses ~1.6M (test) / 160MB (train)

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
        self.num_train_data = 10000
        self.num_test_data = 100
        if self.test:
            self.num_data = self.num_test_data
        else:
            self.num_data = self.num_train_data

        path = (
            pathlib.Path(__file__)
            .parent.absolute()
            .joinpath("data", "shape_program", "test.pt" if self.test else "train.pt")
        )
        if force_regenerate or not path.exists():
            util.logging.info(f"Generating dataset (test = {self.test})...")

            # Make path
            pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

            # Set seed
            util.set_seed(seed)

            # Generate new dataset
            # --Make true generative model
            true_generative_model = shape_program.TrueGenerativeModelFixedScale().to(device)

            # --Generate data
            _, self.obs = true_generative_model.sample((self.num_data,))
            self.obs_id = torch.arange(self.num_data, device=device)

            # Save dataset
            torch.save([self.obs, self.obs_id], path)
            util.logging.info(f"Dataset (test = {self.test}) generated and saved to {path}")
        else:
            util.logging.info(f"Loading dataset (test = {self.test})...")

            # Load dataset
            self.obs, self.obs_id = torch.load(path, map_location=device)
            util.logging.info(f"Dataset (test = {self.test}) loaded {path}")

    def __getitem__(self, idx):
        return self.obs[idx], self.obs_id[idx]

    def __len__(self):
        return self.num_data


def get_csg_data_loader(device, batch_size, test=False):
    if test:
        shuffle = True
    else:
        shuffle = False
    return torch.utils.data.DataLoader(
        CSGDataset(device, test=test), batch_size=batch_size, shuffle=shuffle
    )
