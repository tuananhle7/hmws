import pathlib
import torch
from cmws import util
import ssm
from ssm.util import random_rotation
import numpy as np


def sample_slds(num_timesteps, num_states, continuous_dim, obs_dim):
    # Make an SLDS with the true parameters
    true_slds = ssm.SLDS(obs_dim, num_states, continuous_dim, emissions="gaussian_orthog")

    for k in range(num_states):
        true_slds.dynamics.As[k] = 0.95 * random_rotation(
            continuous_dim, theta=(k + 1) * np.pi / 20
        )

    states_z, states_x, emissions = true_slds.sample(num_timesteps)
    return emissions


def generate_synthetic_data(
    num_data, device, num_timesteps=100, num_states=5, continuous_dim=2, obs_dim=10
):
    return torch.stack(
        [
            torch.tensor(
                sample_slds(num_timesteps, num_states, continuous_dim, obs_dim), device=device
            ).float()
            for _ in range(num_data)
        ]
    )


class SLDSDataset(torch.utils.data.Dataset):
    """Loads or generates a dataset

    Args
        device
        test (bool; default: False)
    """

    def __init__(
        self, device, test=False, num_timesteps=100, num_states=5, continuous_dim=2, obs_dim=10
    ):
        self.device = device
        self.test = test
        if self.test:
            self.num_data = 1
        else:
            self.num_data = 1
        path = (
            pathlib.Path(__file__)
            .parent.absolute()
            .joinpath(
                "data",
                f"{num_timesteps}_{num_states}_{continuous_dim}_{obs_dim}",
                "test.pt" if self.test else "train.pt",
            )
        )
        if not path.exists():
            util.logging.info(f"Generating dataset (test = {self.test})...")

            # Make path
            pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

            # Set seed
            # util.set_seed(1 if self.test else 0)
            util.set_seed(0)

            self.obs = generate_synthetic_data(
                self.num_data,
                device,
                num_timesteps=num_timesteps,
                num_states=num_states,
                continuous_dim=continuous_dim,
                obs_dim=obs_dim,
            )
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


def get_slds_data_loader(
    device, batch_size, test=False, num_timesteps=100, num_states=5, continuous_dim=2, obs_dim=10
):
    if test:
        shuffle = False
    else:
        shuffle = True
    return torch.utils.data.DataLoader(
        SLDSDataset(
            device,
            test=test,
            num_timesteps=num_timesteps,
            num_states=num_states,
            continuous_dim=continuous_dim,
            obs_dim=obs_dim,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )
