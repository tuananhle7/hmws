import pyro
import torch
from cmws import util
from cmws.examples.stacking_3d import render
import pathlib


def sample_stacking_program(num_primitives, device, address_suffix="", fixed_num_blocks=False):
    """Samples blocks to stack from a set [0, ..., num_primitives - 1]
    *without* replacement. The number of blocks is stochastic and
    can be < num_primitives.

    Args
        num_primitives (int)
        device
        address_suffix

    Returns [num_blocks] (where num_blocks is stochastic and between 1 and num_primitives
        (inclusive))
    """

    # Init
    stacking_program = []
    available_primitive_ids = list(range(num_primitives))

    if fixed_num_blocks:
        num_blocks = num_primitives
    else:
        # Sample num_blocks uniformly from [1, ..., num_primitives] (inclusive)
        raw_num_blocks_logits = torch.ones((num_primitives,), device=device)
        raw_num_blocks = pyro.sample(
            f"raw_num_blocks{address_suffix}",
            pyro.distributions.Categorical(logits=raw_num_blocks_logits),
        )
        num_blocks = raw_num_blocks + 1

    # Sample primitive ids
    for block_id in range(num_blocks):
        # Sample primitive
        raw_primitive_id_logits = torch.ones((len(available_primitive_ids),), device=device)
        raw_primitive_id = pyro.sample(
            f"raw_primitive_id_{block_id}{address_suffix}",
            pyro.distributions.Categorical(logits=raw_primitive_id_logits),
        )
        primitive_id = available_primitive_ids.pop(raw_primitive_id)

        # Add to the stacking program based on previous action
        stacking_program.append(primitive_id)

    return torch.tensor(stacking_program, device=device)


def stacking_program_to_str(stacking_program, primitives):
    return [primitives[primitive_id].name for primitive_id in stacking_program]


def sample_raw_locations(stacking_program, address_suffix=""):
    """
    Samples the (raw) horizontal location of blocks in the stacking program.
    p(raw_locations | stacking_program)

    Args
        stacking_program [num_blocks]

    Returns [num_blocks]
    """
    device = stacking_program[0].device
    dist = pyro.distributions.Independent(
        pyro.distributions.Normal(torch.zeros((len(stacking_program),), device=device), 1),
        reinterpreted_batch_ndims=1,
    )
    return pyro.sample(f"raw_locations{address_suffix}", dist)


def generate_from_true_generative_model_single(
    device, num_primitives, num_channels=3, im_size=32, fixed_num_blocks=False
):
    """Generate a synthetic observation

    Returns [num_channels, im_size, im_size]
    """
    assert num_primitives <= 3
    # Define params
    primitives = [
        render.Cube(
            "A", torch.tensor([1.0, 0.0, 0.0], device=device), torch.tensor(0.3, device=device)
        ),
        render.Cube(
            "B", torch.tensor([0.0, 1.0, 0.0], device=device), torch.tensor(0.4, device=device)
        ),
        render.Cube(
            "C", torch.tensor([0.0, 0.0, 1.0], device=device), torch.tensor(0.5, device=device)
        ),
    ][:num_primitives]
    # num_primitives = len(primitives)

    # Sample
    stacking_program = sample_stacking_program(
        num_primitives, device, fixed_num_blocks=fixed_num_blocks
    )
    raw_locations = sample_raw_locations(stacking_program)

    # Render
    img = render.render(
        primitives,
        torch.tensor(len(stacking_program), device=device).long(),
        stacking_program,
        raw_locations,
        im_size,
    )
    assert len(img.shape) == 3

    return img


def generate_from_true_generative_model(
    batch_size, num_primitives, device, num_channels=3, im_size=32, fixed_num_blocks=False,
):
    """Generate a batch of synthetic observations

    Returns [batch_size, num_channels, im_size, im_size]
    """
    return torch.stack(
        [
            generate_from_true_generative_model_single(
                device, num_primitives, im_size=im_size, fixed_num_blocks=fixed_num_blocks,
            )
            for _ in range(batch_size)
        ]
    )


@torch.no_grad()
def generate_obs(num_obs, device, seed=None):
    if seed is not None:
        # Fix seed
        util.set_seed(seed)

    obs = generate_from_true_generative_model(num_obs, num_primitives=3, device=device)

    return obs


@torch.no_grad()
def generate_test_obs(device):
    return generate_obs(10, device, seed=1)


class StackingDataset(torch.utils.data.Dataset):
    """Loads or generates a dataset
    Uses ~1.2M (test) / 120MB (train)

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
            .joinpath("data", "test.pt" if self.test else "train.pt")
        )
        if force_regenerate or not path.exists():
            util.logging.info(f"Generating dataset (test = {self.test})...")

            # Make path
            pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

            # Set seed
            util.set_seed(seed)

            # Generate new dataset
            self.obs = generate_from_true_generative_model(
                self.num_data, num_primitives=3, device=device
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


def get_stacking_data_loader(device, batch_size, test=False):
    if test:
        shuffle = True
    else:
        shuffle = False
    return torch.utils.data.DataLoader(
        StackingDataset(device, test=test), batch_size=batch_size, shuffle=shuffle
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from models import stacking

    num_samples = 5
    path = "test/samples.png"
    generative_model = stacking.GenerativeModel(im_size=256)
    latent, obs = generative_model.sample((num_samples,))
    fig, axs = plt.subplots(1, num_samples, figsize=(num_samples * 4, 4))
    for i in range(num_samples):
        axs[i].imshow(obs.cpu().detach()[i].permute(1, 2, 0))
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    util.save_fig(fig, path)

    path = "test/data.png"
    obs = generate_obs(num_samples, util.get_device())
    fig, axs = plt.subplots(1, num_samples, figsize=(num_samples * 4, 4))
    for i in range(num_samples):
        axs[i].imshow(obs.cpu().detach()[i].permute(1, 2, 0))
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    util.save_fig(fig, path)
