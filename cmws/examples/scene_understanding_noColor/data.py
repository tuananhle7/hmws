import itertools
import random
import torch
from cmws import util
from cmws.examples.scene_understanding import render
import pathlib
import matplotlib.pyplot as plt


def generate_from_true_generative_model_single(
    device,
    num_grid_rows,
    num_grid_cols,
    num_primitives,
    max_num_blocks=3,
    num_channels=3,
    im_size=32,
    fixed_num_blocks=False,
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

    # Determine which cells have stacks
    cells = list(itertools.product(range(num_grid_rows), range(num_grid_cols)))
    num_cells = num_grid_rows * num_grid_cols
    num_stacks = random.randint(1, num_cells)
    cells_with_stack = set(random.sample(cells, num_stacks))

    # Sample
    num_blocks = []
    stacking_program = []
    raw_locations = []
    for row in range(num_grid_rows):
        for col in range(num_grid_cols):
            if (row, col) in cells_with_stack:
                num_blocks_ = random.randint(1, max_num_blocks)
                stacking_program_ = torch.randint(0, num_primitives, [num_blocks_], device=device)
                raw_locations_ = torch.randn(num_blocks_, device=device)
                num_blocks.append(num_blocks_)
                stacking_program_padded = torch.zeros(max_num_blocks, device=device).long()
                stacking_program_padded[: num_blocks[-1]] = stacking_program_
                raw_locations_padded = torch.zeros(max_num_blocks, device=device)
                raw_locations_padded[: num_blocks[-1]] = raw_locations_
                stacking_program.append(stacking_program_padded)
                raw_locations.append(raw_locations_padded)
            else:
                num_blocks.append(0)
                stacking_program.append(torch.zeros(max_num_blocks, device=device).long())
                raw_locations.append(torch.zeros(max_num_blocks, device=device))
    num_blocks = torch.tensor(num_blocks, device=device).long().view(num_grid_rows, num_grid_cols)
    stacking_program = torch.stack(stacking_program).view(
        num_grid_rows, num_grid_rows, max_num_blocks
    )
    raw_locations = torch.stack(raw_locations).view(num_grid_rows, num_grid_rows, max_num_blocks)

    # Render
    img = render.render(primitives, num_blocks, stacking_program, raw_locations, im_size,)
    assert len(img.shape) == 3

    return img


def generate_from_true_generative_model(
    batch_size,
    num_grid_rows,
    num_grid_cols,
    num_primitives,
    device,
    num_channels=3,
    im_size=128,
    fixed_num_blocks=False,
):
    """Generate a batch of synthetic observations

    Returns [batch_size, num_channels, im_size, im_size]
    """
    return torch.stack(
        [
            generate_from_true_generative_model_single(
                device,
                num_grid_rows,
                num_grid_cols,
                num_primitives,
                im_size=im_size,
                fixed_num_blocks=fixed_num_blocks,
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


class SceneUnderstandingDataset(torch.utils.data.Dataset):
    """Loads or generates a dataset
    Uses ~1.2M (test) / 120MB (train)

    Args
        device
        num_grid_rows
        num_grid_cols
        test (bool; default: False)
        force_regenerate (bool; default: False): if False, the dataset is loaded if it exists
            if True, the dataset is regenerated regardless
        seed (int): only used for generation
    """

    def __init__(
        self, device, num_grid_rows=1, num_grid_cols=1, test=False, force_regenerate=False, seed=1
    ):
        self.device = device
        self.num_grid_rows = num_grid_rows
        self.num_grid_cols = num_grid_cols
        self.test = test
        self.num_train_data = 100
        self.num_test_data = 100
        if self.test:
            self.num_data = self.num_test_data
        else:
            self.num_data = self.num_train_data

        path = (
            pathlib.Path(__file__)
            .parent.absolute()
            .joinpath(
                "data",
                f"{self.num_grid_rows}_{self.num_grid_cols}",
                "test.pt" if self.test else "train.pt",
            )
        )
        if force_regenerate or not path.exists():
            util.logging.info(f"Generating dataset (test = {self.test})...")

            # Make path
            pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

            # Set seed
            util.set_seed(seed)

            # Generate new dataset
            self.obs = generate_from_true_generative_model(
                self.num_data,
                num_grid_rows=num_grid_rows,
                num_grid_cols=num_grid_cols,
                num_primitives=3,
                device=device,
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


def get_scene_understanding_data_loader(
    device, num_grid_rows, num_grid_cols, batch_size, test=False
):
    if test:
        shuffle = False
    else:
        shuffle = True
    return torch.utils.data.DataLoader(
        SceneUnderstandingDataset(device, num_grid_rows, num_grid_cols, test=test),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def plot_data():
    device = torch.device("cuda")

    for num_grid_rows, num_grid_cols in [[3, 3], [1, 1], [2, 2]]:
        # Plot train / test data
        timeseries_dataset = {}

        # Train
        timeseries_dataset["train"] = SceneUnderstandingDataset(
            device, num_grid_rows, num_grid_cols
        )

        # Test
        timeseries_dataset["test"] = SceneUnderstandingDataset(
            device, num_grid_rows, num_grid_cols, test=True
        )

        for mode in ["test", "train"]:
            start = 0
            end = 100

            while start < len(timeseries_dataset[mode]):
                obs, obs_id = timeseries_dataset[mode][start:end]
                path = f"./data/{num_grid_rows}_{num_grid_cols}/plots/{mode}/{start:05.0f}_{end:05.0f}.png"

                fig, axss = plt.subplots(10, 10, sharex=True, sharey=True, figsize=(10 * 3, 10 * 3))

                for i in range(len(obs)):
                    ax = axss.flat[i]
                    ax.imshow(obs[i].permute(1, 2, 0).detach().cpu())
                    ax.set_xticks([])
                    ax.set_yticks([])

                util.save_fig(fig, path)

                break

                start = end
                end += 100


if __name__ == "__main__":
    plot_data()
