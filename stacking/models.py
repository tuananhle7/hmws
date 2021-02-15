import pyro
import render
import torch
import torch.nn as nn


def sample_stacking_program(num_primitives, device, address_suffix=""):
    # Init
    stacking_program = []
    num_sampled_primitives = 0
    available_primitive_ids = list(range(num_primitives))

    # Sample first primitive
    raw_primitive_id_logits = torch.ones((len(available_primitive_ids),), device=device)
    raw_primitive_id = pyro.sample(
        f"raw_primitive_id_{num_sampled_primitives}{address_suffix}",
        pyro.distributions.Categorical(logits=raw_primitive_id_logits),
    )
    primitive_id = available_primitive_ids.pop(raw_primitive_id)
    num_sampled_primitives += 1
    stacking_program.append(primitive_id)

    # Sample the rest
    end_program = False
    while (not end_program) and (len(available_primitive_ids) > 0):
        # Sample an action for the next primitive
        # Action 0: put to the left of existing stack
        # Action 1: end program
        num_actions = 2
        action_id_logits = torch.ones((num_actions,), device=device)
        action_id = pyro.sample(
            f"action_id_{num_sampled_primitives}{address_suffix}",
            pyro.distributions.Categorical(logits=action_id_logits),
        )

        if action_id == 1:
            # End program
            end_program = True
            break
        else:
            # Sample primitive
            raw_primitive_id_logits = torch.ones((len(available_primitive_ids),), device=device)
            raw_primitive_id = pyro.sample(
                f"raw_primitive_id_{num_sampled_primitives}{address_suffix}",
                pyro.distributions.Categorical(logits=raw_primitive_id_logits),
            )
            primitive_id = available_primitive_ids.pop(raw_primitive_id)
            num_sampled_primitives += 1

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
    return pyro.sample(f"raw_loc{address_suffix}", dist)


def generate_from_true_generative_model(device, num_channels=3, num_rows=256, num_cols=256):
    # Define params
    primitives = [
        render.Square(
            "A", torch.tensor([1.0, 0.0, 0.0], device=device), torch.tensor(0.1, device=device)
        ),
        render.Square(
            "B", torch.tensor([0.0, 1.0, 0.0], device=device), torch.tensor(0.2, device=device)
        ),
        render.Square(
            "C", torch.tensor([0.0, 0.0, 1.0], device=device), torch.tensor(0.3, device=device)
        ),
    ]
    num_primitives = len(primitives)

    # Sample
    stacking_program = sample_stacking_program(num_primitives, device)
    raw_locations = sample_raw_locations(stacking_program)

    # Render
    img = render.render(
        primitives, stacking_program, raw_locations, num_channels, num_rows, num_cols
    )

    return img


class GenerativeModel(nn.Module):
    """First samples the stacking program (discrete), then their raw positions (continuous),
    then renders them onto an image.
    """

    def __init__(self, im_size=64, num_primitives=3):
        super().__init__()

        # Init
        self.num_channels = 3
        self.num_rows = im_size
        self.num_cols = im_size
        self.num_primitives = num_primitives

        # Primitive parameters (parameters of symbols)
        self.primitives = nn.ModuleList(
            [render.LearnableSquare(f"{i}") for i in range(self.num_primitives)]
        )

        # Rendering parameters
        self.raw_color_sharpness = nn.Parameter(torch.rand(()))
        self.raw_blur = nn.Parameter(torch.rand(()))

    @property
    def device(self):
        return self.primitives[0].device

    def forward(self, obs, observations=None):
        """
        Args:
            obs [batch_size, num_channels, num_rows, num_cols]
        """
        pyro.module("generative_model", self)

        # Extract
        if isinstance(observations, dict):
            batch_size = len(observations)
            obs = torch.stack([observations[f"obs_{i}"] for i in range(batch_size)])
        batch_size, num_channels, num_rows, num_cols = obs.shape
        assert num_channels == self.num_channels
        assert num_rows == self.num_rows
        assert num_cols == self.num_cols

        traces = []
        for batch_id in pyro.plate("batch", batch_size):
            # p(stacking_program)
            stacking_program = sample_stacking_program(
                self.num_primitives, self.device, address_suffix=f"{batch_id}"
            )

            # p(raw_locations | stacking_program)
            raw_locations = sample_raw_locations(stacking_program, address_suffix=f"{batch_id}")

            # p(obs | raw_locations, stacking_program)
            loc = render.soft_render(
                self.primitives,
                stacking_program,
                raw_locations,
                self.raw_color_sharpness,
                self.raw_blur,
                num_channels=self.num_channels,
                num_rows=self.num_rows,
                num_cols=self.num_cols,
            )
            sampled_obs = pyro.sample(
                f"obs_{batch_id}",
                pyro.distributions.Independent(
                    pyro.distributions.Normal(loc=loc, scale=1.0), reinterpreted_batch_ndims=3,
                ),
                obs=obs[batch_id],
            )
            traces.append((stacking_program, raw_locations, sampled_obs))
        return traces


if __name__ == "__main__":
    # Init
    device = "cuda"
    batch_size = 3
    im_size = 64

    # Create obs
    obs = torch.stack(
        [
            generate_from_true_generative_model(device, num_rows=im_size, num_cols=im_size)
            for _ in range(batch_size)
        ]
    )

    # Run through learnable generative model
    generative_model = GenerativeModel().to(device)

    print(generative_model.forward(obs))
