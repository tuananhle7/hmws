import pyro
import render
import torch
import torch.nn as nn
import torchvision.models


def sample_stacking_program(num_primitives, device, address_suffix=""):
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
    device, num_primitives, num_channels=3, num_rows=64, num_cols=64
):
    """Generate a synthetic observation

    Returns [num_channels, num_rows, num_cols]
    """
    assert num_primitives <= 3
    # Define params
    primitives = [
        render.Square(
            "A", torch.tensor([1.0, 0.0, 0.0], device=device), torch.tensor(0.2, device=device)
        ),
        render.Square(
            "B", torch.tensor([0.0, 1.0, 0.0], device=device), torch.tensor(0.3, device=device)
        ),
        render.Square(
            "C", torch.tensor([0.0, 0.0, 1.0], device=device), torch.tensor(0.4, device=device)
        ),
    ][:num_primitives]
    # num_primitives = len(primitives)

    # Sample
    stacking_program = sample_stacking_program(num_primitives, device)
    raw_locations = sample_raw_locations(stacking_program)

    # Render
    img = render.render(
        primitives, stacking_program, raw_locations, num_channels, num_rows, num_cols
    )

    return img


def generate_from_true_generative_model(
    batch_size, num_primitives, device, num_channels=3, num_rows=64, num_cols=64
):
    """Generate a batch of synthetic observations

    Returns [batch_size, num_channels, num_rows, num_cols]
    """
    return torch.stack(
        [
            generate_from_true_generative_model_single(
                device, num_primitives, num_rows=num_rows, num_cols=num_cols
            )
            for _ in range(batch_size)
        ]
    )


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
                self.num_primitives, self.device, address_suffix=f"_{batch_id}"
            )

            # p(raw_locations | stacking_program)
            raw_locations = sample_raw_locations(stacking_program, address_suffix=f"_{batch_id}")

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

            # ABC likelihood
            # sampling_scale = 0.01
            # log_prob_scale = 1.0
            # abc_log_prob = torch.distributions.Independent(
            #     torch.distributions.Normal(loc=loc, scale=log_prob_scale),
            #     reinterpreted_batch_ndims=3,
            # ).log_prob(obs[batch_id])
            # sampled_obs_log_prob = torch.distributions.Independent(
            #     torch.distributions.Normal(loc=loc, scale=sampling_scale),
            #     reinterpreted_batch_ndims=3,
            # ).log_prob(obs[batch_id])
            sampled_obs = pyro.sample(
                f"obs_{batch_id}",
                pyro.distributions.Independent(
                    pyro.distributions.Normal(loc=loc, scale=1.0), reinterpreted_batch_ndims=3,
                ),
                obs=obs[batch_id],
            )
            # pyro.factor(f"L2_{batch_id}", abc_log_prob - sampled_obs_log_prob)
            traces.append((stacking_program, raw_locations, sampled_obs))
        return traces


class Guide(nn.Module):
    """
    This uses an inference compilation architecture

    LSTM input: [obs_embedding, prev_sample_embedding, address_embedding, instance_embedding]
    LSTM output --> distribution params
    """

    def __init__(self, im_size=64, num_primitives=3):
        super().__init__()

        # Init
        self.num_channels = 3
        self.num_rows = im_size
        self.num_cols = im_size
        self.num_primitives = num_primitives

        # Obs embedder
        self.obs_embedding_dim = 100
        self.obs_embedder = torchvision.models.alexnet(
            pretrained=False, num_classes=self.obs_embedding_dim
        )

        # LSTM
        self.sample_embedding_dim = 16
        self.address_embedding_dim = 16
        self.instance_embedding_dim = 16
        self.lstm_input_dim = (
            self.obs_embedding_dim
            + self.sample_embedding_dim
            + self.address_embedding_dim
            + self.instance_embedding_dim
        )
        self.lstm_hidden_dim = 256
        self.lstm_cell = torch.nn.LSTMCell(self.lstm_input_dim, self.lstm_hidden_dim)

        # (Previous) Sample embedders
        # --Raw num blocks
        self.raw_num_blocks_embeddings = nn.Parameter(
            torch.randn((self.num_primitives, self.sample_embedding_dim))
        )

        # --Raw primitive id
        self.raw_primitive_id_embeddings = nn.Parameter(
            torch.randn((self.num_primitives, self.sample_embedding_dim))
        )

        # --Raw locations
        # NOTE: don't need to be embedded because they're the last latent variable

        # Address embeddings
        self.address_to_id = {"raw_num_blocks": 0, "raw_primitive_id": 1, "raw_locations": 2}
        self.num_addresses = len(self.address_to_id)
        self.address_embeddings = nn.Parameter(
            torch.randn((self.num_addresses, self.address_embedding_dim))
        )

        # Instance embeddings
        self.num_instance_embeddings = self.num_primitives
        self.instance_embeddings = nn.Parameter(
            torch.randn((self.num_instance_embeddings, self.instance_embedding_dim))
        )

        # Param extractors
        # --Raw num blocks
        self.raw_num_blocks_param_extractor = nn.Linear(self.lstm_hidden_dim, self.num_primitives)

        # --Raw primitive id
        self.raw_primitive_id_param_extractor = nn.Linear(self.lstm_hidden_dim, self.num_primitives)

        # --Raw locations
        self.raw_locations_param_extractor = nn.Linear(
            self.lstm_hidden_dim, self.num_primitives * 2
        )

    @property
    def device(self):
        return self.raw_num_blocks_embeddings.device

    def get_address_embedding(self, address):
        return self.address_embeddings[self.address_to_id[address]]

    def get_obs_embedding(self, obs):
        """
        Args:
            obs: [batch_size, num_channels, num_rows, num_cols]

        Returns: [batch_size, obs_embedding_dim]
        """
        return self.obs_embedder(obs)

    def forward(self, obs, observations=None):
        """
        Args:
            obs: [batch_size, num_channels, num_rows, num_cols]

        Returns:
            shape_id [batch_size]
            raw_position [batch_size, 2]
        """
        pyro.module("guide", self)
        if isinstance(observations, dict):
            batch_size = len(observations)
            obs = torch.stack([observations[f"obs_{i}"] for i in range(batch_size)])
        batch_size, num_channels, num_rows, num_cols = obs.shape

        # Get obs embedding
        obs_embedding = self.get_obs_embedding(obs)

        traces = []
        for batch_id in pyro.plate("batch", batch_size):
            # Extract obs embedding
            obs_embedding_b = obs_embedding[batch_id]

            # Sample raw num blocks
            # --LSTM Input
            prev_sample_embedding = torch.zeros((self.sample_embedding_dim,), device=self.device)
            address_embedding = self.get_address_embedding("raw_num_blocks")
            instance_embedding = self.instance_embeddings[0]
            lstm_input = torch.cat(
                [obs_embedding_b, prev_sample_embedding, address_embedding, instance_embedding]
            )

            # --Run LSTM
            h, c = self.lstm_cell(lstm_input[None])
            hc = h, c

            # --Extract params
            raw_num_blocks_logits = self.raw_num_blocks_param_extractor(h)[0]

            # --Sample raw_num_blocks
            raw_num_blocks = pyro.sample(
                f"raw_num_blocks_{batch_id}",
                pyro.distributions.Categorical(logits=raw_num_blocks_logits),
            ).long()
            num_blocks = raw_num_blocks + 1

            # Sample primitive ids
            stacking_program = []
            available_primitive_ids = list(range(self.num_primitives))
            for block_id in range(num_blocks):
                # --LSTM Input
                if block_id == 0:
                    prev_sample_embedding = self.raw_num_blocks_embeddings[raw_num_blocks]
                else:
                    prev_sample_embedding = self.raw_primitive_id_embeddings[raw_primitive_id]
                address_embedding = self.get_address_embedding("raw_primitive_id")
                instance_embedding = self.instance_embeddings[block_id]
                lstm_input = torch.cat(
                    [obs_embedding_b, prev_sample_embedding, address_embedding, instance_embedding]
                )

                # --Run LSTM
                h, c = self.lstm_cell(lstm_input[None], hc)
                hc = h, c

                # --Extract params
                raw_primitive_id_logits = self.raw_primitive_id_param_extractor(h)[0][
                    : len(available_primitive_ids)
                ]

                # --Sample raw_primitive_id
                raw_primitive_id = pyro.sample(
                    f"raw_primitive_id_{block_id}_{batch_id}",
                    pyro.distributions.Categorical(logits=raw_primitive_id_logits),
                ).long()
                primitive_id = available_primitive_ids.pop(raw_primitive_id)

                # Add to the stacking program based on previous action
                stacking_program.append(primitive_id)

            # Sample raw locations
            # --LSTM Input
            prev_sample_embedding = self.raw_primitive_id_embeddings[raw_primitive_id]
            address_embedding = self.get_address_embedding("raw_locations")
            instance_embedding = self.instance_embeddings[0]
            lstm_input = torch.cat(
                [obs_embedding_b, prev_sample_embedding, address_embedding, instance_embedding]
            )

            # --Run LSTM
            h, c = self.lstm_cell(lstm_input[None], hc)
            hc = h, c

            # --Extract params
            raw_locations_param = self.raw_locations_param_extractor(h)[0]
            raw_locations_loc = raw_locations_param[:num_blocks]
            raw_locations_scale = raw_locations_param[
                self.num_primitives : self.num_primitives + num_blocks
            ].exp()

            # --Sample raw_locations
            raw_locations = pyro.sample(
                f"raw_locations_{batch_id}",
                pyro.distributions.Independent(
                    pyro.distributions.Normal(raw_locations_loc, raw_locations_scale),
                    reinterpreted_batch_ndims=1,
                ),
            )

            traces.append({"stacking_program": stacking_program, "raw_locations": raw_locations})

        return traces


if __name__ == "__main__":
    # Init
    device = "cuda"
    batch_size = 3
    num_primitives = 2
    im_size = 64

    # Create obs
    obs = generate_from_true_generative_model(
        batch_size, num_primitives, device, num_rows=im_size, num_cols=im_size
    )

    # Run through learnable generative model
    generative_model = GenerativeModel().to(device)

    print(generative_model.forward(obs))

    # Run through guide
    guide = Guide().to(device)
    print(guide.forward(obs))
