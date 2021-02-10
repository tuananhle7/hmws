"""
In this model, we want to learn the symbols (square or heart) of fixed scale as well as the prior
over programs which combine those symbols (one symbol, or two symbols combined with + or -).

The symbols in the dataset have a fixed scale so that things are more easily learnable.
We don't need to make the problem artificially hard since this is only a toy example to demonstrate
the approach.
"""


import pyro
import util
import render
import torch
import torch.nn as nn


class GenerativeModel(nn.Module):
    """The prior over program follows the PCFG (can change to some other prior later)

    Program = Shape | Shape + Shape | Shape - Shape
    Shape = Shape_1 | ... | Shape_N
    """

    def __init__(self, im_size=64, num_primitives=2):
        super().__init__()
        self.im_size = im_size
        self.num_primitives = num_primitives

        # Primitive parameters (parameters of symbols)
        self.mlps = nn.ModuleList(
            [
                util.init_mlp(1, 1, 64, 3, non_linearity=nn.ReLU())
                for _ in range(self.num_primitives)
            ]
        )
        self.logit_multipliers_raw = nn.Parameter(
            torch.ones((self.num_primitives,))
        )  # helps with learning dynamics

        # PCFG parameters (parameters of the symbolic structure)
        # Program = Shape | Shape + Shape | Shape - Shape
        self.program_id_logits = nn.Parameter(torch.ones((3,)))
        # Shape = Shape_1 | ... | Shape_N
        self.shape_id_logits = nn.Parameter(torch.ones((self.num_primitives,)))

    @property
    def device(self):
        return next(self.mlps[0].parameters()).device

    def sample_program(self, batch_id):
        """Samples a program from the PCFG"""
        program_id = pyro.sample(
            f"program_id_{batch_id}", pyro.distributions.Categorical(logits=self.program_id_logits),
        ).long()

        if program_id == 0:
            shape_ids = [
                pyro.sample(
                    f"shape_id_{batch_id}",
                    pyro.distributions.Categorical(logits=self.shape_id_logits),
                ).long(),
            ]
        else:
            shape_ids = [
                pyro.sample(
                    f"shape_id_{i}_{batch_id}",
                    pyro.distributions.Categorical(logits=self.shape_id_logits),
                ).long()
                for i in range(2)
            ]
        return program_id, shape_ids

    def get_shape_obs_logits(self, shape_id, raw_position, num_rows=None, num_cols=None):
        """p(obs | shape_id, raw_position)

        Args
            shape_id (int)
            raw_position [*shape, 2]
            num_rows (int)
            num_cols (int)

        Returns: [*shape, num_rows, num_cols]
        """
        if num_rows is None:
            num_rows = self.im_size
        if num_cols is None:
            num_cols = self.im_size

        # Extract
        position = raw_position.sigmoid() - 0.5
        shape = raw_position.shape[:-1]

        # Get canvas
        # [*shape]
        position_x, position_y = position[..., 0], position[..., 1]
        # [num_rows, num_cols]
        canvas_x, canvas_y = render.get_canvas_xy(num_rows, num_cols, self.device)

        # Shift and scale
        # [num_samples, num_rows, num_cols]
        canvas_x = canvas_x[None] - position_x.view(-1, 1, 1)
        canvas_y = canvas_y[None] - position_y.view(-1, 1, 1)

        # Build input
        # [num_samples * num_rows * num_cols, 1]
        mlp_input = torch.atan2(canvas_y, canvas_x).view(-1, 1)

        # Run MLP
        # [*shape, num_rows, num_cols]
        logits = self.logit_multipliers_raw[shape_id].exp() * (
            self.mlps[shape_id](mlp_input).view(*[*shape, num_rows, num_cols]).exp()
            - torch.sqrt(canvas_x ** 2 + canvas_y ** 2).view(*[*shape, num_rows, num_cols])
        )

        if torch.isnan(logits).any():
            raise RuntimeError("nan")

        return logits

    def get_obs_probs(self, program, raw_positions, num_rows=None, num_cols=None):
        """p_S(obs | program, raw_positions)

        Args
            program
                program_id (long)
                shape_ids (long,) or (long, long)
            raw_positions [2] or ([2], [2])
            num_rows (int)
            num_cols (int)

        Returns: [num_rows, num_cols] (probs)
        """
        if num_rows is None:
            num_rows = self.im_size
        if num_cols is None:
            num_cols = self.im_size

        # Extract
        program_id, shape_ids = program
        if program_id == 0:
            # Extract
            shape_id = shape_ids[0]
            raw_position = raw_positions[0]

            return self.get_shape_obs_logits(shape_id, raw_position, num_rows, num_cols).sigmoid()
        else:
            obs_probss = []
            for shape_id, raw_position in zip(shape_ids, raw_positions):
                obs_probss.append(
                    self.get_shape_obs_logits(shape_id, raw_position, num_rows, num_cols).sigmoid()
                )

            if program_id == 1:
                obs_probs = obs_probss[0] + obs_probss[1]
            elif program_id == 2:
                obs_probs = obs_probss[0] - obs_probss[1]
            else:
                raise RuntimeError("Invalid program")
            return obs_probs.clamp(0, 1)

    def forward(self, obs, observations=None):
        """
        Args:
            obs [batch_size, num_rows, num_cols]
        """
        pyro.module("generative_model", self)
        if isinstance(observations, dict):
            batch_size = len(observations)
            obs = torch.stack([observations[f"obs_{i}"] for i in range(batch_size)])
        batch_size, num_rows, num_cols = obs.shape
        for batch_id in pyro.plate("batch", batch_size):
            # p(program)
            program = self.sample_program(batch_id)
            program_id, shape_ids = program

            # p(raw_position)
            if program_id == 0:
                raw_positions = [
                    pyro.sample(
                        f"raw_position_{batch_id}",
                        pyro.distributions.Independent(
                            pyro.distributions.Normal(
                                torch.zeros((2,), device=self.device),
                                torch.ones((2,), device=self.device),
                            ),
                            reinterpreted_batch_ndims=1,
                        ),
                    ),
                ]
            else:
                raw_positions = [
                    pyro.sample(
                        f"raw_position_{i}_{batch_id}",
                        pyro.distributions.Independent(
                            pyro.distributions.Normal(
                                torch.zeros((2,), device=self.device),
                                torch.ones((2,), device=self.device),
                            ),
                            reinterpreted_batch_ndims=1,
                        ),
                    )
                    for i in range(2)
                ]

            # p(obs | shape_id, raw_position)
            pyro.sample(
                f"obs_{batch_id}",
                pyro.distributions.Independent(
                    pyro.distributions.Bernoulli(
                        probs=self.get_obs_probs(program, raw_positions, num_rows, num_cols)
                    ),
                    reinterpreted_batch_ndims=2,
                ),
                obs=obs[batch_id],
            )


class Guide(nn.Module):
    """
    This uses an inference compilation architecture

    LSTM input: [obs_embedding, sample_embedding, address_embedding]
    LSTM output --> distribution params
    """

    def __init__(self, im_size=64, num_primitives=2):
        super().__init__()
        self.im_size = im_size
        self.num_primitives = num_primitives

        # Obs embedder
        self.obs_embedder = util.init_cnn(output_dim=16)
        self.obs_embedding_dim = 400  # computed manually

        # LSTM
        self.sample_embedding_dim = 16
        self.address_embedding_dim = 16
        self.lstm_input_dim = (
            self.obs_embedding_dim + self.sample_embedding_dim + self.address_embedding_dim
        )
        self.lstm_hidden_dim = 256
        self.lstm_cell = torch.nn.LSTMCell(self.lstm_input_dim, self.lstm_hidden_dim)

        # Address embeddings
        self.address_to_id = {
            "program_id": 0,
            "shape_id": 1,
            "shape_id_0": 2,
            "shape_id_1": 3,
            "raw_position": 4,
            "raw_position_0": 5,
            "raw_position_1": 6,
        }
        self.num_addresses = 7
        self.address_embeddings = nn.Parameter(
            torch.randn((self.num_addresses, self.address_embedding_dim))
        )

        # Sample embedders
        # --Program id
        self.program_id_embeddings = nn.Parameter(torch.randn((3, self.sample_embedding_dim)))

        # --Shape id
        self.shape_id_embeddings = nn.Parameter(
            torch.randn((self.num_primitives, self.sample_embedding_dim))
        )

        # --Raw position embedder
        self.raw_position_embedder = util.init_mlp(
            2, self.sample_embedding_dim, hidden_dim=100, num_layers=3
        )

        # Param extractors
        # --Program id
        self.program_id_param_extractor = nn.Linear(self.lstm_hidden_dim, 3)

        # --Shape id
        self.shape_id_param_extractor = nn.Linear(self.lstm_hidden_dim, self.num_primitives)

        # --Raw position extractor
        self.raw_position_param_extractor = nn.Linear(self.lstm_hidden_dim, 4)

    def get_address_embedding(self, address):
        return self.address_embeddings[self.address_to_id[address]]

    @property
    def device(self):
        return self.program_id_embeddings.device

    def get_obs_embedding(self, obs):
        """
        Args:
            obs: [batch_size, im_size, im_size]

        Returns: [batch_size, obs_embedding_dim]
        """
        batch_size = obs.shape[0]
        return self.obs_embedder(obs[:, None]).view(batch_size, -1)

    def forward(self, obs, observations=None):
        """
        Args:
            obs [batch_size, num_rows, num_cols]

        Returns:
            shape_id [batch_size]
            raw_position [batch_size, 2]
        """
        pyro.module("guide", self)
        if isinstance(observations, dict):
            batch_size = len(observations)
            obs = torch.stack([observations[f"obs_{i}"] for i in range(batch_size)])
        batch_size, num_rows, num_cols = obs.shape

        # Get obs embedding
        obs_embedding = self.get_obs_embedding(obs)

        traces = []
        for batch_id in pyro.plate("batch", batch_size):
            # trace = []

            # Extract obs embedding
            obs_embedding_b = obs_embedding[batch_id]

            # Sample program id
            # --LSTM Input
            prev_sample_embedding = torch.zeros((self.sample_embedding_dim,), device=self.device)
            address_embedding = self.get_address_embedding("program_id")
            lstm_input = torch.cat([obs_embedding_b, prev_sample_embedding, address_embedding])

            # --Run LSTM
            h, c = self.lstm_cell(lstm_input[None])
            hc = h, c

            # --Extract params
            program_id_logits = self.program_id_param_extractor(h)[0]

            # --Sample program_id
            program_id = pyro.sample(
                f"program_id_{batch_id}", pyro.distributions.Categorical(logits=program_id_logits),
            ).long()
            # trace.append(program_id)

            if program_id == 0:
                # Sample shape id
                # --LSTM Input
                prev_sample_embedding = self.program_id_embeddings[program_id]
                address_embedding = self.get_address_embedding("shape_id")
                lstm_input = torch.cat([obs_embedding_b, prev_sample_embedding, address_embedding])

                # --Run LSTM
                h, c = self.lstm_cell(lstm_input[None], hc)
                hc = h, c

                # --Extract params
                shape_id_logits = self.shape_id_param_extractor(h)[0]

                # --Sample shape id
                shape_id = pyro.sample(
                    f"shape_id_{batch_id}", pyro.distributions.Categorical(logits=shape_id_logits),
                ).long()

                # # --Update trace
                # trace.append(("shape_id", shape_id))

                # Sample raw position
                # --LSTM Input
                prev_sample_embedding = self.shape_id_embeddings[shape_id]
                address_embedding = self.get_address_embedding("raw_position")
                lstm_input = torch.cat([obs_embedding_b, prev_sample_embedding, address_embedding])

                # --Run LSTM
                h, c = self.lstm_cell(lstm_input[None], hc)
                hc = h, c

                # --Extract params
                raw_position_param = self.raw_position_param_extractor(h)[0]
                raw_position_loc = raw_position_param[:2]
                raw_position_scale = raw_position_param[2:].exp()

                # --Sample raw_position
                raw_position = pyro.sample(
                    f"raw_position_{batch_id}",
                    pyro.distributions.Independent(
                        pyro.distributions.Normal(raw_position_loc, raw_position_scale),
                        reinterpreted_batch_ndims=1,
                    ),
                )

                # # --Update trace
                # trace.append(("raw_position", raw_position))

                traces.append(((program_id, (shape_id,)), (raw_position,)))
            else:
                # Sample shape id 0
                # --LSTM Input
                prev_sample_embedding = self.program_id_embeddings[program_id]
                address_embedding = self.get_address_embedding("shape_id_0")
                lstm_input = torch.cat([obs_embedding_b, prev_sample_embedding, address_embedding])

                # --Run LSTM
                h, c = self.lstm_cell(lstm_input[None], hc)
                hc = h, c

                # --Extract params
                shape_id_0_logits = self.shape_id_param_extractor(h)[0]

                # --Sample shape id
                shape_id_0 = pyro.sample(
                    f"shape_id_0_{batch_id}",
                    pyro.distributions.Categorical(logits=shape_id_0_logits),
                ).long()

                # # --Update trace
                # trace.append(("shape_id_0", shape_id_0))

                # Sample shape id 1
                # --LSTM Input
                prev_sample_embedding = self.shape_id_embeddings[shape_id_0]
                address_embedding = self.get_address_embedding("shape_id_1")
                lstm_input = torch.cat([obs_embedding_b, prev_sample_embedding, address_embedding])

                # --Run LSTM
                h, c = self.lstm_cell(lstm_input[None], hc)
                hc = h, c

                # --Extract params
                shape_id_1_logits = self.shape_id_param_extractor(h)[0]

                # --Sample shape id
                shape_id_1 = pyro.sample(
                    f"shape_id_1_{batch_id}",
                    pyro.distributions.Categorical(logits=shape_id_1_logits),
                ).long()

                # # --Update trace
                # trace.append(("shape_id_1", shape_id_1))

                # Sample raw position 0
                # --LSTM Input
                prev_sample_embedding = self.shape_id_embeddings[shape_id_1]
                address_embedding = self.get_address_embedding("raw_position_0")
                lstm_input = torch.cat([obs_embedding_b, prev_sample_embedding, address_embedding])

                # --Run LSTM
                h, c = self.lstm_cell(lstm_input[None], hc)
                hc = h, c

                # --Extract params
                raw_position_param = self.raw_position_param_extractor(h)[0]
                raw_position_loc = raw_position_param[:2]
                raw_position_scale = raw_position_param[2:].exp()

                # --Sample raw_position
                raw_position_0 = pyro.sample(
                    f"raw_position_0_{batch_id}",
                    pyro.distributions.Independent(
                        pyro.distributions.Normal(raw_position_loc, raw_position_scale),
                        reinterpreted_batch_ndims=1,
                    ),
                )

                # # --Update trace
                # trace.append(("raw_position_0", raw_position_0))

                # Sample raw position 1
                # --LSTM Input
                prev_sample_embedding = self.raw_position_embedder(raw_position_0[None])[0]
                address_embedding = self.get_address_embedding("raw_position_1")
                lstm_input = torch.cat([obs_embedding_b, prev_sample_embedding, address_embedding])

                # --Run LSTM
                h, c = self.lstm_cell(lstm_input[None], hc)
                hc = h, c

                # --Extract params
                raw_position_param = self.raw_position_param_extractor(h)[0]
                raw_position_loc = raw_position_param[:2]
                raw_position_scale = raw_position_param[2:].exp()

                # --Sample raw_position
                raw_position_1 = pyro.sample(
                    f"raw_position_1_{batch_id}",
                    pyro.distributions.Independent(
                        pyro.distributions.Normal(raw_position_loc, raw_position_scale),
                        reinterpreted_batch_ndims=1,
                    ),
                )

                # # --Update trace
                # trace.append(("raw_position_1", raw_position_1))

                traces.append(
                    ((program_id, (shape_id_0, shape_id_1)), (raw_position_0, raw_position_1))
                )

            # traces.append(trace)

        return traces


def trace_to_str(trace):
    (program_id, shape_ids), raw_positions = trace
    if program_id == 0:
        shape_id = shape_ids[0]
        raw_position = raw_positions[0]
        position = raw_position.sigmoid() - 0.5
        return f"$S_{shape_id}$({position[0]:.2f}, {position[1]:.2f})"
    else:
        operation = "+" if program_id == 1 else "-"
        shape_id_0, shape_id_1 = shape_ids
        raw_position_0, raw_position_1 = raw_positions
        position_0 = raw_position_0.sigmoid() - 0.5
        position_1 = raw_position_1.sigmoid() - 0.5
        return (
            f"$S_{shape_id_0}$({position_0[0]:.2f}, {position_0[1]:.2f})"
            f"{operation}\n"
            f"$S_{shape_id_1}$({position_1[0]:.2f}, {position_1[1]:.2f})"
        )
