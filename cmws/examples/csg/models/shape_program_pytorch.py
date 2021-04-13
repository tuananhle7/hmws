import torch
import torch.nn as nn
from cmws import util
from cmws.examples.csg import render


class GenerativeModel(nn.Module):
    """The prior over program follows the PCFG (can change to some other prior later)

    Program = Shape | Shape + Shape | Shape - Shape
    Shape = Shape_1 | ... | Shape_N
    """

    def __init__(self, im_size=64, num_primitives=2):
        super().__init__()
        self.im_size = im_size
        self.num_primitives = num_primitives

        self.max_num_shapes = 2

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
        self.shape_id_logits = nn.Parameter(torch.ones((self.max_num_shapes, self.num_primitives)))

    @property
    def device(self):
        return next(self.mlps[0].parameters()).device

    @property
    def program_id_dist(self):
        """Prior distribution p(program_id)
        batch_shape [], event_shape []
        """
        return torch.distributions.Categorical(logits=self.program_id_logits)

    def get_num_shapes(self, program_id):
        """Determines number of shapes based on program id

        Args:
            program_id [*shape]

        Returns: [*shape]
        """
        one = torch.ones_like(program_id)
        two = torch.ones_like(program_id) * 2
        return one * (program_id == 0).float() + two * (program_id != 0).float()

    def latent_sample(self, sample_shape=[]):
        """Sample from p(z)

        Args
            sample_shape

        Returns
            latent:
                program_id [*sample_shape]
                shape_ids [*sample_shape, max_num_shapes=2]
                raw_positions [*sample_shape, max_num_shapes=2, 2]
        """
        # Sample program_id
        program_id = self.program_id_dist.sample(sample_shape)

        # Sample stacking_program
        num_shapes = self.get_num_shapes(program_id)
        stacking_program = util.pad_tensor(
            torch.distributions.Categorical(logits=self.shape_id_logits).sample(sample_shape),
            num_shapes,
            0,
        )

        # Sample raw_locations
        # --Compute dist params
        loc = torch.zeros((self.max_num_shapes, 2), device=self.device)
        scale = torch.ones((self.max_num_shapes, 2), device=self.device)
        # --Compute log prob [*shape]
        raw_locations = util.pad_tensor(
            torch.distributions.Independent(
                torch.distributions.Normal(loc, scale), reinterpreted_batch_ndims=1,
            ).sample(sample_shape),
            num_shapes,
            0,
        )

        return program_id, stacking_program, raw_locations

    def latent_log_prob(self, latent):
        """Prior log p(z)

        Args:
            latent:
                program_id [*shape]
                shape_ids [*shape, max_num_shapes=2]
                raw_positions [*shape, max_num_shapes=2, 2]

        Returns: [*shape]
        """
        # Extract
        program_id, shape_ids, raw_positions = latent

        # log p(program_id)
        program_id_log_prob = self.program_id_dist.log_prob(program_id)

        # log p(shape_ids)
        # --Compute num shapes
        num_shapes = self.get_num_shapes(program_id)
        shape_ids_log_prob = util.pad_tensor(
            torch.distributions.Categorical(logits=self.shape_id_logits).log_prob(shape_ids),
            num_shapes,
            0,
        ).sum(-1)

        # log p(raw_positions)
        # --Compute dist params
        loc = torch.zeros((self.max_num_shapes, 2), device=self.device)
        scale = torch.ones((self.max_num_shapes, 2), device=self.device)
        # --Compute log prob [*shape]
        raw_positions_log_prob = util.pad_tensor(
            torch.distributions.Independent(
                torch.distributions.Normal(loc, scale), reinterpreted_batch_ndims=1,
            ).log_prob(raw_positions),
            num_shapes,
            0,
        ).sum(-1)

        return program_id_log_prob + shape_ids_log_prob + raw_positions_log_prob

    def get_shape_obs_logits_single(self, shape_id, raw_position):
        """p(obs | shape_id, raw_position)

        Args
            shape_id (int)
            raw_position [*shape, 2]

        Returns: [*shape, im_size, im_size]
        """
        # Extract
        position = raw_position.sigmoid() - 0.5
        shape = raw_position.shape[:-1]

        # Get canvas
        # [*shape]
        position_x, position_y = position[..., 0], position[..., 1]
        # [im_size, im_size]
        canvas_x, canvas_y = render.get_canvas_xy(self.im_size, self.im_size, self.device)

        # Shift and scale
        # [num_samples, im_size, im_size]
        canvas_x = canvas_x[None] - position_x.view(-1, 1, 1)
        canvas_y = canvas_y[None] - position_y.view(-1, 1, 1)

        # Build input
        # [num_samples * num_rows * num_cols, 1]
        mlp_input = torch.atan2(canvas_y, canvas_x).view(-1, 1)

        # Run MLP
        # [*shape, im_size, im_size]
        logits = self.logit_multipliers_raw[shape_id].exp() * (
            self.mlps[shape_id](mlp_input).view(*[*shape, self.im_size, self.im_size]).exp()
            - torch.sqrt(canvas_x ** 2 + canvas_y ** 2).view(*[*shape, self.im_size, self.im_size])
        )

        if torch.isnan(logits).any():
            raise RuntimeError("nan")

        return logits

    def get_obs_probs_single(self, program, raw_positions):
        """p_S(obs | program, raw_positions)

        Args
            program
                program_id (long)
                shape_ids (long,) or (long, long)
            raw_positions [2] or ([2], [2])

        Returns: [im_size, im_size] (probs)
        """
        # Extract
        program_id, shape_ids = program
        if program_id == 0:
            # Extract
            shape_id = shape_ids[0]
            raw_position = raw_positions[0]

            return self.get_shape_obs_logits_single(shape_id, raw_position).sigmoid()
        else:
            obs_probss = []
            for shape_id, raw_position in zip(shape_ids, raw_positions):
                obs_probss.append(
                    self.get_shape_obs_logits_single(shape_id, raw_position).sigmoid()
                )

            if program_id == 1:
                obs_probs = obs_probss[0] + obs_probss[1]
            elif program_id == 2:
                obs_probs = obs_probss[0] - obs_probss[1]
            else:
                raise RuntimeError("Invalid program")
            return obs_probs.clamp(0, 1)

    def get_obs_probs(self, latent):
        """p_S(obs | program, raw_positions)

        Args
            latent
                program_id [*shape]
                shape_ids [*shape, max_num_shapes=2]
                raw_positions [*shape, max_num_shapes=2, 2]

        Returns: [*shape, im_size, im_size] (probs)
        """
        # Extract
        program_id, shape_ids, raw_positions = latent
        shape = program_id.shape
        num_elements = util.get_num_elements(shape)

        # Flatten
        program_id_flattened = program_id.reshape(-1)
        shape_ids_flattened = shape_ids.reshape(-1, self.max_num_shapes)
        raw_positions_flattened = raw_positions.reshape(-1, self.max_num_shapes, 2)

        # Compute for each element in the batch
        result = []
        for element_id in range(num_elements):
            result.append(
                self.get_obs_probs_single(
                    (program_id_flattened[element_id], shape_ids_flattened[element_id]),
                    raw_positions_flattened[element_id],
                )
            )
        return torch.stack(result).view(*[*shape, self.im_size, self.im_size])

    def log_prob(self, latent, obs):
        """Log joint probability of the generative model
        log p(z, x)

        Args:
            latent:
                program_id [*sample_shape, *shape]
                shape_ids [*sample_shape, *shape, max_num_shapes=2]
                raw_positions [*sample_shape, *shape, max_num_shapes=2, 2]
            obs [*shape, im_size, im_size]

        Returns: [*sample_shape, *shape]
        """
        # p(z)
        latent_log_prob = self.latent_log_prob(latent)

        # p(x | z)
        obs_log_prob = torch.distributions.Independent(
            torch.distributions.Bernoulli(probs=self.get_obs_probs(latent)),
            reinterpreted_batch_ndims=2,
        ).log_prob(obs)

        return latent_log_prob + obs_log_prob

    def log_prob_discrete_continuous(self, discrete_latent, continuous_latent, obs):
        """Log joint probability of the generative model
        log p(z, x)

        Args:
            discrete_latent
                program_id [*discrete_shape, *shape]
                shape_ids [*discrete_shape, *shape, max_num_shapes=2]
            continuous_latent
                raw_positions [*continuous_shape, *discrete_shape, *shape, max_num_shapes=2, 2]
            obs [*shape, im_size, im_size]

        Returns: [*continuous_shape, *discrete_shape, *shape]
        """
        # Extract
        program_id, shape_ids = discrete_latent
        raw_positions = continuous_latent
        shape = obs.shape[:-2]
        discrete_shape = program_id.shape[: -len(shape)]
        continuous_shape = continuous_latent.shape[: -(len(shape) + len(discrete_shape) + 2)]
        continuous_num_elements = util.get_num_elements(continuous_shape)

        # Expand discrete latent
        program_id_expanded = (
            program_id[None]
            .expand(*[continuous_num_elements, *discrete_shape, *shape])
            .view(*[*continuous_shape, *discrete_shape, *shape])
        )
        shape_ids_expanded = (
            shape_ids[None]
            .expand(*[continuous_num_elements, *discrete_shape, *shape, self.max_num_shapes])
            .view(*[*continuous_shape, *discrete_shape, *shape, self.max_num_shapes])
        )

        return self.log_prob((program_id_expanded, shape_ids_expanded, raw_positions), obs)

    @torch.no_grad()
    def sample(self, sample_shape=[]):
        """Sample from p(z, x)

        Args
            sample_shape

        Returns
            latent:
                program_id [*sample_shape]
                shape_ids [*sample_shape, max_num_shapes=2]
                raw_positions [*sample_shape, max_num_shapes=2, 2]
            obs [*sample_shape, im_size, im_size]
        """
        # p(z)
        latent = self.latent_sample(sample_shape)

        # p(x | z)
        obs = self.get_obs_probs(latent)

        return latent, obs


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
        self.max_num_shapes = 2

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

    def log_prob(self, obs, latent):
        """
        Args
            obs [*shape, im_size, im_size]
            latent:
                program_id [*sample_shape, *shape]
                shape_ids [*sample_shape, *shape, max_num_shapes=2]
                raw_positions [*sample_shape, *shape, max_num_shapes=2, 2]

        Returns [*sample_shape, *shape]
        """
        # Extract
        shape = obs.shape[:-2]
        program_id_batched, shape_ids_batched, raw_positions_batched = latent
        sample_shape = program_id_batched.shape[: -len(shape)]
        num_elements = util.get_num_elements(shape)
        num_samples = util.get_num_elements(sample_shape)

        # Get obs embedding (flattened)
        # [num_elements, obs_embedding_dim]
        obs_embedding = self.get_obs_embedding(obs.reshape(-1, self.im_size, self.im_size))

        # Flatten latents
        program_id_flattened = program_id_batched.view(num_samples, num_elements)
        shape_ids_flattened = shape_ids_batched.view(num_samples, num_elements, self.max_num_shapes)
        raw_positions_flattened = raw_positions_batched.view(
            num_samples, num_elements, self.max_num_shapes, 2
        )

        # Init result
        result = torch.zeros((num_samples, num_elements), device=self.device)

        # traces = []
        for sample_id in range(num_samples):
            for element_id in range(num_elements):
                # trace = []
                # Extract latent
                program_id = program_id_flattened[sample_id, element_id]
                shape_ids = shape_ids_flattened[sample_id, element_id]
                raw_positions = raw_positions_flattened[sample_id, element_id]

                # Extract obs embedding
                obs_embedding_b = obs_embedding[element_id]

                # Log_prob(program id)
                # --LSTM Input
                prev_sample_embedding = torch.zeros(
                    (self.sample_embedding_dim,), device=self.device
                )
                address_embedding = self.get_address_embedding("program_id")
                lstm_input = torch.cat([obs_embedding_b, prev_sample_embedding, address_embedding])

                # --Run LSTM
                h, c = self.lstm_cell(lstm_input[None])
                hc = h, c

                # --Extract params
                program_id_logits = self.program_id_param_extractor(h)[0]

                # --Log_prob(program_id)
                program_id_lp = torch.distributions.Categorical(logits=program_id_logits).log_prob(
                    program_id
                )
                # trace.append(program_id)

                if program_id == 0:
                    # Log_prob(shape id)
                    # --LSTM Input
                    prev_sample_embedding = self.program_id_embeddings[program_id]
                    address_embedding = self.get_address_embedding("shape_id")
                    lstm_input = torch.cat(
                        [obs_embedding_b, prev_sample_embedding, address_embedding]
                    )

                    # --Run LSTM
                    h, c = self.lstm_cell(lstm_input[None], hc)
                    hc = h, c

                    # --Extract params
                    shape_id_logits = self.shape_id_param_extractor(h)[0]

                    # --Log_prob(shape id)
                    shape_id_lp = torch.distributions.Categorical(logits=shape_id_logits).log_prob(
                        shape_ids[0]
                    )

                    # # --Update trace
                    # trace.append(("shape_id", shape_id))

                    # Log_prob(raw position)
                    # --LSTM Input
                    prev_sample_embedding = self.shape_id_embeddings[shape_ids[0]]
                    address_embedding = self.get_address_embedding("raw_position")
                    lstm_input = torch.cat(
                        [obs_embedding_b, prev_sample_embedding, address_embedding]
                    )

                    # --Run LSTM
                    h, c = self.lstm_cell(lstm_input[None], hc)
                    hc = h, c

                    # --Extract params
                    raw_position_param = self.raw_position_param_extractor(h)[0]
                    raw_position_loc = raw_position_param[:2]
                    raw_position_scale = raw_position_param[2:].exp()

                    # --Log_prob(raw_position)
                    raw_position_lp = torch.distributions.Independent(
                        torch.distributions.Normal(raw_position_loc, raw_position_scale),
                        reinterpreted_batch_ndims=1,
                    ).log_prob(raw_positions[0])

                    # # --Update trace
                    # trace.append(("raw_position", raw_position))
                    result[sample_id, element_id] = program_id_lp + shape_id_lp + raw_position_lp
                else:
                    # Log_prob(shape id 0)
                    # --LSTM Input
                    prev_sample_embedding = self.program_id_embeddings[program_id]
                    address_embedding = self.get_address_embedding("shape_id_0")
                    lstm_input = torch.cat(
                        [obs_embedding_b, prev_sample_embedding, address_embedding]
                    )

                    # --Run LSTM
                    h, c = self.lstm_cell(lstm_input[None], hc)
                    hc = h, c

                    # --Extract params
                    shape_id_0_logits = self.shape_id_param_extractor(h)[0]

                    # --Log_prob(shape id)
                    shape_id_0_lp = torch.distributions.Categorical(
                        logits=shape_id_0_logits
                    ).log_prob(shape_ids[0])

                    # # --Update trace
                    # trace.append(("shape_id_0", shape_id_0))

                    # Log_prob(shape id 1)
                    # --LSTM Input
                    prev_sample_embedding = self.shape_id_embeddings[shape_ids[0]]
                    address_embedding = self.get_address_embedding("shape_id_1")
                    lstm_input = torch.cat(
                        [obs_embedding_b, prev_sample_embedding, address_embedding]
                    )

                    # --Run LSTM
                    h, c = self.lstm_cell(lstm_input[None], hc)
                    hc = h, c

                    # --Extract params
                    shape_id_1_logits = self.shape_id_param_extractor(h)[0]

                    # --Log_prob(shape id)
                    shape_id_1_lp = torch.distributions.Categorical(
                        logits=shape_id_1_logits
                    ).log_prob(shape_ids[1])

                    # # --Update trace
                    # trace.append(("shape_id_1", shape_id_1))

                    # Log_prob(raw position 0)
                    # --LSTM Input
                    prev_sample_embedding = self.shape_id_embeddings[shape_ids[1]]
                    address_embedding = self.get_address_embedding("raw_position_0")
                    lstm_input = torch.cat(
                        [obs_embedding_b, prev_sample_embedding, address_embedding]
                    )

                    # --Run LSTM
                    h, c = self.lstm_cell(lstm_input[None], hc)
                    hc = h, c

                    # --Extract params
                    raw_position_param = self.raw_position_param_extractor(h)[0]
                    raw_position_loc = raw_position_param[:2]
                    raw_position_scale = raw_position_param[2:].exp()

                    # --Log_prob(raw_position)
                    raw_position_0_lp = torch.distributions.Independent(
                        torch.distributions.Normal(raw_position_loc, raw_position_scale),
                        reinterpreted_batch_ndims=1,
                    ).log_prob(raw_positions[0])

                    # # --Update trace
                    # trace.append(("raw_position_0", raw_position_0))

                    # Log_prob(raw position 1)
                    # --LSTM Input
                    prev_sample_embedding = self.raw_position_embedder(raw_positions[0][None])[0]
                    address_embedding = self.get_address_embedding("raw_position_1")
                    lstm_input = torch.cat(
                        [obs_embedding_b, prev_sample_embedding, address_embedding]
                    )

                    # --Run LSTM
                    h, c = self.lstm_cell(lstm_input[None], hc)
                    hc = h, c

                    # --Extract params
                    raw_position_param = self.raw_position_param_extractor(h)[0]
                    raw_position_loc = raw_position_param[:2]
                    raw_position_scale = raw_position_param[2:].exp()

                    # --Log_prob(raw_position)
                    raw_position_1_lp = torch.distributions.Independent(
                        torch.distributions.Normal(raw_position_loc, raw_position_scale),
                        reinterpreted_batch_ndims=1,
                    ).log_prob(raw_positions[1])

                    # # --Update trace
                    # trace.append(("raw_position_1", raw_position_1))
                    result[sample_id, element_id] = (
                        program_id_lp
                        + shape_id_0_lp
                        + shape_id_1_lp
                        + raw_position_0_lp
                        + raw_position_1_lp
                    )
                # traces.append(trace)

        return result.view(*[*sample_shape, *shape])

    def sample(self, obs, sample_shape=[]):
        """z ~ q(z | x)

        Args
            obs [*shape, im_size, im_size]

        Returns
            program_id [*sample_shape, *shape]
            shape_ids [*sample_shape, *shape, max_num_shapes=2]
            raw_positions [*sample_shape, *shape, max_num_shapes=2, 2]
        """
        # Extract
        shape = obs.shape[:-2]
        num_elements = util.get_num_elements(shape)
        num_samples = util.get_num_elements(sample_shape)

        # Get obs embedding (flattened)
        # [num_elements, obs_embedding_dim]
        obs_embedding = self.get_obs_embedding(obs.view(-1, self.im_size, self.im_size))

        # Init result
        program_id_flattened = torch.zeros((num_samples, num_elements), device=self.device).long()
        shape_ids_flattened = torch.zeros(
            (num_samples, num_elements, self.max_num_shapes), device=self.device
        ).long()
        raw_positions_flattened = torch.zeros(
            (num_samples, num_elements, self.max_num_shapes, 2), device=self.device
        )

        # traces = []
        for sample_id in range(num_samples):
            for element_id in range(num_elements):
                # trace = []

                # Extract obs embedding
                obs_embedding_b = obs_embedding[element_id]

                # Sample program id
                # --LSTM Input
                prev_sample_embedding = torch.zeros(
                    (self.sample_embedding_dim,), device=self.device
                )
                address_embedding = self.get_address_embedding("program_id")
                lstm_input = torch.cat([obs_embedding_b, prev_sample_embedding, address_embedding])

                # --Run LSTM
                h, c = self.lstm_cell(lstm_input[None])
                hc = h, c

                # --Extract params
                program_id_logits = self.program_id_param_extractor(h)[0]

                # --Sample program_id
                program_id = torch.distributions.Categorical(logits=program_id_logits).sample()
                # trace.append(program_id)

                if program_id == 0:
                    # Sample shape id
                    # --LSTM Input
                    prev_sample_embedding = self.program_id_embeddings[program_id]
                    address_embedding = self.get_address_embedding("shape_id")
                    lstm_input = torch.cat(
                        [obs_embedding_b, prev_sample_embedding, address_embedding]
                    )

                    # --Run LSTM
                    h, c = self.lstm_cell(lstm_input[None], hc)
                    hc = h, c

                    # --Extract params
                    shape_id_logits = self.shape_id_param_extractor(h)[0]

                    # --Sample shape id
                    shape_id = torch.distributions.Categorical(logits=shape_id_logits).sample()

                    # # --Update trace
                    # trace.append(("shape_id", shape_id))

                    # Sample raw position
                    # --LSTM Input
                    prev_sample_embedding = self.shape_id_embeddings[shape_id]
                    address_embedding = self.get_address_embedding("raw_position")
                    lstm_input = torch.cat(
                        [obs_embedding_b, prev_sample_embedding, address_embedding]
                    )

                    # --Run LSTM
                    h, c = self.lstm_cell(lstm_input[None], hc)
                    hc = h, c

                    # --Extract params
                    raw_position_param = self.raw_position_param_extractor(h)[0]
                    raw_position_loc = raw_position_param[:2]
                    raw_position_scale = raw_position_param[2:].exp()

                    # --Sample raw_position
                    raw_position = torch.distributions.Independent(
                        torch.distributions.Normal(raw_position_loc, raw_position_scale),
                        reinterpreted_batch_ndims=1,
                    ).sample()

                    # # --Update trace
                    # trace.append(("raw_position", raw_position))
                    program_id_flattened[sample_id, element_id] = program_id
                    shape_ids_flattened[sample_id, element_id, 0] = shape_id
                    raw_positions_flattened[sample_id, element_id, 0] = raw_position
                else:
                    # Sample shape id 0
                    # --LSTM Input
                    prev_sample_embedding = self.program_id_embeddings[program_id]
                    address_embedding = self.get_address_embedding("shape_id_0")
                    lstm_input = torch.cat(
                        [obs_embedding_b, prev_sample_embedding, address_embedding]
                    )

                    # --Run LSTM
                    h, c = self.lstm_cell(lstm_input[None], hc)
                    hc = h, c

                    # --Extract params
                    shape_id_0_logits = self.shape_id_param_extractor(h)[0]

                    # --Sample shape id
                    shape_id_0 = torch.distributions.Categorical(logits=shape_id_0_logits).sample()

                    # # --Update trace
                    # trace.append(("shape_id_0", shape_id_0))

                    # Sample shape id 1
                    # --LSTM Input
                    prev_sample_embedding = self.shape_id_embeddings[shape_id_0]
                    address_embedding = self.get_address_embedding("shape_id_1")
                    lstm_input = torch.cat(
                        [obs_embedding_b, prev_sample_embedding, address_embedding]
                    )

                    # --Run LSTM
                    h, c = self.lstm_cell(lstm_input[None], hc)
                    hc = h, c

                    # --Extract params
                    shape_id_1_logits = self.shape_id_param_extractor(h)[0]

                    # --Sample shape id
                    shape_id_1 = torch.distributions.Categorical(logits=shape_id_1_logits).sample()

                    # # --Update trace
                    # trace.append(("shape_id_1", shape_id_1))

                    # Sample raw position 0
                    # --LSTM Input
                    prev_sample_embedding = self.shape_id_embeddings[shape_id_1]
                    address_embedding = self.get_address_embedding("raw_position_0")
                    lstm_input = torch.cat(
                        [obs_embedding_b, prev_sample_embedding, address_embedding]
                    )

                    # --Run LSTM
                    h, c = self.lstm_cell(lstm_input[None], hc)
                    hc = h, c

                    # --Extract params
                    raw_position_param = self.raw_position_param_extractor(h)[0]
                    raw_position_loc = raw_position_param[:2]
                    raw_position_scale = raw_position_param[2:].exp()

                    # --Sample raw_position
                    raw_position_0 = torch.distributions.Independent(
                        torch.distributions.Normal(raw_position_loc, raw_position_scale),
                        reinterpreted_batch_ndims=1,
                    ).sample()

                    # # --Update trace
                    # trace.append(("raw_position_0", raw_position_0))

                    # Sample raw position 1
                    # --LSTM Input
                    prev_sample_embedding = self.raw_position_embedder(raw_position_0[None])[0]
                    address_embedding = self.get_address_embedding("raw_position_1")
                    lstm_input = torch.cat(
                        [obs_embedding_b, prev_sample_embedding, address_embedding]
                    )

                    # --Run LSTM
                    h, c = self.lstm_cell(lstm_input[None], hc)
                    hc = h, c

                    # --Extract params
                    raw_position_param = self.raw_position_param_extractor(h)[0]
                    raw_position_loc = raw_position_param[:2]
                    raw_position_scale = raw_position_param[2:].exp()

                    # --Sample raw_position
                    raw_position_1 = torch.distributions.Independent(
                        torch.distributions.Normal(raw_position_loc, raw_position_scale),
                        reinterpreted_batch_ndims=1,
                    ).sample()

                    # # --Update trace
                    # trace.append(("raw_position_1", raw_position_1))
                    program_id_flattened[sample_id, element_id] = program_id
                    shape_ids_flattened[sample_id, element_id, 0] = shape_id_0
                    shape_ids_flattened[sample_id, element_id, 1] = shape_id_1
                    raw_positions_flattened[sample_id, element_id, 0] = raw_position_0
                    raw_positions_flattened[sample_id, element_id, 1] = raw_position_1

                # traces.append(trace)

        return (
            program_id_flattened.view(*[*sample_shape, *shape]),
            shape_ids_flattened.view(*[*sample_shape, *shape, self.max_num_shapes]),
            raw_positions_flattened.view(*[*sample_shape, *shape, self.max_num_shapes, 2]),
        )

    def sample_discrete(self, obs, sample_shape=[]):
        """z_d ~ q(z_d | x)

        Args
            obs [*shape, im_size, im_size]
            sample_shape

        Returns
            program_id [*sample_shape, *shape]
            shape_ids [*sample_shape, *shape, max_num_shapes=2]
        """
        # Extract
        shape = obs.shape[:-2]
        num_elements = util.get_num_elements(shape)
        num_samples = util.get_num_elements(sample_shape)

        # Get obs embedding (flattened)
        # [num_elements, obs_embedding_dim]
        obs_embedding = self.get_obs_embedding(obs.view(-1, self.im_size, self.im_size))

        # Init result
        program_id_flattened = torch.zeros((num_samples, num_elements), device=self.device).long()
        shape_ids_flattened = torch.zeros(
            (num_samples, num_elements, self.max_num_shapes), device=self.device
        ).long()

        # traces = []
        for sample_id in range(num_samples):
            for element_id in range(num_elements):
                # trace = []

                # Extract obs embedding
                obs_embedding_b = obs_embedding[element_id]

                # Sample program id
                # --LSTM Input
                prev_sample_embedding = torch.zeros(
                    (self.sample_embedding_dim,), device=self.device
                )
                address_embedding = self.get_address_embedding("program_id")
                lstm_input = torch.cat([obs_embedding_b, prev_sample_embedding, address_embedding])

                # --Run LSTM
                h, c = self.lstm_cell(lstm_input[None])
                hc = h, c

                # --Extract params
                program_id_logits = self.program_id_param_extractor(h)[0]

                # --Sample program_id
                program_id = torch.distributions.Categorical(logits=program_id_logits).sample()
                # trace.append(program_id)

                if program_id == 0:
                    # Sample shape id
                    # --LSTM Input
                    prev_sample_embedding = self.program_id_embeddings[program_id]
                    address_embedding = self.get_address_embedding("shape_id")
                    lstm_input = torch.cat(
                        [obs_embedding_b, prev_sample_embedding, address_embedding]
                    )

                    # --Run LSTM
                    h, c = self.lstm_cell(lstm_input[None], hc)
                    hc = h, c

                    # --Extract params
                    shape_id_logits = self.shape_id_param_extractor(h)[0]

                    # --Sample shape id
                    shape_id = torch.distributions.Categorical(logits=shape_id_logits).sample()

                    # # --Update trace
                    # trace.append(("raw_position", raw_position))
                    program_id_flattened[sample_id, element_id] = program_id
                    shape_ids_flattened[sample_id, element_id, 0] = shape_id
                else:
                    # Sample shape id 0
                    # --LSTM Input
                    prev_sample_embedding = self.program_id_embeddings[program_id]
                    address_embedding = self.get_address_embedding("shape_id_0")
                    lstm_input = torch.cat(
                        [obs_embedding_b, prev_sample_embedding, address_embedding]
                    )

                    # --Run LSTM
                    h, c = self.lstm_cell(lstm_input[None], hc)
                    hc = h, c

                    # --Extract params
                    shape_id_0_logits = self.shape_id_param_extractor(h)[0]

                    # --Sample shape id
                    shape_id_0 = torch.distributions.Categorical(logits=shape_id_0_logits).sample()

                    # # --Update trace
                    # trace.append(("shape_id_0", shape_id_0))

                    # Sample shape id 1
                    # --LSTM Input
                    prev_sample_embedding = self.shape_id_embeddings[shape_id_0]
                    address_embedding = self.get_address_embedding("shape_id_1")
                    lstm_input = torch.cat(
                        [obs_embedding_b, prev_sample_embedding, address_embedding]
                    )

                    # --Run LSTM
                    h, c = self.lstm_cell(lstm_input[None], hc)
                    hc = h, c

                    # --Extract params
                    shape_id_1_logits = self.shape_id_param_extractor(h)[0]

                    # --Sample shape id
                    shape_id_1 = torch.distributions.Categorical(logits=shape_id_1_logits).sample()

                    # # --Update trace
                    # trace.append(("raw_position_1", raw_position_1))
                    program_id_flattened[sample_id, element_id] = program_id
                    shape_ids_flattened[sample_id, element_id, 0] = shape_id_0
                    shape_ids_flattened[sample_id, element_id, 1] = shape_id_1

        return (
            program_id_flattened.view(*[*sample_shape, *shape]),
            shape_ids_flattened.view(*[*sample_shape, *shape, self.max_num_shapes]),
        )

    def sample_continuous(self, obs, discrete_latent, sample_shape=[]):
        """z_c ~ q(z_c | z_d, x)

        Args
            obs [*shape, im_size, im_size]
            discrete_latent
                program_id [*discrete_shape, *shape]
                shape_ids [*discrete_shape, *shape, max_num_shapes=2]
            sample_shape

        Returns
            raw_positions [*sample_shape, *discrete_shape, *shape, *shape, max_num_shapes=2, 2]

        """
        raise NotImplementedError

    def log_prob_discrete(self, obs, discrete_latent):
        """log q(z_d | x)

        Args
            obs [*shape, im_size, im_size]
            discrete_latent
                program_id [*discrete_shape, *shape]
                shape_ids [*discrete_shape, *shape, max_num_shapes=2]

        Returns [*discrete_shape, *shape]
        """
        raise NotImplementedError

    def log_prob_continuous(self, obs, discrete_latent, continuous_latent):
        """log q(z_c | z_d, x)

        Args
            obs [*shape, im_size, im_size]
            discrete_latent
                program_id [*discrete_shape, *shape]
                shape_ids [*discrete_shape, *shape, max_num_shapes=2]
            continuous_latent (raw_positions)
                [*continuous_shape, *discrete_shape, *shape, max_num_shapes=2, 2]

        Returns [*continuous_shape, *discrete_shape, *shape]
        """
        raise NotImplementedError
