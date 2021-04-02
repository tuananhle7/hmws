import torch
import torch.nn as nn
import render
import util


class GenerativeModel(nn.Module):
    """Samples order of blocks (discrete), their raw positions (continuous) and renders image.
    """

    def __init__(
        self,
        num_primitives=3,
        max_num_blocks=3,
        im_size=32,
        obs_scale=1.0,
        obs_dist_type="normal",
        true_primitives=False,
    ):
        super().__init__()

        # Init
        self.num_primitives = num_primitives
        self.max_num_blocks = max_num_blocks
        self.num_channels = 3
        self.num_rows = im_size
        self.num_cols = im_size
        self.obs_scale = obs_scale
        if obs_dist_type == "normal":
            self.obs_dist = torch.distributions.Normal
        elif obs_dist_type == "laplace":
            self.obs_dist = torch.distributions.Laplace

        # Primitive parameters (parameters of symbols)
        self.true_primitives = true_primitives
        if self.true_primitives:
            assert self.num_primitives == 3
        else:
            self.primitive_module_list = nn.ModuleList(
                [render.LearnableSquare(f"{i}") for i in range(self.num_primitives)]
            )

        # Rendering parameters
        self.raw_color_sharpness = nn.Parameter(torch.randn(()))
        self.raw_blur = nn.Parameter(torch.randn(()))

    @property
    def primitives(self):
        if self.true_primitives:
            return [
                render.Square(
                    "A",
                    torch.tensor([1.0, 0.0, 0.0], device=self.device),
                    torch.tensor(0.3, device=self.device),
                ),
                render.Square(
                    "B",
                    torch.tensor([0.0, 1.0, 0.0], device=self.device),
                    torch.tensor(0.4, device=self.device),
                ),
                render.Square(
                    "C",
                    torch.tensor([0.0, 0.0, 1.0], device=self.device),
                    torch.tensor(0.5, device=self.device),
                ),
            ]
        else:
            return self.primitive_module_list

    @property
    def device(self):
        return self.raw_color_sharpness.device

    @property
    def num_blocks_dist(self):
        """Prior distribution p(num_blocks)
        batch_shape [], event_shape []
        """
        return util.CategoricalPlusOne(logits=torch.ones(self.max_num_blocks, device=self.device))

    @property
    def stacking_order_logits(self):
        return torch.ones((self.max_num_blocks, self.num_primitives), device=self.device)

    @property
    def attachment_probs(self):
        prob = torch.ones((self.max_num_blocks,), device=self.device) * 0.1
        return torch.stack([1 - prob, prob], dim=-1)

    def latent_log_prob(self, latent):
        """Prior log p(z)

        Args:
            latent:
                num_blocks [*shape]
                stacking_program
                    stacking_order [*shape, max_num_blocks]
                    attachment [*shape, max_num_blocks]
                raw_locations [*shape, max_num_blocks]

        Returns: [*shape]
        """
        # Extract
        num_blocks, stacking_program, raw_locations = latent

        # Log prob of num_blocks
        num_blocks_log_prob = self.num_blocks_dist.log_prob(num_blocks)

        # Log prob of stacking_program
        logits = torch.ones((self.max_num_blocks, self.num_primitives), device=self.device)
        stacking_program_log_prob = util.pad_tensor(
            torch.distributions.Categorical(logits=logits).log_prob(stacking_program), num_blocks, 0
        ).sum(-1)

        # Log prob of raw_locations
        # --Compute dist params
        loc = torch.zeros(self.max_num_blocks, device=self.device)
        scale = torch.ones(self.max_num_blocks, device=self.device)
        # --Compute log prob [*shape]
        raw_locations_log_prob = util.pad_tensor(
            torch.distributions.Normal(loc, scale).log_prob(raw_locations), num_blocks, 0,
        ).sum(-1)

        return num_blocks_log_prob + stacking_program_log_prob + raw_locations_log_prob

    def latent_sample(self, sample_shape=[]):
        """Sample from p(z)

        Args
            sample_shape

        Returns
            latent:
                num_blocks [*sample_shape]
                stacking_program
                    stacking_order [*sample_shape, max_num_blocks]
                    attachment [*sample_shape, max_num_blocks]
                raw_locations [*sample_shape, max_num_blocks]
        """
        # Sample num_blocks
        num_blocks = self.num_blocks_dist.sample(sample_shape)

        # Sample stacking_order
        stacking_order = torch.distributions.Categorical(logits=self.stacking_order_logits).sample(
            sample_shape
        )

        # Sample attachment
        attachment = torch.distributions.Categorical(probs=self.attachment_probs).sample(
            sample_shape
        )

        # Sample raw_locations
        # --Compute dist params
        loc = torch.zeros(self.max_num_blocks, device=self.device)
        scale = torch.ones(self.max_num_blocks, device=self.device)
        # --Compute log prob [*shape]
        raw_locations = torch.distributions.Normal(loc, scale).sample(sample_shape)

        return num_blocks, (stacking_order, attachment), raw_locations

    def get_obs_loc(self, latent):
        """Location parameter of p(x | z)

        Args:
            latent:
                num_blocks [*shape]
                stacking_program
                    stacking_order [*shape, max_num_blocks]
                    attachment [*shape, max_num_blocks]
                raw_locations [*shape, max_num_blocks]

        Returns: [*shape, num_channels, num_rows, num_cols]
        """
        # Extract stuff
        num_blocks, stacking_program, raw_locations = latent

        # Compute stuff
        return render.soft_render_variable_num_blocks(
            self.primitives,
            num_blocks,
            stacking_program,
            raw_locations,
            self.raw_color_sharpness,
            self.raw_blur,
        )

    def get_obs_loc_hard(self, latent):
        """Location parameter of p(x | z)

        Args:
            latent:
                num_blocks [*shape]
                stacking_program
                    stacking_order [*shape, max_num_blocks]
                    attachment [*shape, max_num_blocks]
                raw_locations [*shape, max_num_blocks]

        Returns: [*shape, num_channels, num_rows, num_cols]
        """
        # Extract stuff
        num_blocks, (stacking_order, attachment), raw_locations = latent

        # Compute stuff
        return render.render_batched(self.primitives, num_blocks, stacking_order, raw_locations,)

    def log_prob(self, latent, obs):
        """Log joint probability of the generative model
        log p(z, x)

        Args:
            latent:
                num_blocks [*shape]
                TODO
                stacking_program
                    stacking_order [*shape, max_num_blocks]
                    attachment [*shape, max_num_blocks]
                raw_locations [*shape, max_num_blocks]
            obs [*shape, num_channels, num_rows, num_cols]

        Returns: [*shape]
        """
        # p(z)
        latent_log_prob = self.latent_log_prob(latent)

        # p(x | z)
        obs_log_prob = torch.distributions.Independent(
            self.obs_dist(loc=self.get_obs_loc(latent), scale=self.obs_scale),
            reinterpreted_batch_ndims=3,
        ).log_prob(obs)

        return latent_log_prob + obs_log_prob

    def sample(self, sample_shape=[]):
        """Sample from p(z, x)

        Args
            sample_shape

        Returns
            latent:
                num_blocks [*sample_shape]
                stacking_program
                    stacking_order [*sample_shape, max_num_blocks]
                    attachment [*sample_shape, max_num_blocks]
                raw_locations [*sample_shape, max_num_blocks]
            obs [*sample_shape, num_channels, num_rows, num_cols]
        """
        # p(z)
        latent = self.latent_sample(sample_shape)

        # p(x | z)
        obs = self.get_obs_loc_hard(latent)

        return latent, obs


class Guide(nn.Module):
    """CNN going from image to order and locations
    """

    def __init__(self, num_primitives=3, max_num_blocks=3, im_size=32):
        super().__init__()

        # Init
        self.num_primitives = num_primitives
        self.max_num_blocks = max_num_blocks
        self.num_channels = 3
        self.num_rows = im_size
        self.num_cols = im_size

        # Obs embedder
        self.obs_embedder = util.init_cnn(16)
        self.obs_embedding_dim = 16  # determined by running the network forward
        # self.obs_embedding_dim = 400  # determined by running the network forward

        # Mapping from obs embedding to num_blocks params
        self.num_blocks_params_extractor = nn.Linear(self.obs_embedding_dim, self.max_num_blocks)

        # Mapping from obs embedding to stacking_program params
        self.stacking_program_params_extractor = nn.Linear(
            self.obs_embedding_dim, self.max_num_blocks * self.num_primitives
        )

        # Mapping from obs embedding to location params
        self.raw_location_params_extractor = nn.Linear(
            self.obs_embedding_dim, self.max_num_blocks * 2
        )

    @property
    def device(self):
        return next(self.obs_embedder.parameters()).device

    def get_dist_params(self, obs):
        """
        Args
            obs: [*shape, num_channels, num_rows, num_cols]

        Returns
            num_blocks_params [*shape, max_num_blocks]
            stacking_program_params [*shape, max_num_blocks, num_primitives]
            raw_locations_params
                loc [*shape, max_num_blocks]
                scale [*shape, max_num_blocks]
        """
        shape = obs.shape[:-3]
        obs_flattened = obs.reshape(-1, self.num_channels, self.num_rows, self.num_cols)

        # [num_elements, obs_embedding_dim]
        obs_embedding = self.obs_embedder(obs_flattened)

        # Num blocks
        num_blocks_params = self.num_blocks_params_extractor(obs_embedding).view(
            *shape, self.max_num_blocks
        )

        # Stacking program
        stacking_program_params = self.stacking_program_params_extractor(obs_embedding).view(
            *shape, self.max_num_blocks, self.num_primitives
        )

        # Raw location
        raw_location_params_extractor = self.raw_location_params_extractor(obs_embedding)
        loc = raw_location_params_extractor[..., : self.max_num_blocks].view(
            *shape, self.max_num_blocks
        )
        scale = (
            raw_location_params_extractor[..., self.max_num_blocks :]
            .exp()
            .view(*shape, self.max_num_blocks)
        )

        return num_blocks_params, stacking_program_params, (loc, scale)

    def log_prob(self, obs, latent):
        """
        Args
            obs [*shape, num_channels, num_rows, num_cols]
            latent:
                num_blocks [*shape]
                TODO
                stacking_program
                    stacking_order [*shape, max_num_blocks]
                    attachment [*shape, max_num_blocks]
                raw_locations [*shape, max_num_blocks]

        Returns [*shape]
        """
        # Extract
        num_blocks, stacking_program, raw_locations = latent

        # Compute params
        num_blocks_params, stacking_program_params, (loc, scale) = self.get_dist_params(obs)

        # log q(Num blocks | x)
        num_blocks_log_prob = util.CategoricalPlusOne(logits=num_blocks_params).log_prob(num_blocks)

        # log q(Stacking program | x)
        stacking_program_log_prob = util.pad_tensor(
            torch.distributions.Categorical(logits=stacking_program_params).log_prob(
                stacking_program
            ),
            num_blocks,
            0,
        ).sum(-1)

        # log q(Raw locations | x)
        raw_locations_log_prob = util.pad_tensor(
            torch.distributions.Normal(loc, scale).log_prob(raw_locations), num_blocks, 0,
        ).sum(-1)

        return num_blocks_log_prob + stacking_program_log_prob + raw_locations_log_prob

    def sample(self, obs, sample_shape=[]):
        """z ~ q(z | x)

        Args
            obs [*shape, num_channels, num_rows, num_cols]

        Returns
            num_blocks [*sample_shape, *shape]
            TODO
            stacking_program
                stacking_order [*sample_shape, *shape, max_num_blocks]
                attachment [*sample_shape, *shape, max_num_blocks]
            raw_locations [*sample_shape, *shape, max_num_blocks]
        """
        # Compute params
        num_blocks_params, stacking_program_params, (loc, scale) = self.get_dist_params(obs)

        # Sample from q(Num blocks | x)
        num_blocks = util.CategoricalPlusOne(logits=num_blocks_params).sample(sample_shape)

        # Sample from q(Stacking program | x)
        stacking_program = torch.distributions.Categorical(logits=stacking_program_params).sample(
            sample_shape
        )

        # Sample from q(Raw locations | x)
        raw_locations = torch.distributions.Normal(loc, scale).sample(sample_shape)

        return num_blocks, stacking_program, raw_locations