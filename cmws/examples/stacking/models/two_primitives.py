import torch
import torch.nn as nn
from cmws import util
from cmws.examples.stacking import render


class GenerativeModel(nn.Module):
    """Samples order of blocks (discrete), their raw positions (continuous) and renders image.
    """

    def __init__(self, im_size=32, obs_scale=1.0, obs_dist_type="normal"):
        super().__init__()

        # Init
        self.num_primitives = 2
        self.num_channels = 3
        self.num_rows = im_size
        self.num_cols = im_size
        self.obs_scale = obs_scale
        if obs_dist_type == "normal":
            self.obs_dist = torch.distributions.Normal
        elif obs_dist_type == "laplace":
            self.obs_dist = torch.distributions.Laplace

        # Prior on order
        self.register_buffer("order_logits", torch.ones(2))

        # Primitive parameters (parameters of symbols)
        self.primitives = nn.ModuleList(
            [render.LearnableSquare(f"{i}") for i in range(self.num_primitives)]
        )

        # Rendering parameters
        self.raw_color_sharpness = nn.Parameter(torch.rand(()))
        self.raw_blur = nn.Parameter(torch.rand(()))

    @property
    def device(self):
        return self.order_logits.device

    @property
    def latent_dist(self):
        """Prior distribution p(z)
        batch_shape [], event_shape ([], [2])
        """
        loc = torch.zeros((2,), device=self.device)
        scale = torch.ones((2,), device=self.device)
        return util.JointDistribution(
            [
                torch.distributions.Categorical(logits=self.order_logits),
                torch.distributions.Independent(
                    torch.distributions.Normal(loc, scale), reinterpreted_batch_ndims=1
                ),
            ]
        )

    def get_obs_loc(self, latent):
        """Location parameter of p(x | z)

        Args:
            latent:
                order [*shape] 0 means primitive 0 is at the bottom
                raw_locations [*shape, 2]

        Returns: [*shape, num_channels, num_rows, num_cols]
        """
        # Extract stuff
        order, raw_locations = latent

        # Make stacking program [*shape, 2]
        stacking_program = torch.stack([order, 1 - order], dim=-1)

        # Compute stuff
        return render.soft_render_batched(
            self.primitives,
            stacking_program,
            raw_locations,
            self.raw_color_sharpness,
            self.raw_blur,
        )

    def get_obs_loc_hard(self, latent):
        """Location parameter of p(x | z)

        Args:
            latent:
                order [*shape] 0 means primitive 0 is at the bottom
                raw_locations [*shape, 2]

        Returns: [*shape, num_channels, num_rows, num_cols]
        """
        # Extract stuff
        order, raw_locations = latent
        shape = order.shape
        num_samples = int(torch.tensor(shape).prod().item())

        # Make stacking program [*shape, 2]
        stacking_program = torch.stack([order, 1 - order], dim=-1)

        # Compute stuff
        loc = []
        for sample_id in range(num_samples):
            loc.append(
                render.render(
                    self.primitives,
                    stacking_program.view(-1, 2)[sample_id],
                    raw_locations.view(-1, 2)[sample_id],
                )
            )
        return torch.stack(loc).view(*[*shape, self.num_channels, self.num_rows, self.num_cols])

    def log_prob(self, latent, obs):
        """Log joint probability of the generative model
        log p(z, x)

        Args:
            latent:
                order [*shape] 0 means primitive 0 is at the bottom
                raw_locations [*shape, 2]
            obs [*shape, num_channels, num_rows, num_cols]

        Returns: [*shape]
        """
        # p(z)
        latent_log_prob = self.latent_dist.log_prob(latent)

        # p(x | z)
        obs_log_prob = torch.distributions.Independent(
            self.obs_dist(loc=self.get_obs_loc(latent), scale=self.obs_scale),
            reinterpreted_batch_ndims=3,
        ).log_prob(obs)

        return latent_log_prob + obs_log_prob


class Guide(nn.Module):
    """CNN going from image to order and locations
    """

    def __init__(self, im_size=32):
        super().__init__()

        # Init
        self.num_channels = 3
        self.num_rows = im_size
        self.num_cols = im_size

        # Obs embedder
        self.obs_embedder = util.init_cnn(16, input_num_channels=3)
        self.obs_embedding_dim = 16  # determined by running the network forward

        # Mapping from obs embedding to location params
        self.raw_location_params_extractor = nn.Linear(self.obs_embedding_dim, 2 * 2)

        # Mapping from obs embedding to order params
        self.order_params_extractor = nn.Linear(self.obs_embedding_dim, 2)

    @property
    def device(self):
        return next(self.obs_embedder.parameters()).device

    def get_dist_params(self, obs):
        """
        Args
            obs: [*shape, num_channels, num_rows, num_cols]

        Returns
            loc [*shape, 2]
            scale [*shape, 2]
            logits [*shape, 2]
        """
        shape = obs.shape[:-3]
        obs_flattened = obs.reshape(-1, self.num_channels, self.num_rows, self.num_cols)

        obs_embedding = self.obs_embedder(obs_flattened)

        # Raw location
        param = self.raw_location_params_extractor(obs_embedding)
        loc = param[..., 0:2].view(*shape, 2)
        scale = param[..., 2:4].exp().view(*shape, 2)

        # Order
        logits = self.order_params_extractor(obs_embedding).view(*shape, 2)

        return loc, scale, logits

    def get_dist(self, obs):
        """Returns q(z | x)

        Args
            obs [*shape, num_channels, num_rows, num_cols]

        Returns distribution object with batch_shape [*shape] and event_shape ([], [2])
        """
        loc, scale, logits = self.get_dist_params(obs)
        return util.JointDistribution(
            [
                torch.distributions.Categorical(logits=logits),
                torch.distributions.Independent(
                    torch.distributions.Normal(loc, scale), reinterpreted_batch_ndims=1
                ),
            ]
        )

    def log_prob(self, obs, latent):
        """
        Args
            obs [*shape, num_channels, num_rows, num_cols]
            latent:
                order [*shape] 0 means primitive 0 is at the bottom
                raw_locations [*shape, 2]

        Returns [*shape]
        """
        return self.get_dist(obs).log_prob(latent)

    def sample(self, obs, sample_shape=[]):
        """z ~ q(z | x)

        Args
            obs [*shape, num_channels, num_rows, num_cols]

        Returns
            order [*sample_shape, *shape]
            raw_locations [*sample_shape, *shape, 2]
        """
        return self.get_dist(obs).sample(sample_shape)
