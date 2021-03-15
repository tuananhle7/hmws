import torch
import torch.nn as nn
import render
import util


class GenerativeModel(nn.Module):
    """Samples raw position (continuous) and renders image.
    """

    def __init__(self, im_size=64, obs_scale=1.0, obs_dist_type="normal"):
        super().__init__()

        # Init
        self.num_channels = 3
        self.num_rows = im_size
        self.num_cols = im_size
        self.obs_scale = obs_scale
        if obs_dist_type == "normal":
            self.obs_dist = torch.distributions.Normal
        elif obs_dist_type == "laplace":
            self.obs_dist = torch.distributions.Laplace

        # Primitive parameters (parameters of symbols)
        self.primitive = render.LearnableSquare()

        # Rendering parameters
        self.raw_color_sharpness = nn.Parameter(torch.rand(()))
        self.raw_blur = nn.Parameter(torch.rand(()))

    @property
    def device(self):
        return self.primitive.device

    @property
    def latent_dist(self):
        """Prior distribution p(z)
        batch_shape [], event_shape []
        """
        loc = torch.zeros((), device=self.device)
        scale = torch.ones((), device=self.device)
        return torch.distributions.Normal(loc, scale)

    def get_obs_loc(self, latent):
        """Location parameter of p(x | z)

        Args:
            latent [*shape]: raw location

        Returns: [*shape, num_channels, num_rows, num_cols]
        """
        # Extract stuff
        raw_location = latent
        shape = latent.shape
        num_samples = int(torch.tensor(shape).prod().item())

        # Compute stuff
        loc = []
        for sample_id in range(num_samples):
            loc.append(
                render.soft_render(
                    [self.primitive],
                    torch.zeros((1,), device=self.device).long(),
                    raw_location.view(-1)[sample_id][None],
                    self.raw_color_sharpness,
                    self.raw_blur,
                )
            )
        return torch.stack(loc).view(*[*shape, self.num_channels, self.num_rows, self.num_cols])

    def get_obs_loc_hard(self, latent):
        """Location parameter of p(x | z)

        Args:
            latent [*shape]: raw location

        Returns: [*shape, num_channels, num_rows, num_cols]
        """
        # Extract stuff
        raw_location = latent
        shape = latent.shape
        num_samples = int(torch.tensor(shape).prod().item())

        # Compute stuff
        loc = []
        for sample_id in range(num_samples):
            loc.append(
                render.render(
                    [self.primitive],
                    torch.zeros((1,), device=self.device).long(),
                    raw_location.view(-1)[sample_id][None],
                )
            )
        return torch.stack(loc).view(*[*shape, self.num_channels, self.num_rows, self.num_cols])

    def log_prob(self, latent, obs):
        """Log joint probability of the generative model
        log p(z, x)

        Args:
            latent [*shape]: raw location
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
    """CNN going from image to location
    """

    def __init__(self, im_size=64):
        super().__init__()

        # Init
        self.num_channels = 3
        self.num_rows = im_size
        self.num_cols = im_size

        # Obs embedder
        self.obs_embedder = util.init_cnn(16)
        self.obs_embedding_dim = 400  # determined by running the network forward

        # Mapping from obs embedding to location params
        self.raw_location_params_extractor = nn.Linear(self.obs_embedding_dim, 2)

    @property
    def device(self):
        return next(self.obs_embedder.parameters()).device

    def get_raw_location_params(self, obs):
        """
        Args
            obs: [*shape, num_channels, num_rows, num_cols]

        Returns
            loc [*shape]
            scale [*shape]
        """
        shape = obs.shape[:-3]
        obs_flattened = obs.reshape(-1, self.num_channels, self.num_rows, self.num_cols)

        obs_embedding = self.obs_embedder(obs_flattened)
        param = self.raw_location_params_extractor(obs_embedding)
        loc = param[..., 0].view(*shape)
        scale = param[..., 1].exp().view(*shape)
        return loc, scale

    def get_dist(self, obs):
        """Returns q(z | x)

        Args
            obs [*shape, num_channels, num_rows, num_cols]

        Returns distribution object with batch_shape [*shape] and event_shape []
        """
        return torch.distributions.Normal(*self.get_raw_location_params(obs))

    def log_prob(self, obs, latent):
        """
        Args
            obs [*shape, num_channels, num_rows, num_cols]
            latent [*shape]

        Returns [*shape]
        """
        return self.get_dist(obs).log_prob(latent)

    def sample(self, obs, sample_shape=[]):
        """z ~ q(z | x)

        Args
            obs [*shape, num_channels, num_rows, num_cols]

        Returns [*sample_shape, *shape]
        """
        return self.get_dist(obs).sample(sample_shape)

    def rsample(self, obs, sample_shape=[]):
        """z ~ q(z | x) (reparameterized)

        Args
            obs [*shape, num_channels, num_rows, num_cols]

        Returns [*sample_shape, *shape]
        """
        return self.get_dist(obs).rsample(sample_shape)
