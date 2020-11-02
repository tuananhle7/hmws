import util
import render
import torch
import torch.nn as nn


class JointDistribution:
    """p(x_{1:N}) = ∏_n p(x_n)
    Args:
        dists: list of distributions p(x_n)
    """

    def __init__(self, dists):
        self.dists = dists

    def sample(self, sample_shape=[]):
        return tuple([dist.sample(sample_shape) for dist in self.dists])

    def log_prob(self, values):
        return sum([dist.log_prob(value) for dist, value in zip(self.dists, values)])


class TrueGenerativeModel(nn.Module):
    def __init__(self, im_size=64):
        super().__init__()
        self.im_size = im_size
        self.register_buffer("blank_canvas", torch.zeros((self.im_size, self.im_size)))

    @property
    def device(self):
        return self.blank_canvas.device

    @property
    def latent_dist(self):
        """
        Returns distribution with batch_shape [] and event_shape ([2], [])
        """
        # Position distribution
        half = torch.ones((2,), device=self.device) * 0.5
        position_dist = torch.distributions.Independent(
            torch.distributions.Uniform(-half, half), reinterpreted_batch_ndims=1
        )

        # Scale distribution
        scale_dist = torch.distributions.Uniform(
            torch.tensor(0.1, device=self.device), torch.tensor(0.9, device=self.device)
        )

        return JointDistribution([position_dist, scale_dist])

    def sample(self, sample_shape=[]):
        """
        Args:
            sample_shape: list-like object (default [])

        Returns:
            latent
                position [*sample_shape, 2]
                scale [*sample_shape]
            obs [*sample_shape, im_size, im_size]
        """
        # Sample latent
        latent = self.latent_dist.sample(sample_shape)

        # Sample obs
        position, scale = latent
        position_flattened, scale_flattened = position.view(-1, 2), scale.view(-1)
        num_samples = int(torch.tensor(sample_shape).prod().long().item())
        obs = []
        for sample_id in range(num_samples):
            obs.append(
                render.render_heart(
                    (position_flattened[sample_id], scale_flattened[sample_id]), self.blank_canvas
                ).detach()
            )
        obs = torch.stack(obs).view(*[*sample_shape, self.im_size, self.im_size])

        return latent, obs


class GenerativeModel(nn.Module):
    def __init__(self, im_size=64):
        super().__init__()
        self.im_size = im_size
        self.mlp = util.init_mlp(2, 1, 100, 3)

    @property
    def device(self):
        return next(self.mlp.parameters()).device

    def get_obs_dist(self, latent, num_rows, num_cols):
        """p(obs | latent)

        Args
            latent
                position [*shape, 2]
                scale [*shape]
            num_rows (int)
            num_cols (int)

        Returns: distribution with batch_shape [*shape] and event_shape
            [num_rows, num_cols]
        """
        # Extract
        position, scale = latent
        shape = scale.shape
        # num_samples = int(torch.tensor(shape).prod().long().item())
        position_x, position_y = position[..., 0], position[..., 1]  # [*shape]
        # [num_rows, num_cols]
        canvas_x, canvas_y = render.get_canvas_xy(num_rows, num_cols, self.device)

        # Shift and scale
        # [num_samples, num_rows, num_cols]
        canvas_x = (canvas_x[None] - position_x.view(-1, 1, 1)) / scale.view(-1, 1, 1)
        canvas_y = (canvas_y[None] - position_y.view(-1, 1, 1)) / scale.view(-1, 1, 1)

        # Build input
        # [num_samples * num_rows * num_cols, 2]
        mlp_input = torch.stack([canvas_x, canvas_y], dim=-1).view(-1, 2)

        # Run MLP
        logits = self.mlp(mlp_input).view(*[*shape, num_rows, num_cols])

        return torch.distributions.Independent(
            torch.distributions.Bernoulli(logits=logits), reinterpreted_batch_ndims=2
        )

    @property
    def latent_dist(self):
        """
        Returns distribution with batch_shape [] and event_shape ([2], [])
        """
        # Position distribution
        half = torch.ones((2,), device=self.device) * 0.5
        position_dist = torch.distributions.Independent(
            torch.distributions.Uniform(-half, half), reinterpreted_batch_ndims=1
        )

        # Scale distribution
        scale_dist = torch.distributions.Uniform(
            torch.tensor(0.1, device=self.device), torch.tensor(0.9, device=self.device)
        )

        return JointDistribution([position_dist, scale_dist])

    def log_prob(self, latent, obs):
        """
        Args:
            latent
                position [*shape, 2]
                scale [*shape]
            obs [*shape, num_rows, num_cols]

        Returns: [*shape]
        """
        return self.latent_dist.log_prob(latent) + self.get_obs_dist(
            latent, self.im_size, self.im_size
        ).log_prob(obs)

    def sample(self, sample_shape=[]):
        """
        Args:
            sample_shape: list-like object (default [])

        Returns:
            latent
                position [*sample_shape, 2]
                scale [*sample_shape]
            obs [*sample_shape, im_size, im_size]
        """
        # Sample latent
        latent = self.latent_dist.sample(sample_shape)

        # Sample obs
        obs = self.get_obs_dist(latent, self.im_size, self.im_size).sample()

        return latent, obs


class Guide(nn.Module):
    def __init__(self, im_size=64):
        super().__init__()
        self.im_size = im_size
        self.cnn = util.init_cnn(output_dim=16)
        self.cnn_features_dim = 400  # computed manually
        self.position_mlp = util.init_mlp(
            self.cnn_features_dim, 2 * 2, hidden_dim=100, num_layers=1
        )
        self.scale_mlp = util.init_mlp(self.cnn_features_dim, 2, hidden_dim=100, num_layers=1)

    @property
    def device(self):
        return next(self.cnn.parameters()).device

    def get_cnn_features(self, obs):
        """
        Args:
            obs: [batch_size, im_size, im_size]

        Returns: [batch_size, cnn_features_dim]
        """
        batch_size = obs.shape[0]
        return self.cnn(obs[:, None]).view(batch_size, -1)

    def get_dist(self, obs):
        """
        Args:
            obs: [batch_size, im_size, im_size]

        Returns: dist with batch_shape [batch_size] and event_shape ([2], [])
        """
        cnn_features = self.get_cnn_features(obs)

        # Position dist
        raw_loc, raw_scale = self.position_mlp(cnn_features).chunk(2, dim=-1)
        loc, scale = raw_loc, raw_scale.exp()
        position_dist = torch.distributions.Independent(
            torch.distributions.Normal(loc, scale), reinterpreted_batch_ndims=1
        )

        # Scale dist
        raw_loc, raw_scale = self.scale_mlp(cnn_features).chunk(2, dim=-1)
        loc, scale = raw_loc.view(-1), raw_scale.view(-1).exp()
        scale_dist = torch.distributions.Normal(loc, scale)

        return JointDistribution([position_dist, scale_dist])

    def log_prob(self, obs, latent):
        """
        Args:
            obs [*shape, im_size, im_size]
            latent
                position [*shape, 2]
                scale [*shape]

        Returns: [*shape]
        """
        obs_flattened = obs.view(-1, self.im_size, self.im_size)
        position, scale = latent
        shape = scale.shape
        latent_flattened = position.view(-1, 2), scale.view(-1)

        return self.get_dist(obs_flattened).log_prob(latent_flattened).view(*shape)

    def sample(self, obs, sample_shape=[]):
        """
        Args:
            obs [*shape, im_size, im_size]

        Returns:
            latent
                position [*sample_shape, *shape, 2]
                scale [*sample_shape, *shape]
        """
        shape = obs.shape[:-1]
        obs_flattened = obs.view(-1, self.im_size, self.im_size)

        position_flattened, scale_flattened = self.get_dist(obs_flattened).sample(sample_shape)
        latent = (
            position_flattened.view(*[*sample_shape, *shape, 2]),
            scale_flattened.view(*[*sample_shape, *shape]),
        )
        return latent
