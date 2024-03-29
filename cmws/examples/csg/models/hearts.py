import torch
import torch.nn as nn
from cmws import util
from cmws.examples.csg import render


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

        return util.JointDistribution([position_dist, scale_dist])

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
        self.mlp = util.init_mlp(2, 1, 100, 3, non_linearity=nn.ReLU())

    @property
    def device(self):
        return next(self.mlp.parameters()).device

    def get_obs_dist(self, latent, num_rows=None, num_cols=None):
        """p(obs | latent)

        Args
            latent
                raw_position [*shape, 2]
                raw_scale [*shape]
            num_rows (int)
            num_cols (int)

        Returns: distribution with batch_shape [*shape] and event_shape
            [num_rows, num_cols]
        """
        if num_rows is None:
            num_rows = self.im_size
        if num_cols is None:
            num_cols = self.im_size
        # Extract
        raw_position, raw_scale = latent
        position = raw_position.sigmoid() - 0.5
        scale = raw_scale.sigmoid() * 0.8 + 0.1

        # util.logging.info(f"position = {position} | scale = {scale}")
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

        if torch.isnan(logits).any():
            raise RuntimeError("nan")

        return torch.distributions.Independent(
            torch.distributions.Bernoulli(logits=logits), reinterpreted_batch_ndims=2
        )

    @property
    def latent_dist(self):
        """
        Returns distribution with batch_shape [] and event_shape ([2], [])
        """
        # Position distribution
        raw_position_dist = torch.distributions.Independent(
            torch.distributions.Normal(
                torch.zeros((2,), device=self.device), torch.ones((2,), device=self.device)
            ),
            reinterpreted_batch_ndims=1,
        )

        # Scale distribution
        raw_scale_dist = torch.distributions.Normal(
            torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device)
        )

        return util.JointDistribution([raw_position_dist, raw_scale_dist])

    def log_prob(self, latent, obs):
        """
        Args:
            latent
                raw_position [*shape, 2]
                raw_scale [*shape]
            obs [*shape, num_rows, num_cols]

        Returns: [*shape]
        """
        latent_log_prob = self.latent_dist.log_prob(latent)
        obs_log_prob = self.get_obs_dist(latent, self.im_size, self.im_size).log_prob(obs)

        if torch.isnan(latent_log_prob + obs_log_prob).any():
            raise RuntimeError("nan")

        return latent_log_prob + obs_log_prob

    def sample(self, sample_shape=[]):
        """
        Args:
            sample_shape: list-like object (default [])

        Returns:
            latent
                raw_position [*sample_shape, 2]
                raw_scale [*sample_shape]
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
        self.raw_position_mlp = util.init_mlp(
            self.cnn_features_dim, 2 * 2, hidden_dim=100, num_layers=1
        )
        self.raw_scale_mlp = util.init_mlp(self.cnn_features_dim, 2, hidden_dim=100, num_layers=1)

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
        raw_loc, raw_scale = self.raw_position_mlp(cnn_features).chunk(2, dim=-1)
        loc, scale = raw_loc, raw_scale.exp()
        raw_position_dist = torch.distributions.Independent(
            torch.distributions.Normal(loc, scale), reinterpreted_batch_ndims=1
        )

        # Scale dist
        raw_loc, raw_scale = self.raw_scale_mlp(cnn_features).chunk(2, dim=-1)
        loc, scale = raw_loc.view(-1), raw_scale.view(-1).exp()
        raw_scale_dist = torch.distributions.Normal(loc, scale)

        return util.JointDistribution([raw_position_dist, raw_scale_dist])

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

        # Flatten latent
        raw_position, raw_scale = latent
        shape = raw_scale.shape
        latent_flattened = raw_position.view(-1, 2), raw_scale.view(-1)

        return self.get_dist(obs_flattened).log_prob(latent_flattened).view(*shape)

    def sample(self, obs, sample_shape=[]):
        """
        Args:
            obs [*shape, im_size, im_size]

        Returns:
            latent
                raw_position [*sample_shape, *shape, 2]
                raw_scale [*sample_shape, *shape]
        """
        shape = obs.shape[:-2]
        obs_flattened = obs.view(-1, self.im_size, self.im_size)

        raw_position_flattened, raw_scale_flattened = self.get_dist(obs_flattened).sample(
            sample_shape
        )
        latent = (
            raw_position_flattened.view(*[*sample_shape, *shape, 2]),
            raw_scale_flattened.view(*[*sample_shape, *shape]),
        )
        return latent

    def rsample(self, obs, sample_shape=[]):
        """
        Args:
            obs [*shape, im_size, im_size]

        Returns:
            latent
                raw_position [*sample_shape, *shape, 2]
                raw_scale [*sample_shape, *shape]
        """
        shape = obs.shape[:-2]
        obs_flattened = obs.view(-1, self.im_size, self.im_size)

        raw_position_flattened, raw_scale_flattened = self.get_dist(obs_flattened).rsample(
            sample_shape
        )
        latent = (
            raw_position_flattened.view(*[*sample_shape, *shape, 2]),
            raw_scale_flattened.view(*[*sample_shape, *shape]),
        )
        return latent
