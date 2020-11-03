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

    def rsample(self, sample_shape=[]):
        return tuple([dist.rsample(sample_shape) for dist in self.dists])

    def log_prob(self, values):
        return sum([dist.log_prob(value) for dist, value in zip(self.dists, values)])


class RectanglePoseDistribution:
    def __init__(self, device):
        self.device = device
        self.lim = torch.tensor(0.8, device=self.device)

    def sample(self, sample_shape):
        """
        Args
            sample_shape

        Returns [*sample_shape, 4]
        """
        minus_lim = -self.lim
        padding = 0.2
        min_x = torch.distributions.Uniform(minus_lim, self.lim - padding).sample(sample_shape)
        max_x = torch.distributions.Uniform(min_x + padding, self.lim).sample()
        min_y = torch.distributions.Uniform(minus_lim, self.lim - padding).sample(sample_shape)
        max_y = torch.distributions.Uniform(min_y + padding, self.lim).sample()
        return torch.stack([min_x, min_y, max_x, max_y], dim=-1)

    def log_prob(self, xy_lims):
        """
        Args
            xy_lims [*shape, 4]

        Returns [*shape]
        """
        # HACK
        shape = xy_lims.shape[:-1]
        return torch.zeros(shape, device=xy_lims.device)
        # min_x, min_y, max_x, max_y = [xy_lims[..., i] for i in range(4)]
        # minus_one = -self.one
        # min_x_log_prob = torch.distributions.Uniform(minus_one, self.one).log_prob(min_x)
        # max_x_log_prob = torch.distributions.Uniform(min_x, self.one).log_prob(max_x)
        # min_y_log_prob = torch.distributions.Uniform(minus_one, self.one).log_prob(min_y)
        # max_y_log_prob = torch.distributions.Uniform(min_y, self.one).log_prob(max_y)
        # return min_x_log_prob + max_x_log_prob + min_y_log_prob + max_y_log_prob


class TrueGenerativeModel(nn.Module):
    def __init__(self, im_size=64):
        super().__init__()
        self.im_size = im_size
        self.register_buffer("blank_canvas", torch.zeros((self.im_size, self.im_size)))

    @property
    def device(self):
        return self.blank_canvas.device

    def get_heart_obs_dist(self, heart_pose, num_rows=None, num_cols=None):
        """p_H(obs | heart_pose)

        Args
            heart_pose
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
        raw_position, raw_scale = heart_pose
        position = raw_position.sigmoid() - 0.5
        scale = raw_scale.sigmoid() * 0.8 + 0.1

        # Create blank canvas
        shape = scale.shape
        blank_canvas = torch.zeros((*shape, num_rows, num_cols), device=self.device)

        return torch.distributions.Independent(
            torch.distributions.Bernoulli(
                probs=render.render_heart((position, scale), blank_canvas).clamp(1e-6, 1 - 1e-6)
            ),
            reinterpreted_batch_ndims=2,
        )

    def get_rectangle_obs_dist(self, rectangle_pose, num_rows=None, num_cols=None):
        """p_H(obs | rectangle_pose)

        Args
            rectangle_pose [*shape, 4]
            num_rows (int)
            num_cols (int)

        Returns: distribution with batch_shape [*shape] and event_shape
            [num_rows, num_cols]
        """
        if num_rows is None:
            num_rows = self.im_size
        if num_cols is None:
            num_cols = self.im_size

        # Create blank canvas
        shape = rectangle_pose.shape[:-1]
        blank_canvas = torch.zeros((*shape, num_rows, num_cols), device=self.device)

        return torch.distributions.Independent(
            torch.distributions.Bernoulli(
                probs=render.render_rectangle(rectangle_pose, blank_canvas).clamp(1e-6, 1 - 1e-6)
            ),
            reinterpreted_batch_ndims=2,
        )

    @property
    def rectangle_pose_dist(self):
        """p_R(z_R)

        Returns distribution with batch_shape [] and event_shape [4]
        """
        return RectanglePoseDistribution(self.device)

    @property
    def heart_pose_dist(self):
        """p_H(z_H)

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

        return JointDistribution([raw_position_dist, raw_scale_dist])

    @property
    def is_heart_dist(self):
        """p_I(I)"""
        return torch.distributions.Bernoulli(probs=torch.ones((), device=self.device) * 0.5)

    def sample(self, sample_shape=[]):
        """
        Args:
            sample_shape: list-like object (default [])

        Returns:
            latent
                is_heart [*sample_shape]
                heart_pose
                    raw_position [*sample_shape, 2]
                    raw_scale [*sample_shape]
                rectangle_pose [*sample_shape, 4]
            obs [*sample_shape, im_size, im_size]
        """
        # Sample LATENT
        is_heart, heart_pose, rectangle_pose = JointDistribution(
            [self.is_heart_dist, self.heart_pose_dist, self.rectangle_pose_dist]
        ).sample(sample_shape)

        # Sample OBS
        # [*sample_shape, im_size, im_size]
        heart_obs = self.get_heart_obs_dist(heart_pose, self.im_size, self.im_size).sample()
        rectangle_obs = self.get_rectangle_obs_dist(
            rectangle_pose, self.im_size, self.im_size
        ).sample()

        # Select OBS
        # [*sample_shape]
        obs = torch.gather(
            torch.stack([rectangle_obs, heart_obs]),  # [2, *sample_shape, im_size, im_size]
            dim=0,
            index=is_heart.long()[None, ..., None, None].expand(
                *[1, *sample_shape, self.im_size, self.im_size]
            ),  # [1, *sample_shape, im_size, im_size]
        )[0]

        return (is_heart, heart_pose, rectangle_pose), obs


class GenerativeModel(nn.Module):
    def __init__(self, im_size=64):
        super().__init__()
        self.im_size = im_size
        self.mlp = util.init_mlp(2, 1, 100, 3, non_linearity=nn.ReLU())

    @property
    def device(self):
        return next(self.mlp.parameters()).device

    def get_heart_obs_dist(self, heart_pose, num_rows=None, num_cols=None):
        """p_H(obs | heart_pose)

        Args
            heart_pose
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
        raw_position, raw_scale = heart_pose
        position = raw_position.sigmoid() - 0.5
        scale = raw_scale.sigmoid() * 0.8 + 0.1

        # util.logging.info(f"position = {position} | scale = {scale}")
        shape = scale.shape
        # num_samples = int(torch.tensor(shape).prod().item())
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

    def get_rectangle_obs_dist(self, rectangle_pose, num_rows=None, num_cols=None):
        """p_H(obs | rectangle_pose)

        Args
            rectangle_pose [*shape, 4]
            num_rows (int)
            num_cols (int)

        Returns: distribution with batch_shape [*shape] and event_shape
            [num_rows, num_cols]
        """
        if num_rows is None:
            num_rows = self.im_size
        if num_cols is None:
            num_cols = self.im_size

        # Create blank canvas
        shape = rectangle_pose.shape[:-1]
        blank_canvas = torch.zeros((*shape, num_rows, num_cols), device=self.device)

        return torch.distributions.Independent(
            torch.distributions.Bernoulli(
                probs=render.render_rectangle(rectangle_pose, blank_canvas).clamp(1e-6, 1 - 1e-6)
            ),
            reinterpreted_batch_ndims=2,
        )

    @property
    def rectangle_pose_dist(self):
        """p_R(z_R)

        Returns distribution with batch_shape [] and event_shape [4]
        """
        return RectanglePoseDistribution(self.device)

    @property
    def heart_pose_dist(self):
        """p_H(z_H)

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

        return JointDistribution([raw_position_dist, raw_scale_dist])

    @property
    def is_heart_dist(self):
        """p_I(I)"""
        return torch.distributions.Bernoulli(probs=torch.ones((), device=self.device) * 0.5)

    def log_prob(self, latent, obs):
        """
        Args:
            latent
                is_heart [*shape]
                heart_pose
                    raw_position [*shape, 2]
                    raw_scale [*shape]
                rectangle_pose [*shape, 4]
            obs [*shape, num_rows, num_cols]

        Returns: [*shape]
        """
        # Deconstruct latent
        is_heart, heart_pose, rectangle_pose = latent

        # PRIOR
        # Evaluate individual log probs
        # [*shape]
        is_heart_log_prob = self.is_heart_dist.log_prob(is_heart)
        # [*shape]
        heart_pose_log_prob = self.heart_pose_dist.log_prob(heart_pose)
        # [*shape]
        rectangle_pose_log_prob = self.rectangle_pose_dist.log_prob(rectangle_pose)

        # Combine log probs
        # [*shape]
        pose_log_prob = torch.gather(
            torch.stack([rectangle_pose_log_prob, heart_pose_log_prob]),  # [2, *shape]
            dim=0,
            index=is_heart.long()[None],  # [1, *shape]
        )[0]

        latent_log_prob = is_heart_log_prob + pose_log_prob

        # LIKELIHOOD
        # Evaluate individual log probs
        # [*shape]
        heart_obs_log_prob = self.get_heart_obs_dist(
            heart_pose, self.im_size, self.im_size
        ).log_prob(obs)
        # [*shape]
        rectangle_obs_log_prob = self.get_rectangle_obs_dist(
            rectangle_pose, self.im_size, self.im_size
        ).log_prob(obs)

        # Combine log probs
        # [*shape]
        obs_log_prob = torch.gather(
            torch.stack([rectangle_obs_log_prob, heart_obs_log_prob]),  # [2, *shape]
            dim=0,
            index=is_heart.long()[None],  # [1, *shape]
        )[0]

        result = latent_log_prob + obs_log_prob

        if torch.isnan(result).any():
            raise RuntimeError("nan")

        if torch.isinf(result).any():
            raise RuntimeError("nan")

        return result

    def sample(self, sample_shape=[]):
        """
        Args:
            sample_shape: list-like object (default [])

        Returns:
            latent
                is_heart [*sample_shape]
                heart_pose
                    raw_position [*sample_shape, 2]
                    raw_scale [*sample_shape]
                rectangle_pose [*sample_shape, 4]
            obs [*sample_shape, im_size, im_size]
        """
        # Sample LATENT
        is_heart, heart_pose, rectangle_pose = JointDistribution(
            [self.is_heart_dist, self.heart_pose_dist, self.rectangle_pose_dist]
        ).sample(sample_shape)

        # Sample OBS
        # [*sample_shape, im_size, im_size]
        heart_obs = self.get_heart_obs_dist(heart_pose, self.im_size, self.im_size).sample()
        rectangle_obs = self.get_rectangle_obs_dist(
            rectangle_pose, self.im_size, self.im_size
        ).sample()

        # Select OBS
        # [*sample_shape, im_size, im_size]
        obs = torch.gather(
            torch.stack([rectangle_obs, heart_obs]),  # [2, *sample_shape, im_size, im_size]
            dim=0,
            index=is_heart.long()[None, ..., None, None].expand(
                *[1, *sample_shape, self.im_size, self.im_size]
            ),  # [1, *sample_shape, im_size, im_size]
        )[0]

        return (is_heart, heart_pose, rectangle_pose), obs

    def get_obs_probs(self, latent):
        """
        Args:
            latent
                is_heart [*shape]
                heart_pose
                    raw_position [*shape, 2]
                    raw_scale [*shape]
                rectangle_pose [*shape, 4]

        Returns:
            obs_probs [*shape, im_size, im_size]
        """
        is_heart, heart_pose, rectangle_pose = latent
        shape = is_heart.shape

        # Sample OBS
        # [*shape, im_size, im_size]
        heart_obs_probs = self.get_heart_obs_dist(
            heart_pose, self.im_size, self.im_size
        ).base_dist.probs
        rectangle_obs_probs = self.get_rectangle_obs_dist(
            rectangle_pose, self.im_size, self.im_size
        ).base_dist.probs

        # Select OBS
        # [*shape, im_size, im_size]
        obs_probs = torch.gather(
            torch.stack([rectangle_obs_probs, heart_obs_probs]),  # [2, *shape, im_size, im_size]
            dim=0,
            index=is_heart.long()[None, ..., None, None].expand(
                *[1, *shape, self.im_size, self.im_size]
            ),  # [1, *shape, im_size, im_size]
        )[0]

        return obs_probs


class Guide(nn.Module):
    def __init__(self, im_size=64):
        super().__init__()
        self.im_size = im_size
        self.rectangle_cnn = util.init_cnn(output_dim=16)
        self.heart_cnn = util.init_cnn(output_dim=16)
        self.is_heart_cnn = util.init_cnn(output_dim=16)
        self.cnn_features_dim = 400  # computed manually

        # Object id MLP
        self.is_heart_mlp = util.init_mlp(self.cnn_features_dim, 1, hidden_dim=100, num_layers=1)

        # Heart pose MLP
        self.raw_position_mlp = util.init_mlp(
            self.cnn_features_dim, 2 * 2, hidden_dim=100, num_layers=1
        )
        self.raw_scale_mlp = util.init_mlp(self.cnn_features_dim, 2, hidden_dim=100, num_layers=1)

        # Rectangle pose MLP
        self.rectangle_mlp = util.init_mlp(
            self.cnn_features_dim, 4 * 2, hidden_dim=100, num_layers=1
        )

    @property
    def device(self):
        return next(self.rectangle_cnn.parameters()).device

    def get_cnn_features(self, obs, cnn_type):
        """
        Args:
            obs: [batch_size, im_size, im_size]
            cnn_type

        Returns: [batch_size, cnn_features_dim]
        """
        batch_size = obs.shape[0]
        if cnn_type == "heart":
            cnn = self.heart_cnn
        elif cnn_type == "rectangle":
            cnn = self.rectangle_cnn
        elif cnn_type == "is_heart":
            cnn = self.is_heart_cnn
        return cnn(obs[:, None]).view(batch_size, -1)

    def get_is_heart_dist(self, obs):
        """q_I(z_I | x)

        Args:
            obs: [*shape, im_size, im_size]

        Returns: dist with batch_shape [*shape] and event_shape []
        """
        shape = obs.shape[:-2]
        num_samples = int(torch.tensor(shape).prod().item())

        # [num_samples, cnn_features_dim]
        cnn_features = self.get_cnn_features(
            obs.view(num_samples, self.im_size, self.im_size), "is_heart"
        )

        logits = self.is_heart_mlp(cnn_features).view(*shape)

        if torch.isnan(logits).any():
            raise RuntimeError("nan")

        return torch.distributions.Bernoulli(logits=logits)

    def get_heart_pose_dist(self, obs):
        """q_H(z_H | x)

        Args:
            obs: [*shape, im_size, im_size]

        Returns: dist with batch_shape [*shape] and event_shape ([2], [])
        """
        shape = obs.shape[:-2]
        num_samples = int(torch.tensor(shape).prod().item())

        # [num_samples, cnn_features_dim]
        cnn_features = self.get_cnn_features(
            obs.view(num_samples, self.im_size, self.im_size), "heart"
        )

        # Position dist
        position_raw_loc, position_raw_scale = self.raw_position_mlp(cnn_features).chunk(2, dim=-1)
        position_loc, position_scale = (
            position_raw_loc.view(*[*shape, 2]),
            position_raw_scale.exp().view(*[*shape, 2]),
        )
        raw_position_dist = torch.distributions.Independent(
            torch.distributions.Normal(position_loc, position_scale), reinterpreted_batch_ndims=1
        )

        # Scale dist
        scale_raw_loc, scale_raw_scale = self.raw_scale_mlp(cnn_features).chunk(2, dim=-1)
        scale_loc, scale_scale = scale_raw_loc.view(*shape), scale_raw_scale.view(*shape).exp()
        raw_scale_dist = torch.distributions.Normal(scale_loc, scale_scale)

        return JointDistribution([raw_position_dist, raw_scale_dist])

    def get_rectangle_pose_dist(self, obs):
        """q_R(z_R | x)

        Args:
            obs: [*shape, im_size, im_size]

        Returns: dist with batch_shape [*shape] and event_shape [4]
        """
        shape = obs.shape[:-2]
        num_samples = int(torch.tensor(shape).prod().item())

        # [num_samples, cnn_features_dim]
        cnn_features = self.get_cnn_features(
            obs.view(num_samples, self.im_size, self.im_size), "rectangle"
        )

        raw_loc, raw_scale = self.rectangle_mlp(cnn_features).chunk(2, dim=-1)
        loc, scale = raw_loc.view(*[*shape, 4]), raw_scale.exp().view(*[*shape, 4])
        return torch.distributions.Independent(
            torch.distributions.Normal(loc, scale), reinterpreted_batch_ndims=1
        )

    def get_dist(self, obs):
        """
        Args:
            obs: [batch_size, im_size, im_size]

        Returns: dist with batch_shape [batch_size] and event_shape ([], ([2], []), [4])
        """
        is_heart_dist = self.get_is_heart_dist(obs)
        heart_pose_dist = self.get_heart_pose_dist(obs)
        rectangle_pose_dist = self.get_rectangle_pose_dist(obs)

        return JointDistribution([is_heart_dist, heart_pose_dist, rectangle_pose_dist])

    def log_prob(self, obs, latent):
        """
        Args:
            obs [*shape, im_size, im_size]
            latent
                is_heart [*shape]
                heart_pose
                    raw_position [*shape, 2]
                    raw_scale [*shape]
                rectangle_pose [*shape, 4]

        Returns: [*shape]
        """
        # Deconstruct latent
        is_heart, heart_pose, rectangle_pose = latent

        # Compute individual log probs
        # [*shape]
        is_heart_log_prob = self.get_is_heart_dist(obs).log_prob(is_heart)
        heart_pose_log_prob = self.get_heart_pose_dist(obs).log_prob(heart_pose)
        rectangle_pose_log_prob = self.get_rectangle_pose_dist(obs).log_prob(rectangle_pose)

        # Combine log probs
        result = (
            is_heart_log_prob
            + torch.gather(
                torch.stack([rectangle_pose_log_prob, heart_pose_log_prob]),  # [2, *shape]
                dim=0,
                index=is_heart.long()[None],  # [1, *shape]
            )[0]
        )

        if torch.isnan(result).any():
            raise RuntimeError("nan")

        if torch.isinf(result).any():
            raise RuntimeError("nan")

        return result

    def sample(self, obs, sample_shape=[]):
        """
        Args:
            obs [*shape, im_size, im_size]

        Returns:
            latent
                is_heart [*sample_shape, *shape]
                heart_pose
                    raw_position [*sample_shape, *shape, 2]
                    raw_scale [*sample_shape, *shape]
                rectangle_pose [*sample_shape, *shape, 4]

        """
        return (
            self.get_is_heart_dist(obs).sample(sample_shape),
            self.get_heart_pose_dist(obs).sample(sample_shape),
            self.get_rectangle_pose_dist(obs).sample(sample_shape),
        )
