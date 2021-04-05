import torch
import torch.nn as nn
from cmws import util
from cmws.examples.csg import render
from cmws.examples.csg import util as csg_util


def shape_pose_to_str(shape_pose, shape_id):
    """
    Args
        shape_pose
            raw_position [2]
            raw_scale []

    Returns (str)
    """
    raw_position, raw_scale = shape_pose
    position = raw_position.sigmoid() - 0.5
    scale = raw_scale.sigmoid() * 0.8 + 0.1
    return (
        f"S{int(shape_id.item())}(x={position[0].item():.1f},"
        f"y={position[0].item():.1f},s={scale.item():.1f})"
    )


def latent_to_str(latent):
    """
    Args
        latent
            shape_id []
            shape_1_pose
                raw_position [2]
                raw_scale []
            shape_2_pose
                raw_position [2]
                raw_scale []

    Returns (str)
    """
    shape_id, shape_1_pose, shape_2_pose = latent
    if int(shape_id.item()) == 0:
        return shape_pose_to_str(shape_1_pose, shape_id)
    else:
        return shape_pose_to_str(shape_2_pose, shape_id)


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
        return csg_util.RectanglePoseDistribution(self.device)

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

        return util.JointDistribution([raw_position_dist, raw_scale_dist])

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
        is_heart, heart_pose, rectangle_pose = util.JointDistribution(
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
        self.mlps = nn.ModuleList(
            [util.init_mlp(2, 1, 100, 3, non_linearity=nn.ReLU()) for _ in range(2)]
        )
        self.raw_analytic_shape_params = nn.ParameterList(
            [nn.Parameter(torch.randn((3,))) for _ in range(2)]
        )

    @property
    def device(self):
        return next(self.mlps[0].parameters()).device

    def get_analytic_shape_density(self, shape_id, points):
        """
        Args
            points [*shape, 2]

        Returns [*shape]
        """
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Bivariate_case
        x_std = self.raw_analytic_shape_params[shape_id][0].exp()
        y_std = self.raw_analytic_shape_params[shape_id][1].exp()
        xy_cov = (self.raw_analytic_shape_params[shape_id][2].sigmoid() * 2 - 1) * x_std * y_std
        covariance_matrix = torch.stack(
            [torch.stack([x_std ** 2, xy_cov]), torch.stack([xy_cov, y_std ** 2])], dim=0
        )
        return (
            torch.distributions.MultivariateNormal(
                torch.zeros((2,), device=points.device), covariance_matrix
            )
            .log_prob(points)
            .exp()
        )

    def _get_mlp_input(self, shape_id, shape_pose, num_rows=None, num_cols=None):
        """
        Args
            shape_id (int)
            shape_pose
                raw_position [*shape, 2]
                raw_scale [*shape]
            num_rows (int)
            num_cols (int)

        Returns: [num_samples * num_rows * num_cols, 2]
        """
        if num_rows is None:
            num_rows = self.im_size
        if num_cols is None:
            num_cols = self.im_size
        # Extract
        raw_position, raw_scale = shape_pose
        position = raw_position.sigmoid() - 0.5
        scale = raw_scale.sigmoid() * 0.8 + 0.1

        position_x, position_y = position[..., 0], position[..., 1]  # [*shape]
        # [num_rows, num_cols]
        canvas_x, canvas_y = render.get_canvas_xy(num_rows, num_cols, self.device)

        # Shift and scale
        # [num_samples, num_rows, num_cols]
        canvas_x = (canvas_x[None] - position_x.view(-1, 1, 1)) / scale.view(-1, 1, 1)
        canvas_y = (canvas_y[None] - position_y.view(-1, 1, 1)) / scale.view(-1, 1, 1)

        # Build input
        # [num_samples * num_rows * num_cols, 2]
        return torch.stack([canvas_x, canvas_y], dim=-1).view(-1, 2)

    def _get_analytic_shape_density(self, shape_id, shape_pose, num_rows=None, num_cols=None):
        """p_S(obs | shape_pose)

        Args
            shape_id (int)
            shape_pose
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
        _, raw_scale = shape_pose
        shape = raw_scale.shape

        # Build input
        # [num_samples * num_rows * num_cols, 2]
        mlp_input = self._get_mlp_input(shape_id, shape_pose, num_rows, num_cols)

        return self.get_analytic_shape_density(shape_id, mlp_input).view(
            *[*shape, num_rows, num_cols]
        )

    def _get_deep_shape_density(self, shape_id, shape_pose, num_rows=None, num_cols=None):
        """p_S(obs | shape_pose)

        Args
            shape_id (int)
            shape_pose
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
        _, raw_scale = shape_pose
        shape = raw_scale.shape

        # Build input
        # [num_samples * num_rows * num_cols, 2]
        mlp_input = self._get_mlp_input(shape_id, shape_pose, num_rows, num_cols)

        # Run MLP
        return self.mlps[shape_id](mlp_input).view(*[*shape, num_rows, num_cols])

    def get_shape_obs_dist(self, shape_id, shape_pose, num_rows=None, num_cols=None):
        """p_S(obs | shape_pose)

        Args
            shape_id (int)
            shape_pose
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
        _, raw_scale = shape_pose
        shape = raw_scale.shape

        # Build input
        # [num_samples * num_rows * num_cols, 2]
        mlp_input = self._get_mlp_input(shape_id, shape_pose, num_rows, num_cols)

        # Run MLP
        deep_shape_density = self.mlps[shape_id](mlp_input).view(*[*shape, num_rows, num_cols])
        analytic_shape_density = self.get_analytic_shape_density(shape_id, mlp_input).view(
            *[*shape, num_rows, num_cols]
        )
        logits = analytic_shape_density * (1 + deep_shape_density)

        if torch.isnan(logits).any():
            raise RuntimeError("nan")

        return torch.distributions.Independent(
            torch.distributions.Bernoulli(logits=logits), reinterpreted_batch_ndims=2
        )

    @property
    def shape_pose_dist(self):
        """p_S(z_S)

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

    @property
    def shape_id_dist(self):
        """p_I(I)"""
        return torch.distributions.Bernoulli(probs=torch.ones((), device=self.device) * 0.5)

    def log_prob(self, latent, obs):
        """
        Args:
            latent
                shape_id [*shape]
                shape_1
                    raw_position [*shape, 2]
                    raw_scale [*shape]
                shape_2
                    raw_position [*shape, 2]
                    raw_scale [*shape]
            obs [*shape, num_rows, num_cols]

        Returns: [*shape]
        """
        # Deconstruct latent
        shape_id, shape_1_pose, shape_2_pose = latent

        # PRIOR
        # Evaluate individual log probs
        # [*shape]
        shape_id_log_prob = self.shape_id_dist.log_prob(shape_id)
        # [*shape]
        shape_1_pose_log_prob = self.shape_pose_dist.log_prob(shape_1_pose)
        # [*shape]
        shape_2_pose_log_prob = self.shape_pose_dist.log_prob(shape_2_pose)

        # Combine log probs
        # [*shape]
        pose_log_prob = torch.gather(
            torch.stack([shape_1_pose_log_prob, shape_2_pose_log_prob]),  # [2, *shape]
            dim=0,
            index=shape_id.long()[None],  # [1, *shape]
        )[0]

        latent_log_prob = shape_id_log_prob + pose_log_prob

        # LIKELIHOOD
        # Evaluate individual log probs
        # [*shape]
        shape_1_obs_log_prob = self.get_shape_obs_dist(
            0, shape_1_pose, self.im_size, self.im_size
        ).log_prob(obs)
        # [*shape]
        shape_2_obs_log_prob = self.get_shape_obs_dist(
            1, shape_2_pose, self.im_size, self.im_size
        ).log_prob(obs)

        # Combine log probs
        # [*shape]
        obs_log_prob = torch.gather(
            torch.stack([shape_1_obs_log_prob, shape_2_obs_log_prob]),  # [2, *shape]
            dim=0,
            index=shape_id.long()[None],  # [1, *shape]
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
                shape_id [*sample_shape]
                shape_1_pose
                    raw_position [*sample_shape, 2]
                    raw_scale [*sample_shape]
                shape_2_pose
                    raw_position [*sample_shape, 2]
                    raw_scale [*sample_shape]
            obs [*sample_shape, im_size, im_size]
        """
        # Sample LATENT
        shape_id, shape_1_pose, shape_2_pose = util.JointDistribution(
            [self.shape_id_dist, self.shape_pose_dist, self.shape_pose_dist]
        ).sample(sample_shape)

        # Sample OBS
        # [*sample_shape, im_size, im_size]
        shape_1_obs = self.get_shape_obs_dist(0, shape_1_pose, self.im_size, self.im_size).sample()
        shape_2_obs = self.get_shape_obs_dist(1, shape_2_pose, self.im_size, self.im_size).sample()

        # Select OBS
        # [*sample_shape, im_size, im_size]
        obs = torch.gather(
            torch.stack([shape_1_obs, shape_2_obs]),  # [2, *sample_shape, im_size, im_size]
            dim=0,
            index=shape_id.long()[None, ..., None, None].expand(
                *[1, *sample_shape, self.im_size, self.im_size]
            ),  # [1, *sample_shape, im_size, im_size]
        )[0]

        return (shape_id, shape_1_pose, shape_2_pose), obs

    def get_obs_probs(self, latent):
        """
        Args:
            latent
                shape_id [*shape]
                shape_1_pose
                    raw_position [*shape, 2]
                    raw_scale [*shape]
                shape_2_pose
                    raw_position [*shape, 2]
                    raw_scale [*shape]

        Returns:
            obs_probs [*shape, im_size, im_size]
        """
        shape_id, shape_1_pose, shape_2_pose = latent
        shape = shape_id.shape

        # Sample OBS
        # [*shape, im_size, im_size]
        shape_1_obs_probs = self.get_shape_obs_dist(
            0, shape_1_pose, self.im_size, self.im_size
        ).base_dist.probs
        shape_2_obs_probs = self.get_shape_obs_dist(
            1, shape_2_pose, self.im_size, self.im_size
        ).base_dist.probs

        # Select OBS
        # [*shape, im_size, im_size]
        obs_probs = torch.gather(
            torch.stack([shape_1_obs_probs, shape_2_obs_probs]),  # [2, *shape, im_size, im_size]
            dim=0,
            index=shape_id.long()[None, ..., None, None].expand(
                *[1, *shape, self.im_size, self.im_size]
            ),  # [1, *shape, im_size, im_size]
        )[0]

        return obs_probs


class Guide(nn.Module):
    def __init__(self, im_size=64):
        super().__init__()
        self.im_size = im_size
        self.cnn = util.init_cnn(output_dim=16)
        self.cnn_features_dim = 400  # computed manually

        # Object id MLP
        self.shape_id_mlp = util.init_mlp(self.cnn_features_dim, 1, hidden_dim=100, num_layers=3)

        # Shape pose MLPs
        self.raw_position_mlps = nn.ModuleList(
            [
                util.init_mlp(self.cnn_features_dim, 2 * 2, hidden_dim=100, num_layers=3)
                for _ in range(2)
            ]
        )
        self.raw_scale_mlps = nn.ModuleList(
            [
                util.init_mlp(self.cnn_features_dim, 2, hidden_dim=100, num_layers=3)
                for _ in range(2)
            ]
        )

    @property
    def device(self):
        return next(self.shape_id_mlp.parameters()).device

    def get_cnn_features(self, obs):
        """
        Args:
            obs: [batch_size, im_size, im_size]

        Returns: [batch_size, cnn_features_dim]
        """
        batch_size = obs.shape[0]
        return self.cnn(obs[:, None]).view(batch_size, -1)

    def get_shape_id_dist(self, obs):
        """q_I(z_I | x)

        Args:
            obs: [*shape, im_size, im_size]

        Returns: dist with batch_shape [*shape] and event_shape []
        """
        shape = obs.shape[:-2]
        num_samples = int(torch.tensor(shape).prod().item())

        # [num_samples, cnn_features_dim]
        cnn_features = self.get_cnn_features(obs.reshape(num_samples, self.im_size, self.im_size))

        logits = self.shape_id_mlp(cnn_features).view(*shape)

        if torch.isnan(logits).any():
            raise RuntimeError("nan")

        return torch.distributions.Bernoulli(logits=logits)

    def get_shape_pose_dist(self, shape_id, obs):
        """q_S(z_S | x)

        Args:
            shape_id (int)
            obs: [*shape, im_size, im_size]

        Returns: dist with batch_shape [*shape] and event_shape ([2], [])
        """
        shape = obs.shape[:-2]
        num_samples = int(torch.tensor(shape).prod().item())

        # [num_samples, cnn_features_dim]
        cnn_features = self.get_cnn_features(obs.reshape(num_samples, self.im_size, self.im_size))

        # Position dist
        position_raw_loc, position_raw_scale = self.raw_position_mlps[shape_id](cnn_features).chunk(
            2, dim=-1
        )
        position_loc, position_scale = (
            position_raw_loc.view(*[*shape, 2]),
            position_raw_scale.exp().view(*[*shape, 2]),
        )
        raw_position_dist = torch.distributions.Independent(
            torch.distributions.Normal(position_loc, position_scale), reinterpreted_batch_ndims=1
        )

        # Scale dist
        scale_raw_loc, scale_raw_scale = self.raw_scale_mlps[shape_id](cnn_features).chunk(
            2, dim=-1
        )
        scale_loc, scale_scale = scale_raw_loc.view(*shape), scale_raw_scale.view(*shape).exp()
        raw_scale_dist = torch.distributions.Normal(scale_loc, scale_scale)

        return util.JointDistribution([raw_position_dist, raw_scale_dist])

    def log_prob(self, obs, latent):
        """
        Args:
            obs [*shape, im_size, im_size]
            latent
                shape_id [*shape]
                shape_1_pose
                    raw_position [*shape, 2]
                    raw_scale [*shape]
                shape_2_pose
                    raw_position [*shape, 2]
                    raw_scale [*shape]

        Returns: [*shape]
        """
        # Deconstruct latent
        shape_id, shape_1_pose, shape_2_pose = latent

        # Compute individual log probs
        # [*shape]
        shape_id_log_prob = self.get_shape_id_dist(obs).log_prob(shape_id)
        shape_1_pose_log_prob = self.get_shape_pose_dist(0, obs).log_prob(shape_1_pose)
        shape_2_pose_log_prob = self.get_shape_pose_dist(1, obs).log_prob(shape_2_pose)

        # Combine log probs
        result = (
            shape_id_log_prob
            + torch.gather(
                torch.stack([shape_1_pose_log_prob, shape_2_pose_log_prob]),  # [2, *shape]
                dim=0,
                index=shape_id.long()[None],  # [1, *shape]
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
                shape_id [*sample_shape, *shape]
                shape_1_pose
                    raw_position [*sample_shape, *shape, 2]
                    raw_scale [*sample_shape, *shape]
                shape_2_pose
                    raw_position [*sample_shape, *shape, 2]
                    raw_scale [*sample_shape, *shape]

        """
        return (
            self.get_shape_id_dist(obs).sample(sample_shape),
            self.get_shape_pose_dist(0, obs).sample(sample_shape),
            self.get_shape_pose_dist(1, obs).sample(sample_shape),
        )