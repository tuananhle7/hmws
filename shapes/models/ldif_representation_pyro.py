import pyro
import util
import render
import torch
import torch.nn as nn


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
        mlp_input = self.get_mlp_input(shape_id, shape_pose, num_rows, num_cols)

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
        mlp_input = self.get_mlp_input(shape_id, shape_pose, num_rows, num_cols)

        # Run MLP
        return self.mlps[shape_id](mlp_input).view(*[*shape, num_rows, num_cols])

    def get_mlp_input(self, shape_id, shape_pose, num_rows=None, num_cols=None):
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
        mlp_input = self.get_mlp_input(shape_id, shape_pose, num_rows, num_cols)

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

    def get_shape_obs_dist_pyro(self, shape_id, shape_pose, num_rows=None, num_cols=None):
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
        mlp_input = self.get_mlp_input(shape_id, shape_pose, num_rows, num_cols)

        # Run MLP
        deep_shape_density = self.mlps[shape_id](mlp_input).view(*[*shape, num_rows, num_cols])
        analytic_shape_density = self.get_analytic_shape_density(shape_id, mlp_input).view(
            *[*shape, num_rows, num_cols]
        )
        logits = analytic_shape_density * (1 + deep_shape_density)

        if torch.isnan(logits).any():
            raise RuntimeError("nan")

        return pyro.distributions.Independent(
            pyro.distributions.Bernoulli(logits=logits), reinterpreted_batch_ndims=2
        )

    def sample_shape_pose(self, tag):
        # Position distribution
        raw_position = pyro.sample(
            f"{tag}_raw_position",
            pyro.distributions.Independent(
                pyro.distributions.Normal(
                    torch.zeros((2,), device=self.device), torch.ones((2,), device=self.device)
                ),
                reinterpreted_batch_ndims=1,
            ),
        )

        # Scale distribution
        raw_scale = pyro.sample(
            f"{tag}_raw_scale",
            pyro.distributions.Normal(
                torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device)
            ),
        )

        return raw_position, raw_scale

    def forward(self, obs):
        """
        Args:
            obs [batch_size, num_rows, num_cols]
        """
        pyro.module("generative_model", self)
        batch_size, num_rows, num_cols = obs.shape
        for batch_id in pyro.plate("batch", batch_size):
            shape_id = pyro.sample(
                f"shape_id_{batch_id}",
                pyro.distributions.Bernoulli(probs=torch.ones((), device=self.device) * 0.5),
            ).long()
            shape_pose = self.sample_shape_pose(f"shape_pose_{batch_id}")
            pyro.sample(
                f"obs_{batch_id}",
                self.get_shape_obs_dist_pyro(shape_id, shape_pose),
                obs=obs[batch_id],
            )

    def get_obs_probs(self, latent):
        """FOR PLOTTING

        Args:
            latent
                shape_id [*shape]
                shape_pose
                    raw_position [*shape, 2]
                    raw_scale [*shape]
        Returns:
            obs_probs [*shape, im_size, im_size]
        """
        shape_id, shape_pose = latent
        shape = shape_id.shape

        # Sample OBS
        # [*shape, im_size, im_size]
        shape_1_obs_probs = self.get_shape_obs_dist(
            0, shape_pose, self.im_size, self.im_size
        ).base_dist.probs
        shape_2_obs_probs = self.get_shape_obs_dist(
            1, shape_pose, self.im_size, self.im_size
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

    def forward(self, obs):
        """
        Args:
            obs [batch_size, num_rows, num_cols]

        Returns:
            raw_position [batch_size, 2]
            raw_scale [batch_size]
        """
        pyro.module("guide", self)
        batch_size, num_rows, num_cols = obs.shape

        # Get cnn features
        cnn_features = self.get_cnn_features(obs)

        # Get shape_id logits
        # [batch_size]
        logits = self.shape_id_mlp(cnn_features).view(-1)

        shape_id, raw_position, raw_scale = [], [], []
        for batch_id in pyro.plate("batch", batch_size):
            # Shape id
            shape_id.append(
                pyro.sample(
                    f"shape_id_{batch_id}", pyro.distributions.Bernoulli(logits=logits[batch_id]),
                ).long()
            )

            # Raw position
            position_raw_loc, position_raw_scale = self.raw_position_mlps[shape_id[-1]](
                cnn_features[batch_id][None]
            ).chunk(2, dim=-1)
            position_loc, position_scale = (
                position_raw_loc.view(-1),
                position_raw_scale.exp().view(-1),
            )
            raw_position.append(
                pyro.sample(
                    f"shape_pose_{batch_id}_raw_position",
                    pyro.distributions.Independent(
                        pyro.distributions.Normal(position_loc, position_scale),
                        reinterpreted_batch_ndims=1,
                    ),
                )
            )

            # Raw scale
            scale_raw_loc, scale_raw_scale = self.raw_scale_mlps[shape_id[-1]](
                cnn_features[batch_id][None]
            ).chunk(2, dim=-1)
            scale_loc, scale_scale = scale_raw_loc[0, 0], scale_raw_scale.exp()[0, 0]
            raw_scale.append(
                pyro.sample(
                    f"shape_pose_{batch_id}_raw_scale",
                    pyro.distributions.Normal(scale_loc, scale_scale),
                )
            )
        return torch.stack(shape_id), (torch.stack(raw_position), torch.stack(raw_scale))
