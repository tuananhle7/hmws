import pyro
import torch
import torch.nn as nn
from cmws import util
from cmws.examples.csg import render
from cmws.examples.csg import util as csg_util


class GenerativeModel(nn.Module):
    def __init__(self, im_size=64, num_primitives=2, has_shape_scale=False):
        super().__init__()
        self.im_size = im_size
        self.num_primitives = num_primitives
        self.has_shape_scale = has_shape_scale
        self.mlps = nn.ModuleList(
            [
                util.init_mlp(1, 1, 64, 3, non_linearity=nn.ReLU())
                for _ in range(self.num_primitives)
            ]
        )
        self.logit_multipliers_raw = nn.Parameter(torch.ones((self.num_primitives,)))

    @property
    def device(self):
        return next(self.mlps[0].parameters()).device

    def get_shape_obs_logits(self, shape_id, shape_pose, num_rows=None, num_cols=None):
        """p_S(obs | shape_pose)

        Args
            shape_id (int)
            shape_pose
                raw_position [*shape, 2]
                raw_scale [*shape]
            num_rows (int)
            num_cols (int)

        Returns: [*shape, num_rows, num_cols]
        """
        if num_rows is None:
            num_rows = self.im_size
        if num_cols is None:
            num_cols = self.im_size

        # Extract
        raw_position, raw_scale = shape_pose
        position = raw_position.sigmoid() - 0.5
        scale = raw_scale.sigmoid() * 0.8 + 0.1
        shape = scale.shape

        # Get canvas
        # [*shape]
        position_x, position_y = position[..., 0], position[..., 1]
        # [num_rows, num_cols]
        canvas_x, canvas_y = render.get_canvas_xy(num_rows, num_cols, self.device)

        # Shift and scale
        # [num_samples, num_rows, num_cols]
        canvas_x = (canvas_x[None] - position_x.view(-1, 1, 1)) / scale.view(-1, 1, 1)
        canvas_y = (canvas_y[None] - position_y.view(-1, 1, 1)) / scale.view(-1, 1, 1)

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

    def forward(self, obs):
        """
        Args:
            obs [batch_size, num_rows, num_cols]
        """
        pyro.module("generative_model", self)
        batch_size, num_rows, num_cols = obs.shape
        for batch_id in pyro.plate("batch", batch_size):
            # p(shape_id)
            shape_id = pyro.sample(
                f"shape_id_{batch_id}",
                pyro.distributions.Categorical(
                    logits=torch.ones((self.num_primitives), device=self.device)
                ),
            ).long()

            # p(raw_position)
            tag = f"shape_pose_{batch_id}"
            raw_position = pyro.sample(
                f"{tag}_raw_position",
                pyro.distributions.Independent(
                    pyro.distributions.Normal(
                        torch.zeros((2,), device=self.device), torch.ones((2,), device=self.device)
                    ),
                    reinterpreted_batch_ndims=1,
                ),
            )

            # p(raw_scale)
            if self.has_shape_scale:
                raw_scale = pyro.sample(
                    f"{tag}_raw_scale",
                    pyro.distributions.Normal(
                        torch.tensor(0.0, device=self.device),
                        torch.tensor(1.0, device=self.device),
                    ),
                )
            else:
                raw_scale = torch.tensor(0.0, device=self.device)

            # p(obs | shape_id, raw_position, raw_scale)
            pyro.sample(
                f"obs_{batch_id}",
                pyro.distributions.Independent(
                    pyro.distributions.Bernoulli(
                        logits=self.get_shape_obs_logits(
                            shape_id, (raw_position, raw_scale), num_rows, num_cols
                        )
                    ),
                    reinterpreted_batch_ndims=2,
                ),
                obs=obs[batch_id],
            )


class Guide(nn.Module):
    def __init__(self, im_size=64, num_primitives=2, has_shape_scale=False):
        super().__init__()
        self.im_size = im_size
        self.num_primitives = num_primitives
        self.has_shape_scale = has_shape_scale
        self.pose_net = util.init_cnn(output_dim=16)
        self.shape_id_net = util.init_cnn(output_dim=16)
        self.cnn_features_dim = 400  # computed manually

        # Object id MLP
        self.shape_id_mlp = util.init_mlp(
            self.cnn_features_dim, self.num_primitives, hidden_dim=100, num_layers=3
        )

        # Shape pose MLPs
        self.raw_position_mlps = nn.ModuleList(
            [
                util.init_mlp(self.cnn_features_dim, 2 * 2, hidden_dim=100, num_layers=3)
                for _ in range(self.num_primitives)
            ]
        )
        self.raw_scale_mlps = nn.ModuleList(
            [
                util.init_mlp(self.cnn_features_dim, 2, hidden_dim=100, num_layers=3)
                for _ in range(self.num_primitives)
            ]
        )

    @property
    def device(self):
        return next(self.shape_id_mlp.parameters()).device

    def get_cnn_features(self, obs, pose_or_shape_id):
        """
        Args:
            obs: [batch_size, im_size, im_size]
            pose_or_shape_id: str

        Returns: [batch_size, cnn_features_dim]
        """
        batch_size = obs.shape[0]
        if pose_or_shape_id == "pose":
            cnn = self.pose_net
        elif pose_or_shape_id == "shape_id":
            cnn = self.shape_id_net
        return cnn(obs[:, None]).view(batch_size, -1)

    def forward(self, obs):
        """
        Args:
            obs [batch_size, num_rows, num_cols]

        Returns:
            shape_id [batch_size]
            raw_position [batch_size, 2]
            raw_scale [batch_size]
        """
        pyro.module("guide", self)
        batch_size, num_rows, num_cols = obs.shape

        # Get pose cnn features
        pose_cnn_features = self.get_cnn_features(obs, "pose")

        # CONTINUE HERE
        # Get shape_id logits
        # [batch_size, num_primitives]
        logits = self.shape_id_mlp(cnn_features)

        shape_id, raw_position, raw_scale = [], [], []
        for batch_id in pyro.plate("batch", batch_size):
            # q(shape_id | obs)
            shape_id.append(
                pyro.sample(
                    f"shape_id_{batch_id}", pyro.distributions.Categorical(logits=logits[batch_id]),
                ).long()
            )

            # q(raw_position | obs)
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

            # q(raw_scale | obs)
            if self.has_shape_scale:
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
            else:
                raw_scale.append(torch.tensor(0.0, device=self.device))
        return torch.stack(shape_id), (torch.stack(raw_position), torch.stack(raw_scale))
