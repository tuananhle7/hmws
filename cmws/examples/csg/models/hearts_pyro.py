import pyro
import torch
import torch.nn as nn
from cmws import util
from cmws.examples.csg import render


class GenerativeModel(nn.Module):
    def __init__(self, im_size=64):
        super().__init__()
        self.im_size = im_size
        self.occupancy_net = util.init_mlp(2, 1, 100, 3, non_linearity=nn.ReLU())

    @property
    def device(self):
        return next(self.occupancy_net.parameters()).device

    def forward(self, obs):
        """
        Args:
            obs [batch_size, num_rows, num_cols]
        """
        pyro.module("generative_model", self)
        batch_size, num_rows, num_cols = obs.shape
        for batch_id in pyro.plate("batch", batch_size):
            # Sample raw latents
            # [2]
            raw_position = pyro.sample(
                f"raw_position_{batch_id}",
                pyro.distributions.Independent(
                    pyro.distributions.Normal(
                        torch.zeros((2,), device=self.device), torch.ones((2,), device=self.device)
                    ),
                    reinterpreted_batch_ndims=1,
                ),
            )
            # []
            raw_scale = pyro.sample(
                f"raw_scale_{batch_id}",
                pyro.distributions.Normal(
                    torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device)
                ),
            )

            # Transform raw latents
            position = raw_position.sigmoid() - 0.5
            scale = raw_scale.sigmoid() * 0.8 + 0.1

            # Set up canvas
            position_x, position_y = position
            # [num_rows, num_cols]
            canvas_x, canvas_y = render.get_canvas_xy(num_rows, num_cols, self.device)

            # Shift and scale
            # [num_rows, num_cols]
            canvas_x = (canvas_x - position_x) / scale
            canvas_y = (canvas_y - position_y) / scale

            # Build input
            # [num_rows * num_cols, 2]
            occupancy_net_input = torch.stack([canvas_x, canvas_y], dim=-1).view(-1, 2)

            # Render
            logits = self.occupancy_net(occupancy_net_input).view(num_rows, num_cols)

            # Observe
            pyro.sample(
                "obs_{}".format(batch_id),
                pyro.distributions.Independent(
                    pyro.distributions.Bernoulli(logits=logits), reinterpreted_batch_ndims=2
                ),
                obs=obs[batch_id],
            )

    def get_obs_dist(self, latent, num_rows=None, num_cols=None):
        """p(obs | latent) -- ONLY FOR PLOTTING

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
        occupancy_net_input = torch.stack([canvas_x, canvas_y], dim=-1).view(-1, 2)

        # Run occupancy_net
        logits = self.occupancy_net(occupancy_net_input).view(*[*shape, num_rows, num_cols])

        if torch.isnan(logits).any():
            raise RuntimeError("nan")

        return torch.distributions.Independent(
            torch.distributions.Bernoulli(logits=logits), reinterpreted_batch_ndims=2
        )


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

    def forward(self, obs):
        """
        Args:
            obs [batch_size, num_rows, num_cols]
        """
        pyro.module("guide", self)
        batch_size, num_rows, num_cols = obs.shape

        # Get cnn features
        cnn_features = self.get_cnn_features(obs)

        # Position dist params
        raw_loc, raw_scale = self.raw_position_mlp(cnn_features).chunk(2, dim=-1)
        # [batch_size, 2], [batch_size, 2]
        raw_position_loc, raw_position_scale = raw_loc, raw_scale.exp()

        # Scale dist params
        raw_loc, raw_scale = self.raw_scale_mlp(cnn_features).chunk(2, dim=-1)
        raw_scale_loc, raw_scale_scale = raw_loc.view(-1), raw_scale.view(-1).exp()

        raw_position, raw_scale = [], []
        for batch_id in pyro.plate("batch", batch_size):
            raw_position.append(
                pyro.sample(
                    f"raw_position_{batch_id}",
                    pyro.distributions.Independent(
                        pyro.distributions.Normal(
                            raw_position_loc[batch_id], raw_position_scale[batch_id]
                        ),
                        reinterpreted_batch_ndims=1,
                    ),
                )
            )
            raw_scale.append(
                pyro.sample(
                    f"raw_scale_{batch_id}",
                    pyro.distributions.Normal(raw_scale_loc[batch_id], raw_scale_scale[batch_id]),
                )
            )

        return torch.stack(raw_position), torch.stack(raw_scale)
