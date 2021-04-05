import itertools
import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import rendering
import util


def get_arcs(start_point, primitives, ids):
    """
    Args:
        start_point: tensor [batch_size, 2]
        primitives: tensor [num_primitives, 5]
            primitives[i] = [dx, dy, theta, sharpness, width]
        ids: long tensor [batch_size, num_arcs]

    Returns:
        arcs: tensor [batch_size, num_arcs, 7]
            arcs[i] = [x_start, y_start, dx, dy, theta, sharpness, width]
    """
    arcs = primitives[ids]  # [batch_size, num_arcs, 5]
    dxdy = arcs[..., :2]  # [batch_size, num_arcs, 5]
    start_and_dxdy = torch.cat(
        [start_point.unsqueeze(-2), dxdy], dim=-2
    )  # [batch_size, num_arcs + 1, 2]
    # [batch_size, num_arcs, 2]
    start_points = torch.cumsum(start_and_dxdy, dim=-2)[:, :-1]
    arcs = torch.cat([start_points, arcs], dim=-1)
    return arcs


def get_image_probs(ids, on_offs, start_point, primitives, rendering_params, num_rows, num_cols):
    """
    Args:
        ids: tensor [batch_size, num_arcs]
        on_offs: tensor [batch_size, num_arcs]
        start_point: tensor [batch_size, 2]
        primitives: tensor [num_primitives, 5]
            primitives[i] = [dx, dy, theta, sharpness, width]
        rendering_params: tensor [2]
            rendering_params[0] = scale of value
            rendering_params[1] = bias of value
        num_rows: int
        num_cols: int
    Returns:
        probs: tensor [batch_size, num_rows, num_cols]
    """
    arcs = get_arcs(start_point, primitives, ids)
    probs = rendering.get_probs(arcs, on_offs, rendering_params, num_rows, num_cols)
    return probs


class GenerativeModelIdsAndOnOffsDistribution(torch.distributions.Distribution):
    """Distribution on ids and on_offs.

    Args:
        lstm_cell: LSTMCell
        linear: Linear
        batch_size: int
        num_arcs: int
        uniform_mixture: bool or float

    Returns: distribution object with batch_shape [] and
        event_shape [num_arcs, 5]
    """

    def __init__(self, lstm_cell, linear, num_arcs, uniform_mixture):
        super().__init__()
        self.lstm_cell = lstm_cell
        self.linear = linear
        self.num_arcs = num_arcs
        self.uniform_mixture = uniform_mixture

        self.lstm_hidden_size = self.lstm_cell.hidden_size
        self.lstm_input_size = self.lstm_cell.input_size
        self.num_primitives = self.lstm_input_size - 2

    @property
    def device(self):
        return next(self.lstm_cell.parameters()).device

    def sample(self, sample_shape=torch.Size()):
        """
        Args:
            sample_shape: torch.Size

        Returns:
            ids_and_on_offs: [*sample_shape, num_arcs, 2]
                ids_and_on_offs[..., 0] are ids
                ids_and_on_offs[..., 1] are on_offs
        """
        num_samples = torch.tensor(sample_shape).long().prod()

        lstm_input = torch.zeros((*sample_shape, self.lstm_input_size), device=self.device).view(
            -1, self.lstm_input_size
        )
        h = torch.zeros((num_samples, self.lstm_hidden_size), device=self.device)
        c = torch.zeros((num_samples, self.lstm_hidden_size), device=self.device)

        ids = []
        on_offs = []
        for arc_id in range(self.num_arcs):
            h, c = self.lstm_cell(lstm_input, (h, c))
            logits = self.linear(h)
            if self.uniform_mixture:
                p = self.uniform_mixture if type(self.uniform_mixture) is float else 0.2
                id_logits = util.logit_uniform_mixture(logits[:, : self.num_primitives], p)
                on_off_logits = util.logit_uniform_mixture(logits[:, self.num_primitives :], p)
            else:
                id_logits = logits[:, : self.num_primitives]
                on_off_logits = logits[:, self.num_primitives :]
            id_dist = torch.distributions.Categorical(logits=id_logits)
            on_off_dist = torch.distributions.Categorical(logits=on_off_logits)

            ids.append(id_dist.sample())
            if arc_id == 0:
                on_off = torch.zeros(num_samples, device=self.device,).long()
            else:
                on_off = on_off_dist.sample()
            on_offs.append(on_off)

            lstm_input = torch.cat(
                [
                    F.one_hot(ids[-1], num_classes=self.num_primitives).float(),
                    F.one_hot(on_offs[-1], num_classes=2).float(),
                ],
                dim=1,
            )
        return torch.stack([torch.stack(ids, dim=1), torch.stack(on_offs, dim=1)], dim=-1).view(
            *sample_shape, *self.batch_shape, self.num_arcs, 2
        )

    def log_prob(self, ids_and_on_offs):
        """
        Args:
            ids_and_on_offs: [*sample_shape, num_arcs, 2] or
                ids_and_on_offs[..., 0] are ids
                ids_and_on_offs[..., 1] are on_offs

        Returns: tensor [*sample_shape]
        """
        sample_shape = ids_and_on_offs.shape[:-2]
        num_samples = torch.tensor(sample_shape).prod().long().item()
        lstm_input = torch.zeros((num_samples, self.lstm_input_size), device=self.device)

        h = torch.zeros((num_samples, self.lstm_hidden_size), device=self.device)
        c = torch.zeros((num_samples, self.lstm_hidden_size), device=self.device)
        ids = ids_and_on_offs[..., 0]
        on_offs = ids_and_on_offs[..., 1]

        result = 0
        for arc_id in range(self.num_arcs):
            h, c = self.lstm_cell(lstm_input, (h, c))
            logits = self.linear(h)
            if self.uniform_mixture:
                p = self.uniform_mixture if type(self.uniform_mixture) is float else 0.2
                id_logits = util.logit_uniform_mixture(logits[:, : self.num_primitives], p)
                on_off_logits = util.logit_uniform_mixture(logits[:, self.num_primitives :], p)
            else:
                id_logits = logits[:, : self.num_primitives]
                on_off_logits = logits[:, self.num_primitives :]
            id_dist = torch.distributions.Categorical(
                logits=id_logits.view(*sample_shape, *self.batch_shape, self.num_primitives)
            )
            on_off_dist = torch.distributions.Categorical(
                logits=on_off_logits.view(*sample_shape, *self.batch_shape, 2)
            )

            result += id_dist.log_prob(ids[..., arc_id])
            if arc_id == 0:
                pass
            else:
                result += on_off_dist.log_prob(on_offs[..., arc_id])

            lstm_input = torch.cat(
                [
                    F.one_hot(
                        ids[..., arc_id].reshape(-1), num_classes=self.num_primitives
                    ).float(),
                    F.one_hot(on_offs[..., arc_id].reshape(-1), num_classes=2).float(),
                ],
                dim=1,
            )
        return result


@functools.lru_cache(maxsize=None)
def get_rotation_matrix(angle):
    angle = -angle
    return torch.tensor(
        [[math.cos(angle), math.sin(angle), 0], [-math.sin(angle), math.cos(angle), 0]]
    ).float()


@functools.lru_cache(maxsize=None)
def get_translation_matrix(x, y):
    return torch.tensor([[1, 0, -x], [0, 1, y]]).float()


@functools.lru_cache(maxsize=None)
def get_shear_matrix(horizontal_angle, vertical_angle):
    return torch.tensor(
        [[1, math.tan(horizontal_angle), 0], [math.tan(vertical_angle), 1, 0]]
    ).float()


@functools.lru_cache(maxsize=None)
def get_scale_matrix(horizontal_squeeze, vertical_squeeze):
    return torch.tensor([[horizontal_squeeze, 0, 0], [0, vertical_squeeze, 0]]).float()


@functools.lru_cache(maxsize=None)
def compose2(theta_1, theta_2):
    return torch.mm(theta_2, torch.cat([theta_1, torch.tensor([[0, 0, 1]]).float()]))


@functools.lru_cache(maxsize=None)
def compose(*thetas):
    result = compose2(thetas[0], thetas[1])
    for i in range(2, len(thetas)):
        result = compose2(result, thetas[i])
    return result


@functools.lru_cache(maxsize=None)
def get_thetas(device):
    num_discretizations = 3
    shear_horizontal_angles = torch.linspace(-0.1 * math.pi, 0.1 * math.pi, num_discretizations)
    shear_vertical_angles = shear_horizontal_angles
    rotate_angles = shear_horizontal_angles
    scale_horizontal_squeezes = torch.linspace(0.9, 1.1, num_discretizations)
    scale_vertical_squeezes = scale_horizontal_squeezes
    translate_xs = torch.linspace(-0.2, 0.2, num_discretizations)
    translate_ys = translate_xs

    transform_paramss = itertools.product(
        shear_horizontal_angles,
        shear_vertical_angles,
        rotate_angles,
        scale_horizontal_squeezes,
        scale_vertical_squeezes,
        translate_xs,
        translate_ys,
    )
    result = []
    for (
        shear_horizontal_angle,
        shear_vertical_angle,
        rotate_angle,
        scale_horizontal_squeeze,
        scale_vertical_squeeze,
        translate_x,
        translate_y,
    ) in transform_paramss:
        shear_matrix = get_shear_matrix(shear_horizontal_angle, shear_vertical_angle)
        rotate_matrix = get_rotation_matrix(rotate_angle)
        scale_matrix = get_scale_matrix(scale_horizontal_squeeze, scale_vertical_squeeze)
        translate_matrix = get_translation_matrix(translate_x, translate_y)
        result.append(compose(shear_matrix, rotate_matrix, scale_matrix, translate_matrix))
    return torch.stack(result).to(device)


class AffineLikelihood(torch.distributions.Distribution):
    def __init__(self, cond, thetas=None):
        """
        Args:
            thetas: tensor [num_thetas, 2, 3]
            cond: tensor of probs [batch_size, 28, 28] in [0, 1]

        Returns: distribution object with batch_shape [batch_size] and
            event_shape [28, 28]
        """
        super().__init__()
        if thetas is None:
            self.thetas = get_thetas(cond.device)
        else:
            self.thetas = thetas
        self.num_thetas = len(self.thetas)
        self.batch_size = len(cond)
        self.cond_expanded = cond[None].expand(self.num_thetas, self.batch_size, 28, 28)
        self.grid = F.affine_grid(self.thetas, self.cond_expanded.size(), align_corners=True)

        # num_thetas, batch_size, 28, 28
        self.cond_transformed = F.grid_sample(self.cond_expanded, self.grid, align_corners=True)
        self.dist = torch.distributions.Independent(
            torch.distributions.Bernoulli(probs=self.cond_transformed), reinterpreted_batch_ndims=2
        )

    def log_prob(self, obs):
        """
        Args:
            obs: tensor [batch_size, 28, 28] in {0, 1}

        Returns: tensor [batch_size]
        """
        return torch.logsumexp(self.dist.log_prob(obs), dim=0) - math.log(self.num_thetas)

    def sample(self, sample_shape=torch.Size()):
        """
        Args:
            sample_shape: torch.Size

        Returns: [*sample_shape, batch_size, 28, 28]
        """
        mixture_ids = torch.randint(self.num_thetas, sample_shape)
        probs = self.cond_transformed[mixture_ids.view(-1)].view(
            *sample_shape, self.batch_size, 28, 28
        )
        return torch.distributions.Bernoulli(probs=probs).sample()


class GenerativeModel(nn.Module):
    def __init__(
        self,
        num_primitives,
        initial_max_curve,
        big_arcs,
        lstm_hidden_size,
        num_rows,
        num_cols,
        num_arcs,
        likelihood="bernoulli",
        uniform_mixture=False,
    ):
        super(GenerativeModel, self).__init__()
        self._prior = nn.Module()
        self._likelihood = nn.Module()

        self.lstm_hidden_size = lstm_hidden_size
        self.big_arcs = big_arcs

        lstm_input_size = num_primitives + 2

        self.num_primitives = num_primitives
        # raw_primitives[i] = [raw_dx, raw_dy, raw_theta, raw_sharpness,
        #                      raw_width]
        # constraints:
        #   dx in [-0.66, 0.66]
        #   dy in [-0.66, 0.66]
        #   theta in [-pi, pi]
        #   sharpness in [-inf, inf]
        #   width in [0, 0.1]
        # https://github.com/insperatum/wsvae/blob/master/examples/mnist/mnist.py#L775
        raw_dxdy_min = -0.4
        raw_dxdy_max = 0.4
        # self._likelihood.raw_dxdy = nn.Parameter(torch.randn((num_primitives, 2)) * 0.25)
        self._likelihood.raw_dxdy = nn.Parameter(
            torch.rand((num_primitives, 2)) * (raw_dxdy_max - raw_dxdy_min) + raw_dxdy_min
        )

        # https://github.com/insperatum/wsvae/blob/master/examples/mnist/mnist.py#L769
        # self._likelihood.raw_theta = nn.Parameter(torch.randn((num_primitives,)) *
        #                               math.pi / 4.)
        # account for the -pi in get_primitives

        init_theta_min = 1 - initial_max_curve
        init_theta_max = 1 + initial_max_curve
        theta_init = (
            torch.rand((num_primitives,)) * (init_theta_max - init_theta_min) + init_theta_min
        )
        if self.big_arcs:
            # self.theta_factor = 10
            # p = theta_init / 2
            # p = 0.01 + 0.98*p
            # assert p.min()>0 and p.max()<1
            # self._likelihood.raw_theta = nn.Parameter((p.log() - (1-p).log())/self.theta_factor)
            self._likelihood.raw_theta = nn.Parameter(theta_init)
        else:
            self.theta_factor = 10
            p = theta_init - 0.5
            p = 0.01 + 0.98 * p
            assert p.min() > 0 and p.max() < 1
            self._likelihood.raw_theta = nn.Parameter((p.log() - (1 - p).log()) / self.theta_factor)

        # https://github.com/insperatum/wsvae/blob/master/examples/mnist/mnist.py#L773-L774
        self._likelihood.raw_sharpness = nn.Parameter(torch.ones(num_primitives) * 20.0)
        # modified so that exp(raw_sharpness) + 5 is approx. 20
        # self._likelihood.raw_sharpness = nn.Parameter(torch.ones(num_primitives) * 2.71)

        # https://github.com/insperatum/wsvae/blob/master/examples/mnist/mnist.py#L771-L772
        self._likelihood.raw_width = nn.Parameter(torch.ones(num_primitives) * -2.0)

        self._prior.lstm_cell = nn.LSTMCell(
            input_size=lstm_input_size, hidden_size=lstm_hidden_size
        )
        self._prior.linear = nn.Linear(lstm_hidden_size, num_primitives + 2)
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_arcs = num_arcs
        self.register_buffer("start_point", torch.tensor([0.5, 0.5]))

        # _likelihood.raw_rendering_params = [raw_scale, raw_bias]
        # constraints:
        #   scale in [0, inf]
        #   bias in [-inf, inf]
        self._likelihood.raw_rendering_params = nn.Parameter(torch.tensor([3.0, -3.0]))

        self.likelihood = likelihood
        if likelihood == "learned-affine":
            theta_affine = torch.randn(40, 2, 3) * 0.03
            theta_affine[:, 0, 0] += 1
            theta_affine[:, 1, 1] += 1
            self._likelihood.theta_affine = nn.Parameter(theta_affine)

        self.uniform_mixture = uniform_mixture

        assert set(self.parameters()) == set(self._prior.parameters()) | set(
            self._likelihood.parameters()
        )
        assert len(set(self._prior.parameters()) & set(self._likelihood.parameters())) == 0

    def get_rendering_params(self):
        scale = F.softplus(self._likelihood.raw_rendering_params[0]).unsqueeze(0)
        bias = self._likelihood.raw_rendering_params[1].unsqueeze(0)
        return torch.cat([scale, bias])

    def get_primitives(self):
        # https://github.com/insperatum/wsvae/blob/master/examples/mnist/mnist.py#L801-L804
        dxdy = torch.tanh(self._likelihood.raw_dxdy) * 0.66

        if self.big_arcs:
            # theta = (
            #     (torch.sigmoid(self._likelihood.raw_theta * self.theta_factor) - 0.5)
            #     * math.pi
            #     * 0.99
            #     * 2
            # )
            theta = torch.remainder(self._likelihood.raw_theta * math.pi, 2 * math.pi) - math.pi
        else:
            theta = (
                (torch.sigmoid(self._likelihood.raw_theta * self.theta_factor) - 0.5)
                * math.pi
                * 0.99
            )

        # https://github.com/insperatum/wsvae/blob/master/examples/mnist/mnist.py#L816-L820
        sharpness = self._likelihood.raw_sharpness
        # modified to be > 5
        # sharpness = torch.exp(self._likelihood.raw_sharpness) + 5

        # https://github.com/insperatum/wsvae/blob/master/examples/mnist/mnist.py#L808-L814
        width = torch.sigmoid(self._likelihood.raw_width) * 0.1
        # modified to be sharper
        # width = torch.sigmoid(self._likelihood.raw_width) * 0.05
        return torch.cat(
            [dxdy, theta.unsqueeze(-1), sharpness.unsqueeze(-1), width.unsqueeze(-1)], dim=1
        )

    def get_latent_dist(self):
        """p(z)

        Returns: distribution with batch shape [] and event shape [num_arcs, 2].
        """
        return GenerativeModelIdsAndOnOffsDistribution(
            self._prior.lstm_cell, self._prior.linear, self.num_arcs, self.uniform_mixture
        )

    def get_obs_params(self, latent):
        """
        Args:
            latent: tensor [batch_size, num_arcs, 2]

        Returns:
            probs: tensor [batch_size, num_rows, num_cols]
        """
        batch_size = latent.shape[0]
        ids = latent[..., 0]
        on_offs = latent[..., 1]
        start_point = self.start_point.unsqueeze(0).expand(batch_size, -1)

        arcs = get_arcs(start_point, self.get_primitives(), ids)
        return rendering.get_logits(
            arcs, on_offs, self.get_rendering_params(), self.num_rows, self.num_cols
        )

    def get_obs_dist(self, latent):
        """Likelihood p(x | z)

        Args:
            latent: tensor [batch_size, num_arcs, 2]

        Returns: distribution with batch_shape [batch_size] and event_shape
            [num_rows, num_cols]
        """
        logits = self.get_obs_params(latent)
        if self.likelihood == "learned-affine":
            return AffineLikelihood(logits.sigmoid(), self._likelihood.theta_affine)
        elif self.likelihood == "bernoulli":
            return torch.distributions.Independent(
                torch.distributions.Bernoulli(logits=logits), reinterpreted_batch_ndims=2
            )
        else:
            raise NotImplementedError

    def get_log_prob(self, latent, obs, obs_id=None, get_accuracy=False):
        """Log of joint probability.

        Args:
            latent: tensor [num_particles, batch_size, num_arcs, 2]
            obs: tensor of shape [batch_size, num_rows, num_cols]
            obs_id: tensor of shape [batch_size]

        Returns: tensor of shape [num_particles, batch_size]
        """
        num_particles, batch_size, num_arcs, _ = latent.shape
        _, num_rows, num_cols = obs.shape
        latent_log_prob = self.get_latent_dist().log_prob(latent)

        obs_dist = self.get_obs_dist(latent.view(num_particles * batch_size, num_arcs, 2))
        obs_log_prob = obs_dist.log_prob(
            obs[None]
            .expand(num_particles, batch_size, num_rows, num_cols)
            .reshape(num_particles * batch_size, num_rows, num_cols)
        ).view(num_particles, batch_size)

        if get_accuracy:
            return latent_log_prob + obs_log_prob, None
        else:
            return latent_log_prob + obs_log_prob

    def get_log_probss(self, latent, obs, obs_id):
        """Log of joint probability.

        Args:
            latent: tensor [num_particles, batch_size, num_arcs, 2]
            obs: tensor of shape [batch_size, num_rows, num_cols]
            obs_id: tensor of shape [batch_size]

        Returns: tuple of tensor of shape [num_particles, batch_size]
        """
        num_particles, batch_size, num_arcs, _ = latent.shape
        _, num_rows, num_cols = obs.shape
        latent_log_prob = self.get_latent_dist().log_prob(latent)
        obs_dist = self.get_obs_dist(latent.view(num_particles * batch_size, num_arcs, 2))
        obs_log_prob = obs_dist.log_prob(
            obs[None]
            .expand(num_particles, batch_size, num_rows, num_cols)
            .reshape(num_particles * batch_size, num_rows, num_cols)
        ).view(num_particles, batch_size)

        return latent_log_prob, obs_log_prob

    def sample_latent_and_obs(self, num_samples=1):
        """Args:
            num_samples: int

        Returns:
            latent: tensor of shape [num_samples, num_arcs, 2]
            obs: tensor of shape [num_samples, num_rows, num_cols]
        """
        latent_dist = self.get_latent_dist()
        latent = latent_dist.sample((num_samples,))
        obs_dist = self.get_obs_dist(latent)
        obs = obs_dist.sample()

        return latent, obs

    def sample_obs(self, num_samples=1):
        """Args:
            num_samples: int

        Returns:
            obs: tensor of shape [num_samples, num_rows, num_cols]
        """

        return self.sample_latent_and_obs(num_samples)[1]


class GuideIdsAndOnOffsDistribution(torch.distributions.Distribution):
    """Distribution on ids and on_offs q(latent | obs)

    Args:
        obs_embedding: tensor [batch_size, obs_embedding_dim]
        lstm_cell: LSTMCell
        linear: Linear
        num_arcs: int
        uniform_mixture: bool

    Returns: distribution object with batch_shape [batch_size]
        and event_shape [num_arcs, 5]
    """

    def __init__(self, obs_embedding, lstm_cell, linear, num_arcs, uniform_mixture=False):
        super().__init__()
        self.obs_embedding = obs_embedding
        self.lstm_cell = lstm_cell
        self.linear = linear
        self.num_arcs = num_arcs
        self.uniform_mixture = uniform_mixture

        self.batch_size = obs_embedding.shape[0]
        self.obs_embedding_dim = obs_embedding.shape[1]

        self.lstm_hidden_size = self.lstm_cell.hidden_size
        self.lstm_input_size = self.lstm_cell.input_size
        self.num_primitives = self.lstm_input_size - 2 - self.obs_embedding_dim
        self._batch_shape = [self.batch_size]
        self.num_batch = self.batch_size
        self._event_shape = [num_arcs, 2]

    @property
    def device(self):
        return next(self.lstm_cell.parameters()).device

    def sample(self, sample_shape=torch.Size()):
        """
        Args:
            sample_shape: torch.Size

        Returns:
            ids_and_on_offs: [*sample_shape, batch_size, num_arcs, 5] or
                [*sample_shape, num_particles, batch_size, num_arcs, 5]
                    ids_and_on_offs[..., 0] are ids
                    ids_and_on_offs[..., 1] are on_offs
                    ids_and_on_offs[..., 2:4] are variations on dx, dy and theta
        """
        num_samples = int(torch.tensor(sample_shape).prod().item())

        batch_shape = [self.batch_size]
        lstm_input = torch.zeros(
            (num_samples * self.batch_size, self.lstm_input_size), device=self.device
        )
        obs_embedding_expanded = (
            self.obs_embedding[None]
            .expand(num_samples, self.batch_size, self.obs_embedding_dim)
            .reshape(-1, self.obs_embedding_dim)
        )
        lstm_input[:, -self.obs_embedding_dim :] = obs_embedding_expanded
        hc = None

        ids = []
        on_offs = []
        for arc_id in range(self.num_arcs):
            hc = self.lstm_cell(lstm_input, hc)
            h, c = hc
            logits = self.linear(h)
            if self.uniform_mixture:
                p = self.uniform_mixture if type(self.uniform_mixture) is float else 0.2
                id_logits = util.logit_uniform_mixture(logits[:, : self.num_primitives], p)
                on_off_logits = util.logit_uniform_mixture(logits[:, self.num_primitives :], p)
            else:
                id_logits = logits[:, : self.num_primitives]
                on_off_logits = logits[:, self.num_primitives :]
            id_dist = torch.distributions.Categorical(logits=id_logits)
            on_off_dist = torch.distributions.Categorical(logits=on_off_logits)

            ids.append(id_dist.sample())
            if arc_id == 0:
                on_off = (
                    torch.zeros((*sample_shape, *batch_shape), device=self.device).long().view(-1)
                )
            else:
                on_off = on_off_dist.sample()
            on_offs.append(on_off)

            lstm_input = torch.cat(
                [
                    F.one_hot(ids[-1], num_classes=self.num_primitives).float(),
                    F.one_hot(on_offs[-1], num_classes=2).float(),
                    obs_embedding_expanded,
                ],
                dim=1,
            )

        return torch.stack([torch.stack(ids, dim=-1), torch.stack(on_offs, dim=-1)], dim=-1).view(
            *sample_shape, *batch_shape, self.num_arcs, 2
        )

    def log_prob(self, ids_and_on_offs):
        """
        Args:
            ids_and_on_offs: [*sample_shape, batch_size, num_arcs, 5] or
                [*sample_shape, num_particles, batch_size, num_arcs, 5]
                    ids_and_on_offs[..., 0] are ids
                    ids_and_on_offs[..., 1] are on_offs
                    ids_and_on_offs[..., 2:4] are variations on dx, dy and theta

        Returns: tensor [*sample_shape, batch_size] or [*sample_shape, num_particles, batch_size]
        """
        sample_shape = ids_and_on_offs.shape[:-3]
        ids = ids_and_on_offs[..., 0]
        on_offs = ids_and_on_offs[..., 1]
        num_samples = int(torch.tensor(sample_shape).prod().item())

        lstm_input = torch.zeros(
            (*sample_shape, *self.batch_shape, self.lstm_input_size), device=self.device
        ).view(-1, self.lstm_input_size)
        obs_embedding_expanded = (
            self.obs_embedding[None]
            .expand(num_samples, *self.batch_shape, self.obs_embedding_dim)
            .reshape(-1, self.obs_embedding_dim)
        )
        lstm_input[:, -self.obs_embedding_dim :] = obs_embedding_expanded
        hc = None

        result = 0
        for arc_id in range(self.num_arcs):
            hc = self.lstm_cell(lstm_input, hc)
            h, c = hc
            logits = self.linear(h)
            if self.uniform_mixture:
                p = self.uniform_mixture if type(self.uniform_mixture) is float else 0.2
                id_logits = util.logit_uniform_mixture(logits[:, : self.num_primitives], p)
                on_off_logits = util.logit_uniform_mixture(logits[:, self.num_primitives :], p)
            else:
                id_logits = logits[:, : self.num_primitives]
                on_off_logits = logits[:, self.num_primitives :]
            id_dist = torch.distributions.Categorical(
                logits=id_logits.view(*sample_shape, *self.batch_shape, self.num_primitives)
            )
            on_off_dist = torch.distributions.Categorical(
                logits=on_off_logits.view(*sample_shape, *self.batch_shape, 2)
            )

            result += id_dist.log_prob(ids[..., arc_id])
            if arc_id == 0:
                pass
            else:
                result += on_off_dist.log_prob(on_offs[..., arc_id])

            lstm_input = torch.cat(
                [
                    F.one_hot(ids[..., arc_id].view(-1), num_classes=self.num_primitives).float(),
                    F.one_hot(on_offs[..., arc_id].view(-1), num_classes=2).float(),
                    obs_embedding_expanded,
                ],
                dim=1,
            )
        return result


class Guide(nn.Module):
    """q(latent | obs)"""

    def __init__(
        self,
        num_primitives,
        lstm_hidden_size,
        num_rows,
        num_cols,
        num_arcs,
        obs_embedding_dim,
        uniform_mixture=False,
    ):
        super(Guide, self).__init__()
        self.obs_embedding_dim = obs_embedding_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_input_size = num_primitives + 2 + obs_embedding_dim
        self.lstm_cell = nn.LSTMCell(
            input_size=self.lstm_input_size, hidden_size=self.lstm_hidden_size
        )
        self.obs_embedder = util.cnn(self.obs_embedding_dim)
        # TODO: consider feeding partial image into the LSTM
        # self.partial_image_embedder_cnn = nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=3, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(128, 128, kernel_size=3, padding=0),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, self.observed_obs_embedding_dim,
        #               kernel_size=3, padding=0),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        self.linear = nn.Linear(self.lstm_hidden_size, num_primitives + 2)
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_arcs = num_arcs
        self.register_buffer("start_point", torch.tensor([0.5, 0.5]))
        self.uniform_mixture = uniform_mixture

    def get_obs_embedding(self, obs):
        """
        Args:
            obs: tensor [batch_size, num_rows, num_cols]

        Returns: tensor [batch_size, obs_embedding_dim]
        """
        batch_size = obs.shape[0]
        result = self.obs_embedder(obs.unsqueeze(1)).view(batch_size, -1)
        assert result.shape[1] == self.obs_embedding_dim
        return result

    def get_latent_dist(self, obs):
        """q(z | x)

        Args:
            obs: tensor of shape [batch_size, num_rows, num_cols]

        Returns: distribution with batch_shape [batch_size]
            and event_shape [num_arcs, 2]
        """
        return GuideIdsAndOnOffsDistribution(
            self.get_obs_embedding(obs),
            self.lstm_cell,
            self.linear,
            self.num_arcs,
            self.uniform_mixture,
        )

    def sample_from_latent_dist(self, latent_dist, num_particles):
        """Samples from q(latent | obs)

        Args:
            latent_dist: distribution with batch_shape [batch_size] or
                [num_particles, batch_size] and event_shape [num_arcs, 2]
            num_particles: int

        Returns:
            latent: tensor of shape [num_particles, batch_size, num_arcs, 2]
        """
        sample_shape = [num_particles]
        return latent_dist.sample(sample_shape)

    def sample(self, obs, num_particles):
        """Samples from q(latent | obs)

        Args:
            obs: tensor of shape [batch_size, num_rows, num_cols]
            num_particles: int

        Returns:
            latent: tensor of shape [num_particles, batch_size, num_arcs, 2]
        """
        latent_dist = self.get_latent_dist(obs)
        return self.sample_from_latent_dist(latent_dist, num_particles)

    def get_log_prob_from_latent_dist(self, latent_dist, latent):
        """Log q(latent | obs).

        Args:
            latent_dist: distribution with batch_shape [batch_size] or
                [num_particles, batch_size] and event_shape [num_arcs, 2]
            latent: tensor of shape [num_particles, batch_size, num_arcs, 2]

        Returns: tensor of shape [num_particles, batch_size]
        """
        return latent_dist.log_prob(latent)

    def get_log_prob(self, latent, obs):
        """Log q(latent | obs).

        Args:
            latent: tensor of shape [num_particles, batch_size, num_arcs, 2]
            obs: tensor of shape [batch_size, num_rows, num_cols]

        Returns: tensor of shape [num_particles, batch_size]
        """
        return self.get_log_prob_from_latent_dist(self.get_latent_dist(obs), latent)
