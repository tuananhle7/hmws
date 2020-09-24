"""
Wraps p(z, x) and q(z | x) to create

p(z)p(c)p(x | z, c)

and

q(z | x) q(c | z, x)
"""
import torch
import torch.nn as nn
import models
import rendering
import einops


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


class GenerativeMotorNoiseDist(nn.Module):
    """p(c)

    Distribution of
    batch_shape [] event_shape [num_arcs, 3]
    """

    def __init__(self, num_arcs):
        super().__init__()
        self.num_arcs = num_arcs
        self.register_buffer("loc", torch.zeros((num_arcs, 3)))
        self.register_buffer("scale", 0.1 * torch.ones((num_arcs, 3)))

    @property
    def dist(self):
        return torch.distributions.Independent(
            torch.distributions.Normal(self.loc, self.scale), reinterpreted_batch_ndims=2
        )

    def sample(self, sample_shape=[]):
        return self.dist.sample(sample_shape)

    def log_prob(self, motor_noise):
        return self.dist.log_prob(motor_noise)


class GuideMotorNoiseDist:
    """q(c | z, x)

    Distribution of
    batch_shape [*shape] event_shape [num_arcs, 3]

    Args:
        obs: tensor of shape [*shape, num_rows, num_cols]
        num_arcs (int)

    # TODO: consider different architectures
    """

    def __init__(self, obs, num_arcs):
        shape = obs.shape[:-2]
        device = obs.device
        self.num_arcs = num_arcs
        self.loc = torch.zeros((*[*shape, num_arcs, 3]), device=device)
        self.scale = 0.1 * torch.ones((*[*shape, num_arcs, 3]), device=device)

    @property
    def dist(self):
        return torch.distributions.Independent(
            torch.distributions.Normal(self.loc, self.scale), reinterpreted_batch_ndims=2
        )

    def sample(self, sample_shape=[]):
        return self.dist.sample(sample_shape)

    def log_prob(self, motor_noise):
        return self.dist.log_prob(motor_noise)


class GenerativeModel(nn.Module):
    """p(z)p(c)p(x | z, c)
    """

    def __init__(self, ids_and_on_offs_generative_model_args):
        super().__init__()
        self.ids_and_on_offs_generative_model = models.GenerativeModel(
            *ids_and_on_offs_generative_model_args
        )
        self.num_arcs = self.ids_and_on_offs_generative_model.num_arcs
        self.num_primitives = self.ids_and_on_offs_generative_model.num_primitives
        self.generative_motor_noise_dist = GenerativeMotorNoiseDist(self.num_arcs)
        self.get_primitives = self.ids_and_on_offs_generative_model.get_primitives
        self.get_rendering_params = self.ids_and_on_offs_generative_model.get_rendering_params
        self.num_rows = self.ids_and_on_offs_generative_model.num_rows
        self.num_cols = self.ids_and_on_offs_generative_model.num_cols

    @property
    def start_point(self):
        return self.ids_and_on_offs_generative_model.start_point

    def get_latent_dist(self):
        """ p(z)p(c)

        Returns: distribution with batch shape [] and
        event shape [num_arcs, 2], [num_arcs, 3].
        """
        return JointDistribution(
            [
                self.ids_and_on_offs_generative_model.get_latent_dist(),
                self.generative_motor_noise_dist,
            ]
        )

    def get_obs_params(self, latent):
        """
        Args:
            latent:
                ids_and_on_offs: [batch_size, num_arcs, 2]
                motor_noise: [batch_size, num_arcs, 3]

        Returns:
            probs: tensor [batch_size, num_rows, num_cols]
        """
        ids_and_on_offs, motor_noise = latent
        batch_size = ids_and_on_offs.shape[0]
        ids = ids_and_on_offs[..., 0]
        on_offs = ids_and_on_offs[..., 1]
        start_point = self.start_point.unsqueeze(0).expand(batch_size, -1)

        # [batch_size, num_arcs, 7]
        arcs = models.get_arcs(
            start_point, self.ids_and_on_offs_generative_model.get_primitives(), ids
        )

        # Add motor noise!
        arcs[:, :, 2:5] += motor_noise

        return rendering.get_logits(
            arcs,
            on_offs,
            self.ids_and_on_offs_generative_model.get_rendering_params(),
            self.ids_and_on_offs_generative_model.num_rows,
            self.ids_and_on_offs_generative_model.num_cols,
        )

    def get_obs_dist(self, latent):
        """Likelihood p(x | z, c)

        Args:
            latent:
                ids_and_on_offs: [batch_size, num_arcs, 2]
                motor_noise: [batch_size, num_arcs, 3]

        Returns: distribution with batch_shape [batch_size] and event_shape
            [num_rows, num_cols]
        """
        logits = self.get_obs_params(latent)
        if self.ids_and_on_offs_generative_model.likelihood == "learned-affine":
            return models.AffineLikelihood(
                logits.sigmoid(), self.ids_and_on_offs_generative_model._likelihood.theta_affine
            )
        elif self.ids_and_on_offs_generative_model.likelihood == "bernoulli":
            return torch.distributions.Independent(
                torch.distributions.Bernoulli(logits=logits), reinterpreted_batch_ndims=2
            )
        else:
            raise NotImplementedError

    def get_log_prob(self, latent, obs):
        """Log of joint probability.

        p(x, z, c)

        Args:
            latent:
                ids_and_on_offs: [num_particles, batch_size, num_arcs, 2]
                motor_noise: [num_particles, batch_size, num_arcs, 3]
            obs: [batch_size, num_rows, num_cols]

        Returns: tensor of shape [num_particles, batch_size]
        """
        ids_and_on_offs, motor_noise = latent
        num_particles, batch_size = ids_and_on_offs.shape[:2]
        num_rows, num_cols = obs.shape[1:]

        # [num_particles, batch_size]
        latent_log_prob = self.get_latent_dist().log_prob(latent)

        # [num_particles, batch_size]
        obs_log_prob = einops.rearrange(
            self.get_obs_dist(
                tuple(
                    [
                        einops.rearrange(
                            x, "num_particles batch_size ... -> (num_particles batch_size) ...",
                        )
                        for x in latent
                    ]
                )
            ).log_prob(
                einops.repeat(
                    obs,
                    "batch_size ... -> (num_particles batch_size) ...",
                    num_particles=num_particles,
                )
            ),
            "(num_particles batch_size) -> num_particles batch_size",
            num_particles=num_particles,
        )

        return latent_log_prob + obs_log_prob

    def sample_latent_and_obs(self, num_samples=1):
        """Args:
            num_samples: int

        Returns:
            latent
                ids_and_on_offs [num_samples, num_arcs, 2]
                motor_noise [num_samples, num_arcs, 3]
            obs: tensor of shape [num_samples, num_rows, num_cols]
        """
        latent = self.get_latent_dist().sample((num_samples,))
        obs = self.get_obs_dist(latent).sample()

        return latent, obs


class Guide(nn.Module):
    """Guide for the motor noise model

    q(z | x) q(c | z, x)
    """

    def __init__(self, ids_and_on_offs_guide_args):
        super().__init__()
        self.ids_and_on_offs_guide = models.Guide(*ids_and_on_offs_guide_args)
        self.num_arcs = self.ids_and_on_offs_guide.num_arcs

    def sample_ids_and_on_offs(self, obs, num_particles):
        """Sample from q(z | x)

        Args:
            obs: tensor of shape [batch_size, num_rows, num_cols]
            num_particles: int

        Returns: tensor of shape [num_particles, batch_size, num_arcs, 2]
        """
        return self.ids_and_on_offs_guide.sample(obs, num_particles)

    def get_motor_noise_dist(self, ids_and_on_offs, obs):
        """q(c | z, x)

        Args:
            ids_and_on_offs: [*shape, num_arcs, 2]
            obs: tensor of shape [*shape, num_rows, num_cols]

        Returns: distribution with batch_shape [*shape]
            and event_shape [num_arcs, 3]
        """
        return GuideMotorNoiseDist(obs, self.num_arcs)

    def get_latent_dist(self, obs):
        """q(z | x) q(c | z, x)

        Args:
            obs: tensor of shape [batch_size, num_rows, num_cols]

        Returns: distribution with batch_shape [batch_size]
            and event_shape ([num_arcs, 2], [num_arcs, 3])
        """
        return JointDistribution(
            [self.ids_and_on_offs_guide.get_latent_dist(obs), self.get_motor_noise_dist(None, obs),]
        )

    def sample_from_latent_dist(self, latent_dist, num_particles):
        """Samples from q(z | x) q(c | z, x)

        Args:
            latent_dist: distribution with batch_shape [batch_size]
                and event_shape ([num_arcs, 2], [num_arcs, 3])
            num_particles: int

        Returns:
            ids_and_on_offs: [num_particles, batch_size, num_arcs, 2]
            motor_noise: [num_particles, batch_size, num_arcs, 3]
        """
        return latent_dist.sample([num_particles])

    def sample(self, obs, num_particles):
        """Samples from q(z | x) q(c | z, x)

        Args:
            obs: tensor of shape [batch_size, num_rows, num_cols]
            num_particles: int

        Returns:
            ids_and_on_offs: [num_particles, batch_size, num_arcs, 2]
            motor_noise: [num_particles, batch_size, num_arcs, 3]
        """
        return self.sample_from_latent_dist(self.get_latent_dist(obs), num_particles)

    def get_log_prob_from_latent_dist(self, latent_dist, latent):
        """Log q(z | x) q(c | z, x).

        Args:
            latent_dist: distribution with batch_shape [batch_size]
                and event_shape ([num_arcs, 2], [num_arcs, 3])
            latent:
                ids_and_on_offs: [num_particles, batch_size, num_arcs, 2]
                motor_noise: [num_particles, batch_size, num_arcs, 3]

        Returns: tensor of shape [num_particles, batch_size]
        """
        return latent_dist.log_prob(latent)

    def get_log_prob(self, latent, obs):
        """Log q(z | x) q(c | z, x).

        Args:
            latent
                ids_and_on_offs: [num_particles, batch_size, num_arcs, 2]
                motor_noise: [num_particles, batch_size, num_arcs, 3]
            obs: tensor of shape [batch_size, num_rows, num_cols]

        Returns: tensor of shape [num_particles, batch_size]
        """
        return self.get_log_prob_from_latent_dist(self.get_latent_dist(obs), latent)
