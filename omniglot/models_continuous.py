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


class GenerativeMotorNoise(nn.Module):
    """p(c)

    Distribution of
    batch_shape [] event_shape [num_arcs, 3]
    """

    def __init__(self, num_arcs):
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


class GuideMotorNoise:
    """q(c | z, x)

    Distribution of
    batch_shape [] event_shape [num_arcs, 3]

    # TODO: consider different architectures
    """

    def __init__(self, num_arcs):
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


class GenerativeModel(nn.Module):
    """p(z)p(c)p(x | z, c)
    """

    def __init__(self, discrete_generative_model_args, device):
        super().__init__()
        self.discrete_generative_model = models.GenerativeModel(
            *discrete_generative_model_args, alphabet_dim=0, device=device
        )
        self.generative_motor_noise = GenerativeMotorNoise(self.discrete_generative_model.num_arcs)

    def get_latent_dist(self):
        """
        Returns: distribution with batch shape [] and event shape [num_arcs, 2], [num_arcs, 3].
        """
        return JointDistribution(
            self.discrete_generative_model.get_latent_dist(), self.generative_motor_noise
        )

    def get_obs_params(self, latent):
        """
        Args:
            latent:
                discrete_latent: [batch_size, num_arcs, 2]
                continuous_latent: [batch_size, num_arcs, 3]

        Returns:
            probs: tensor [batch_size, num_rows, num_cols]
        """
        discrete_latent, continuous_latent = latent
        batch_size = discrete_latent.shape[0]
        ids = discrete_latent[..., 0]
        on_offs = discrete_latent[..., 1]
        start_point = self.start_point.unsqueeze(0).expand(batch_size, -1)

        # [batch_size, num_arcs, 7]
        arcs = models.get_arcs(start_point, self.get_primitives(), ids)

        # Add motor noise!
        arcs[:, :, 2:5] += continuous_latent

        return rendering.get_logits(
            arcs, on_offs, self.get_rendering_params(), self.num_rows, self.num_cols
        )

    def get_obs_log_prob(self, latent, obs):
        """Log likelihood.

        p(x | z, c)

        Args:
            latent:
                discrete_latent: [num_particles, batch_size, num_arcs, 2]
                continuous_latent: [num_particles, batch_size, num_arcs, 3]
            obs: [batch_size, num_rows, num_cols]

        Returns: tensor of shape [num_particles, batch_size]
        """
        pass

    def get_log_prob(self, latent, obs):
        """Log of joint probability.

        Args:
            latent:
                discrete_latent: [num_particles, batch_size, num_arcs, 2]
                continuous_latent: [num_particles, batch_size, num_arcs, 3]
            obs: [batch_size, num_rows, num_cols]

        Returns: tensor of shape [num_particles, batch_size]
        """
        num_images_per_alphabet = obs.shape[1]
        latent_log_prob = self.get_latent_dist(num_images_per_alphabet).log_prob(latent)
        obs_log_prob = self.get_obs_log_prob(latent, obs)
        return latent_log_prob + obs_log_prob

    # TODO
    @torch.no_grad()
    def sample(self, alphabet, num_samples, batch_size=None):
        """
        Args:
            alphabet: [batch_size, alphabet_dim]
            num_samples
            batch_size

        Returns: [num_samples, batch_size, num_rows, num_cols]
        """
        if alphabet is None:
            # [num_samples, batch_size, num_arcs, 2]
            image_latent = self.discrete_generative_model.get_latent_dist().sample(
                (num_samples, batch_size)
            )
        else:
            batch_size, alphabet_dim = alphabet.shape
            alphabet = alphabet[None].expand(num_samples, batch_size, alphabet_dim)
            # [num_samples, batch_size, num_arcs, 2]
            image_latent = self.discrete_generative_model.get_latent_dist(alphabet).sample()

        return self.discrete_generative_model.get_obs_dist(image_latent).sample()


class Guide(nn.Module):
    """Guide for a hierarchical generative model.

        q(alphabet | image_{1:M}) * prod_m q(image_latent_m | image_m, alphabet)
    """

    def __init__(self, alphabet_dim, image_guide_args):
        super().__init__()
        self.alphabet_dim = alphabet_dim
        self.alphabet_embedder = util.cnn(2 * self.alphabet_dim)
        self.image_guide = models.Guide(*image_guide_args, alphabet_dim=self.alphabet_dim)

    def get_alphabet_params(self, obs):
        """Params of q(alphabet | image_{1:M})

        Args:
            obs: [batch_size, num_images_per_alphabet, num_rows, num_cols]

        Returns:
            loc: tensor [batch_size, alphabet_dim]
            scale: tensor [batch_size, alphabet_dim]
        """
        batch_size, num_images_per_alphabet, num_rows, num_cols = obs.shape
        obs_flattened = obs.view(batch_size * num_images_per_alphabet, 1, num_rows, num_cols)
        raw_alphabet_loc, raw_alphabet_scale = (
            self.alphabet_embedder(obs_flattened)
            .view(batch_size, num_images_per_alphabet, -1)
            .chunk(2, dim=-1)
        )

        return raw_alphabet_loc, raw_alphabet_scale.exp()

    def get_latent_dist(self, obs):
        """Args:
            obs: [batch_size, num_images_per_alphabet, num_rows, num_cols]

        Returns: distribution with batch shape [batch_size] and
            event shape ([alphabet_dim], [num_images_per_alphabet, num_arcs, 2]).
        """
        alphabet_params = self.get_alphabet_params(obs)
        return GuideLatentDist(alphabet_params, obs, self.image_guide,)
