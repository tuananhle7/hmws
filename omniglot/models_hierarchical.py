import util
import models
import torch
import torch.nn as nn


class GenerativeModelLatentDist(torch.distributions.Distribution):
    """The prior for the hierarchical generative model.
        p(alphabet, image_latent_{1:M}).

    Args:
        num_images_per_alphabet
        image_generative_model

    Returns: distribution with batch shape [num_particles, batch_size] and
        event shape
        ([alphabet_dim], [num_images_per_alphabet, num_arcs, 2]).
    """

    def __init__(self, num_images_per_alphabet, image_generative_model):
        super().__init__()
        self.num_images_per_alphabet = num_images_per_alphabet
        self.image_generative_model = image_generative_model

        self.alphabet_dim = self.image_generative_model.alphabet_dim

        self.register_buffer("alphabet_dist_loc", torch.zeros(self.alphabet_dim))
        self.register_buffer("alphabet_dist_scale", torch.ones(self.alphabet_dim))

    @property
    def alphabet_dist(self):
        """p(alphabet)

        batch_shape [], event_shape [alphabet_dim]"""
        return torch.distributions.Independent(
            torch.distributions.Normal(loc=self.alphabet_dist_loc, scale=self.alphabet_dist_scale,),
            reinterpreted_batch_ndims=1,
        )

    def sample(self, sample_shape=torch.Size()):
        raise NotImplementedError

    def log_prob(self, latent):
        """
        Args:
            latent:
                alphabet: [num_particles, batch_size, alphabet_dim]
                image_latent: [num_particles, batch_size, num_images_per_alphabet, num_arcs, 2]

        Returns: tensor [num_particles, batch_size]
        """
        alphabet, image_latent = latent
        # [num_particles, batch_size, num_images_per_alphabet, num_arcs, 2]
        # -> [num_images_per_alphabet, num_particles, batch_size, num_arcs, 2]
        image_latent = image_latent.permute(2, 0, 1, 3, 4)

        # [num_particles, batch_size]
        alphabet_log_prob = self.alphabet_dist.log_prob(alphabet)

        # [num_images_per_alphabet, num_particles, batch_size]
        image_log_probs = self.image_generative_model.get_latent_dist(alphabet).log_prob(
            image_latent
        )

        return alphabet_log_prob + image_log_probs.sum(dim=0)


class GenerativeModel(nn.Module):
    """Hierarchical generative model that reuses a single image
    generative model.
        p(alphabet) prod_m p(image_latent_m | alphabet)
                           p(image_m | image_latent_m)
    """

    def __init__(self, alphabet_dim, image_generative_model_args, device):
        super().__init__()
        self.image_generative_model = models.GenerativeModel(
            *image_generative_model_args, alphabet_dim=alphabet_dim, device=device
        )

    def get_latent_dist(self, num_images_per_alphabet):
        """
        Returns: distribution with batch shape [num_particles, batch_size] and
            event shape ([alphabet_dim], [num_images_per_alphabet, num_arcs, 2]).
        """
        return GenerativeModelLatentDist(num_images_per_alphabet, self.image_generative_model)

    def get_obs_log_prob(self, latent, obs):
        """Log likelihood.

        Args:
            latent:
                alphabet: [num_particles, batch_size, alphabet_dim]
                image_latent: [num_particles, batch_size, num_images_per_alphabet, num_arcs, 2]
            obs: [batch_size, num_images_per_alphabet, num_rows, num_cols]

        Returns: tensor of shape [num_particles, batch_size]
        """
        alphabet, image_latent = latent
        num_particles, batch_size, num_images_per_alphabet, num_arcs, _ = image_latent.shape
        num_rows, num_cols = obs.shape[-2:]
        obs_expanded = obs[None].expand(
            num_particles, batch_size, num_images_per_alphabet, num_rows, num_cols
        )

        return (
            self.image_generative_model.get_obs_dist(
                image_latent.view(num_particles * batch_size * num_images_per_alphabet, num_arcs, 2)
            )
            .log_prob(
                obs_expanded.view(
                    num_particles * batch_size * num_images_per_alphabet, num_rows, num_cols
                )
            )
            .view(num_particles, batch_size, num_images_per_alphabet)
            .sum(dim=-1)
        )

    def get_log_prob(self, latent, obs):
        """Log of joint probability.

        Args:
            latent:
                alphabet: [num_particles, batch_size, alphabet_dim]
                image_latent: [num_particles, batch_size, num_images_per_alphabet, num_arcs, 2]
            obs: [batch_size, num_images_per_alphabet, num_rows, num_cols]

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
            image_latent = self.image_generative_model.get_latent_dist().sample(
                (num_samples, batch_size)
            )
        else:
            batch_size, alphabet_dim = alphabet.shape
            alphabet = alphabet[None].expand(num_samples, batch_size, alphabet_dim)
            # [num_samples, batch_size, num_arcs, 2]
            image_latent = self.image_generative_model.get_latent_dist(alphabet).sample()

        return self.image_generative_model.get_obs_dist(image_latent).sample()


class GuideLatentDist(torch.distributions.Distribution):
    """Guide for a hierarchical generative model.
    This is a Distribution object.

        q(alphabet | image_{1:N}) *
        prod_n q(image_latent_n | image_n, alphabet)

    Args:
        alphabet_params:
            loc: tensor [batch_size, alphabet_dim]
            scale: tensor [batch_size, alphabet_dim]
        obs: [batch_size, num_images_per_alphabet, num_rows, num_cols]
        image_guide

    Returns: distribution with batch shape [batch_size] and
        event shape ([alphabet_dim], [num_images_per_alphabet, num_arcs, 2]).
    """

    def __init__(self, alphabet_params, obs, image_guide):
        super().__init__()
        self.obs = obs
        self.batch_size, self.num_images_per_alphabet, self.num_rows, self.num_cols = obs.shape[:2]
        self.alphabet_dim = alphabet_params[0].shape[1]
        # batch shape [batch_size], event shape [alphabet_dim]
        self.alphabet_dist = torch.distributions.Independent(
            torch.distributions.Normal(*alphabet_params), reinterpreted_batch_ndims=1
        )

    def sample(self, sample_shape=torch.Size()):
        """Only implemented for sample_shape = [num_particles].

        Returns:
            alphabet: [*sample_shape, batch_size, alphabet_dim]
            image_latent: [*sample_shape, batch_size, num_images_per_alphabet, num_arcs, 2]
        """
        if len(sample_shape) != 1:
            raise NotImplementedError

        num_particles = sample_shape[0]
        # [num_particles, batch_size, alphabet_dim]
        alphabet = self.alphabet_dist.sample(sample_shape).detach()

        # [num_particles, batch_size * num_images_per_alphabet, alphabet_dim]
        alphabet_expanded = (
            alphabet[:, :, None]
            .expand(num_particles, self.batch_size, self.num_images_per_alphabet, self.alphabet_dim)
            .view(num_particles, self.batch_size * self.num_images_per_alphabet, self.alphabet_dim)
        )

        # [batch_size * num_images_per_alphabet, num_rows, num_cols]
        obs_flattened = self.obs.view(-1, self.num_rows, self.num_cols,)

        # batch_shape [num_particles, batch_size * num_images_per_alphabet], event_shape [num_arcs, 2]
        image_guide_dist = self.image_guide.get_latent_dist(obs_flattened, alphabet_expanded,)

        # [num_particles, batch_size, num_images_per_alphabet, num_arcs, 2]
        image_latent = image_guide_dist.sample().view(
            num_particles, self.batch_size, self.num_images_per_alphabet, self.num_arcs, 2
        )

        return alphabet, image_latent

    def log_prob(self, latent):
        """
        Args:
            alphabet: [num_particles, batch_size, alphabet_dim]
            image_latent: [num_particles, batch_size, num_images_per_alphabet, num_arcs, 2]

        Returns: tensor [num_particles, batch_size]
        """
        alphabet, image_latent = latent
        num_particles = image_latent.shape[0]

        # [num_particles, batch_size]
        alphabet_log_prob = self.alphabet_dist.log_prob(alphabet)

        # [num_particles, batch_size * num_images_per_alphabet, alphabet_dim]
        alphabet_expanded = (
            alphabet[:, :, None]
            .expand(num_particles, self.batch_size, self.num_images_per_alphabet, self.alphabet_dim)
            .view(num_particles, self.batch_size * self.num_images_per_alphabet, self.alphabet_dim)
        )

        # [batch_size * num_images_per_alphabet, num_rows, num_cols]
        obs_flattened = self.obs.view(-1, self.num_rows, self.num_cols)

        # batch_shape [num_particles, batch_size * num_images_per_alphabet], event_shape [num_arcs, 2]
        image_guide_dist = self.image_guide.get_latent_dist(obs_flattened, alphabet_expanded)

        # [num_particles, batch_size * num_images_per_alphabet, num_arcs, 2]
        image_latent_flattened = image_latent.view(
            num_particles, self.batch_size * self.num_images_per_alphabet, self.num_arcs, 2
        )

        # [num_particles, batch_size]
        image_latent_log_prob = (
            image_guide_dist.log_prob(image_latent_flattened)
            .view(num_particles, self.batch_size, self.num_imagaes_per_alphabet)
            .sum(dim=-1)
        )

        return alphabet_log_prob + image_latent_log_prob


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
