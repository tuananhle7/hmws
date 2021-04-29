import torch
import torch.nn as nn


class GenerativeModel(nn.Module):
    """
    """

    def __init__(self):
        super().__init__()

        raise NotImplementedError()

    @property
    def device(self):
        raise NotImplementedError()

    def latent_log_prob(self, latent):
        """Prior log p(z)

        Args:
            latent:
                raw_expression [*shape, max_num_chars]
                eos [*shape, max_num_chars]
                gp_raw_params [*shape, max_num_chars]

        Returns: [*shape]
        """
        raise NotImplementedError()

    def latent_sample(self, sample_shape=[]):
        """Sample from p(z)

        Args
            sample_shape

        Returns
            latent:
                raw_expression [*sample_shape, max_num_chars]
                eos [*sample_shape, max_num_chars]
                gp_raw_params [*sample_shape, max_num_chars]
        """
        raise NotImplementedError()

    def discrete_latent_sample(self, sample_shape=[]):
        """Sample from p(z_d)

        Args
            sample_shape

        Returns
            latent:
                raw_expression [*sample_shape, max_num_chars]
                eos [*sample_shape, max_num_chars]
        """
        raise NotImplementedError()

    def log_prob(self, latent, obs):
        """Log joint probability of the generative model
        log p(z, x)

        Args:
            latent:
                raw_expression [*sample_shape, *shape, max_num_chars]
                eos [*sample_shape, *shape, max_num_chars]
                gp_raw_params [*sample_shape, *shape, max_num_chars]

            obs [*shape, num_timesteps]

        Returns: [*sample_shape, *shape]
        """
        raise NotImplementedError()

    def log_prob_discrete_continuous(self, discrete_latent, continuous_latent, obs):
        """Log joint probability of the generative model
        log p(z, x)

        Args:
            discrete_latent
                raw_expression [*discrete_shape, *shape, max_num_chars]
                eos [*discrete_shape, *shape, max_num_chars]
            continuous_latent
                gp_raw_params [*continuous_shape, *discrete_shape, *shape, max_num_chars]

            obs [*shape, num_timesteps]

        Returns: [*continuous_shape, *discrete_shape, *shape]
        """
        raise NotImplementedError()

    @torch.no_grad()
    def sample(self, sample_shape=[]):
        """Sample from p(z, x)

        Args
            sample_shape

        Returns
            latent:
                raw_expression [*sample_shape, max_num_chars]
                eos [*sample_shape, max_num_chars]
                gp_raw_params [*sample_shape, max_num_chars]

            obs [*sample_shape, num_timesteps]
        """
        raise NotImplementedError()


class Guide(nn.Module):
    """
    """

    def __init__(self):
        super().__init__()

        raise NotImplementedError()

    @property
    def device(self):
        raise NotImplementedError()

    def log_prob(self, obs, latent):
        """
        Args
            obs [*shape, num_timesteps]
            latent:
                raw_expression [*sample_shape, *shape, max_num_chars]
                eos [*sample_shape, *shape, max_num_chars]
                gp_raw_params [*sample_shape, *shape, max_num_chars]

        Returns [*sample_shape, *shape]
        """
        raise NotImplementedError()

    def sample(self, obs, sample_shape=[]):
        """z ~ q(z | x)

        Args
            obs [*shape, num_timesteps]

        Returns
            raw_expression [*sample_shape, *shape, max_num_chars]
            eos [*sample_shape, *shape, max_num_chars]
            gp_raw_params [*sample_shape, *shape, max_num_chars]
        """
        raise NotImplementedError()

    def sample_discrete(self, obs, sample_shape=[]):
        """z_d ~ q(z_d | x)

        Args
            obs [*shape, num_timesteps]
            sample_shape

        Returns
            raw_expression [*sample_shape, *shape, max_num_chars]
            eos [*sample_shape, *shape, max_num_chars]
        """
        raise NotImplementedError()

    def sample_continuous(self, obs, discrete_latent, sample_shape=[]):
        """z_c ~ q(z_c | z_d, x)

        Args
            obs [*shape, num_timesteps]
            discrete_latent
                raw_expression [*discrete_shape, *shape, max_num_chars]
                eos [*discrete_shape, *shape, max_num_chars]
            sample_shape

        Returns
                gp_raw_params [*sample_shape, *discrete_shape, *shape, max_num_chars]
        """
        raise NotImplementedError()

    def log_prob_discrete(self, obs, discrete_latent):
        """log q(z_d | x)

        Args
            obs [*shape, num_timesteps]
            discrete_latent
                raw_expression [*discrete_shape, *shape, max_num_chars]
                eos [*discrete_shape, *shape, max_num_chars]

        Returns [*discrete_shape, *shape]
        """
        raise NotImplementedError()

    def log_prob_continuous(self, obs, discrete_latent, continuous_latent):
        """log q(z_c | z_d, x)

        Args
            obs [*shape, num_timesteps]
            discrete_latent
                raw_expression [*discrete_shape, *shape, max_num_chars]
                eos [*discrete_shape, *shape, max_num_chars]

            continuous_latent (gp_raw_params)
                [*continuous_shape, *discrete_shape, *shape, max_num_chars]

        Returns [*continuous_shape, *discrete_shape, *shape]
        """
        raise NotImplementedError()
