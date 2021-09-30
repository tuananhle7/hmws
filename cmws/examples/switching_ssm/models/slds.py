import torch
import torch.nn as nn


class GenerativeModel(nn.Module):
    """
    """

    def __init__(self, num_timesteps):
        super().__init__()

    @property
    def device(self):
        pass

    def latent_log_prob(self, latent):
        """Prior log p(z)

        Args:
            latent:
                discrete_states [*shape, num_timesteps]
                continuous_states [*shape, num_timesteps, continuous_dim]

        Returns: [*shape]
        """
        pass

    def latent_sample(self, sample_shape=[]):
        """Sample from p(z)

        Args
            sample_shape

        Returns
            latent:
                discrete_states [*sample_shape, num_timesteps]
                continuous_states [*sample_shape, num_timesteps, continuous_dim]
        """
        pass

    def discrete_latent_sample(self, sample_shape=[]):
        """Sample from p(z_d)

        Args
            sample_shape

        Returns
            discrete_latent (discrete_states) [*sample_shape, num_timesteps]
        """
        pass

    def log_prob(self, latent, obs):
        """Log joint probability of the generative model
        log p(z, x)

        Args:
            latent:
                discrete_states [*sample_shape, *shape, num_timesteps]
                continuous_states [*sample_shape, *shape, num_timesteps, continuous_dim]

            obs [*shape, num_timesteps, obs_dim]

        Returns: [*sample_shape, *shape]
        """
        pass

    def log_prob_discrete_continuous(self, discrete_latent, continuous_latent, obs):
        """Log joint probability of the generative model
        log p(z, x)

        Args:
            discrete_latent (discrete_states)
                [*discrete_shape, *shape, num_timesteps]
            continuous_latent (continuous_states)
                [*continuous_shape, *discrete_shape, *shape, num_timesteps, continuous_dim]

            obs [*shape, num_timesteps, obs_dim]

        Returns: [*continuous_shape, *discrete_shape, *shape]
        """
        pass

    @torch.no_grad()
    def sample(self, sample_shape=[]):
        """Sample from p(z, x)

        Args
            sample_shape

        Returns
            latent:
                discrete_states [*sample_shape, num_timesteps]
                continuous_states [*sample_shape, num_timesteps, continuous_dim]

            obs [*sample_shape, num_timesteps, obs_dim]
        """
        # Sample latent
        latent = self.latent_sample(sample_shape)

        # Sample obs
        obs = self.sample_obs(latent)

        return latent, obs

    @torch.no_grad()
    def sample_obs(self, latent, sample_shape=[]):
        """Sample from p(x | z)

        Args
            latent:
                discrete_states [*shape, num_timesteps]
                continuous_states [*shape, num_timesteps, continuous_dim]
            sample_shape

        Returns
            obs [*sample_shape, *shape, num_timesteps, obs_dim]
        """
        pass


class Guide(nn.Module):
    """
    """

    def __init__(self, num_timesteps, lstm_hidden_dim):
        super().__init__()

    @property
    def device(self):
        pass

    def log_prob(self, obs, latent):
        """
        Args
            obs [*shape, num_timesteps, obs_dim]
            latent:
                discrete_states [*sample_shape, *shape, num_timesteps]
                continuous_states [*sample_shape, *shape, num_timesteps, continuous_dim]

        Returns [*sample_shape, *shape]
        """
        pass

    def sample(self, obs, sample_shape=[]):
        """z ~ q(z | x)

        Args
            obs [*shape, num_timesteps, obs_dim]

        Returns
            discrete_states [*sample_shape, *shape, num_timesteps]
            continuous_states [*sample_shape, *shape, num_timesteps, continuous_dim]
        """
        pass

    def sample_discrete(self, obs, sample_shape=[]):
        """z_d ~ q(z_d | x)

        Args
            obs [*shape, num_timesteps, obs_dim]
            sample_shape

        Returns
            discrete_states [*sample_shape, *shape, num_timesteps]
        """
        pass

    def _sample_continuous(self, reparam, obs, discrete_latent, sample_shape=[]):
        """z_c ~ q(z_c | z_d, x)

        Args
            reparam (bool)
            obs [*shape, num_timesteps, obs_dim]
            discrete_latent (discrete_states) [*discrete_shape, *shape, num_timesteps]
            sample_shape

        Returns
            continuous_states
                [*sample_shape, *discrete_shape, *shape, num_timesteps, continuous_dim]
        """
        pass

    def sample_continuous(self, obs, discrete_latent, sample_shape=[]):
        """z_c ~ q(z_c | z_d, x)

        Args
            reparam (bool)
            obs [*shape, num_timesteps, obs_dim]
            discrete_latent (discrete_states) [*discrete_shape, *shape, num_timesteps]
            sample_shape

        Returns
            continuous_states
                [*sample_shape, *discrete_shape, *shape, num_timesteps, continuous_dim]
        """
        return self._sample_continuous(False, obs, discrete_latent, sample_shape=sample_shape)

    def rsample_continuous(self, obs, discrete_latent, sample_shape=[]):
        """z_c ~ q(z_c | z_d, x) (reparameterized)

        Args
            reparam (bool)
            obs [*shape, num_timesteps, obs_dim]
            discrete_latent (discrete_states) [*discrete_shape, *shape, num_timesteps]
            sample_shape

        Returns
            continuous_states
                [*sample_shape, *discrete_shape, *shape, num_timesteps, continuous_dim]
        """
        return self._sample_continuous(True, obs, discrete_latent, sample_shape=sample_shape)

    def log_prob_discrete(self, obs, discrete_latent):
        """log q(z_d | x)

        Args
            obs [*shape, num_timesteps, obs_dim]
            discrete_latent (discrete_states) [*discrete_shape, *shape, num_timesteps]

        Returns [*discrete_shape, *shape]
        """
        pass

    def log_prob_continuous(self, obs, discrete_latent, continuous_latent):
        """log q(z_c | z_d, x)

        Args
            obs [*shape, num_timesteps]
            discrete_latent (discrete_states) [*discrete_shape, *shape, num_timesteps]
            continuous_latent (discrete_states)
                [*continuous_shape, *discrete_shape, *shape, num_timesteps, continuous_dim]

        Returns [*continuous_shape, *discrete_shape, *shape]
        """
        pass
