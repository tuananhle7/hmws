import torch
import torch.nn as nn
import cmws


class GenerativeModel(nn.Module):
    """Switching Linear Dynamical Model
    https://github.com/lindermanlab/ssm/blob/db6d73670cb9ff0ea6ddf24926297aa4c706a3d3/notebooks/3%20Switching%20Linear%20Dynamical%20System.ipynb

    Args
        num_states (int): K
        continuous_dim (int): D
        obs_dim (int): N
        single_subspace (bool; default True)
            https://github.com/lindermanlab/ssm/blob/master/ssm/lds.py#L40
    """

    def __init__(self, num_states, continuous_dim, obs_dim, num_timesteps, single_subspace=True):
        super().__init__()
        self.num_states = num_states
        self.continuous_dim = continuous_dim
        self.obs_dim = obs_dim
        self.num_timesteps = num_timesteps
        self.single_subspace = single_subspace

        # Initial state params
        # --Logits for discrete state
        self.discrete_init_logits = nn.Parameter(torch.randn(self.num_states))

        # --Locs and scales for continuous state
        self.continuous_init_locs = nn.Parameter(torch.randn(self.num_states, self.continuous_dim))
        self.continuous_init_log_scales = nn.Parameter(
            torch.randn(self.num_states, self.continuous_dim)
        )

        # Transition model params
        # --Logits for Q matrix [K, K]
        self.state_transition_logits = nn.Parameter(torch.randn(self.num_states, self.num_states))

        # --Dynamics matrices A_k ∊ [D, D]
        self.dynamics_matrices = nn.Parameter(
            torch.randn(self.num_states, self.continuous_dim, self.continuous_dim)
        )

        # --Dynamics offset b_k ∊ [D]
        self.dynamics_offsets = nn.Parameter(torch.randn(self.num_states, self.continuous_dim))

        # --Dynamics log scales
        self.dynamics_log_scales = nn.Parameter(torch.randn(self.num_states, self.continuous_dim))

        # Emission model params
        if self.single_subspace:
            # --Emission matrix C ∊ [N, D]
            self.emission_matrix = nn.Parameter(torch.randn(self.obs_dim, self.continuous_dim))

            # --Emission offset d ∊ [N]
            self.emission_offset = nn.Parameter(torch.randn(self.obs_dim))

            # --Emission log scale
            self.emission_log_scale = nn.Parameter(torch.randn(self.obs_dim))
        else:
            # --Emission matrices C_k ∊ [N, D]
            self.emission_matrices = nn.Parameter(
                torch.randn(self.num_states, self.obs_dim, self.continuous_dim)
            )

            # --Emission offset d_k ∊ [N]
            self.emission_offsets = nn.Parameter(torch.randn(self.num_states, self.obs_dim))

            # --Emission log scales
            self.emission_log_scales = nn.Parameter(torch.randn(self.num_states, self.obs_dim))

    @property
    def device(self):
        return self.state_transition_logits.device

    @property
    def init_discrete_state_dist(self):
        """p(s_1)
        
        Initial distribution for the discrete state
        batch_shape [], event_shape []
        """
        return torch.distributions.Categorical(logits=self.discrete_init_logits)

    def init_continuous_state_dist(self, init_discrete_state):
        """p(z_1 | s_1)

        Initial distribution for the continuous state

        Args
            init_discrete_state [*shape]
        
        Returns distribution with batch_shape [*shape], event_shape [continuous_dim]
        """
        loc = self.continuous_init_locs[init_discrete_state]
        scale = self.continuous_init_log_scales[init_discrete_state].exp()
        return torch.distributions.Independent(
            torch.distributions.Normal(loc, scale), reinterpreted_batch_ndims=1
        )

    def discrete_state_dist(self, prev_discrete_state):
        """p(s_t | s_{t - 1})

        Transition distribution for the discrete state

        Args
            prev_discrete_state [*shape]
        
        Returns distribution with batch_shape [*shape], event_shape []
        """
        logits = self.state_transition_logits[prev_discrete_state]
        return torch.distributions.Categorical(logits=logits)

    def continuous_state_dist(self, prev_continuous_state, discrete_state):
        """p(z_t | z_{t - 1}, s_t)

        Dynamics distribution for the continuous state

        Args
            prev_continuous_state [*shape, continuous_dim]
            discrete_state [*shape]

        Returns distribution with batch_shape [*shape], event_shape [continuous_dim]
        """
        loc = (
            torch.einsum(
                "...ij,...j->...i", self.dynamics_matrices[discrete_state], prev_continuous_state
            )
            + self.dynamics_offsets[discrete_state]
        )
        scale = self.dynamics_log_scales[discrete_state].exp()
        return torch.distributions.Independent(
            torch.distributions.Normal(loc, scale), reinterpreted_batch_ndims=1
        )

    def latent_log_prob(self, latent):
        """Prior log p(z)

        Args:
            latent:
                discrete_states [*shape, num_timesteps]
                continuous_states [*shape, num_timesteps, continuous_dim]

        Returns: [*shape]
        """
        log_prob = 0
        discrete_states, continuous_states = latent

        # Init log prob
        init_discrete_state = discrete_states[..., 0]  # [*shape]
        init_continuous_state = continuous_states[..., 0, :]  # [*shape, continuous_dim]
        log_prob += self.init_discrete_state_dist.log_prob(
            init_discrete_state
        ) + self.init_continuous_state_dist(init_discrete_state).log_prob(init_continuous_state)

        # Next log probs
        for timestep in range(1, self.num_timesteps):
            # [*shape]
            prev_discrete_state = discrete_states[..., timestep - 1]
            # [*shape, continuous_dim]
            prev_continuous_state = continuous_states[..., timestep - 1, :]
            # [*shape]
            discrete_state = discrete_states[..., timestep]
            # [*shape, continuous_dim]
            continuous_state = continuous_states[..., timestep, :]

            log_prob += self.discrete_state_dist(prev_discrete_state).log_prob(
                discrete_state
            ) + self.continuous_state_dist(prev_continuous_state, discrete_state).log_prob(
                continuous_state
            )

        return log_prob

    def latent_sample(self, sample_shape=[]):
        """Sample from p(s_{1:T}, z_{1:T})

        Args
            sample_shape

        Returns
            latent:
                discrete_states [*sample_shape, num_timesteps]
                continuous_states [*sample_shape, num_timesteps, continuous_dim]
        """
        # Sample p(s_{1:T})
        discrete_states = self.discrete_latent_sample(sample_shape)
        continuous_states = []

        # Sample p(z_1 | s_1)
        continuous_states.append(self.init_continuous_state_dist(discrete_states[..., 0]).sample())

        # Sample p(z_{2:T} | s_{2:T})
        for timestep in range(1, self.num_timesteps):
            continuous_states.append(
                self.continuous_state_dist(
                    continuous_states[-1], discrete_states[..., timestep]
                ).sample()
            )

        continuous_states = torch.stack(continuous_states, -2)
        return discrete_states, continuous_states

    def discrete_latent_sample(self, sample_shape=[]):
        """Sample from p(s_{1:t})

        Args
            sample_shape

        Returns
            discrete_latent (discrete_states) [*sample_shape, num_timesteps]
        """
        discrete_states = []

        # Init log prob
        discrete_states.append(self.init_discrete_state_dist.sample(sample_shape))

        # Next log probs
        for timestep in range(1, self.num_timesteps):
            discrete_states.append(self.discrete_state_dist(discrete_states[-1]).sample())

        discrete_states = torch.stack(discrete_states, -1)
        return discrete_states

    def obs_dist(self, discrete_state, continuous_state):
        """Emission distribution p(x_t | z_t, s_t)
        
        Args
            discrete_state [*shape]
            continuous_state [*shape, continuous_dim]
        
        Returns distribution with batch_shape [*shape], event_shape [obs_dim]
        """
        if self.single_subspace:
            loc = (
                torch.einsum("ij,...j->...i", self.emission_matrix, continuous_state)
                + self.emission_offset
            )
            scale = self.emission_log_scale.exp()
        else:
            loc = (
                torch.einsum(
                    "...ij,...j->...i", self.emission_matrices[discrete_state], continuous_state
                )
                + self.emission_offsets[discrete_state]
            )
            scale = self.emission_log_scales[discrete_state].exp()
        return torch.distributions.Independent(
            torch.distributions.Normal(loc, scale), reinterpreted_batch_ndims=1
        )

    def obs_log_prob(self, latent, obs):
        """Log joint probability of the generative model
        log p(x_{1:T} | z_{1:T}, s_{1:T})

        Args:
            latent:
                discrete_states [*sample_shape, *shape, num_timesteps]
                continuous_states [*sample_shape, *shape, num_timesteps, continuous_dim]

            obs [*shape, num_timesteps, obs_dim]

        Returns: [*sample_shape, *shape]
        """
        # Extract
        discrete_states, continuous_states = latent
        shape = obs.shape[:-2]
        sample_shape = discrete_states.shape[: -(1 + len(shape))]
        num_samples = cmws.util.get_num_elements(sample_shape)

        log_prob = 0
        for timestep in range(self.num_timesteps):
            # [*sample_shape, *shape, obs_dim]
            obs_expanded = (
                obs[..., timestep, :][None]
                .expand(*[num_samples, *shape, self.obs_dim])
                .view(*[*sample_shape, *shape, self.obs_dim])
            )
            log_prob += self.obs_dist(
                discrete_states[..., timestep], continuous_states[..., timestep, :]
            ).log_prob(obs_expanded)
        return log_prob

    def log_prob(self, latent, obs):
        """Log joint probability of the generative model
        log p(s_{1:T}, z_{1:T}, x_{1:T})

        Args:
            latent:
                discrete_states [*sample_shape, *shape, num_timesteps]
                continuous_states [*sample_shape, *shape, num_timesteps, continuous_dim]

            obs [*shape, num_timesteps, obs_dim]

        Returns: [*sample_shape, *shape]
        """
        return self.latent_log_prob(latent) + self.obs_log_prob(latent, obs)

    def log_prob_discrete_continuous(self, discrete_latent, continuous_latent, obs):
        """Log joint probability of the generative model
        log p(s_{1:T}, z_{1:T}, x_{1:T})

        Args:
            discrete_latent (discrete_states)
                [*discrete_shape, *shape, num_timesteps]
            continuous_latent (continuous_states)
                [*continuous_shape, *discrete_shape, *shape, num_timesteps, continuous_dim]

            obs [*shape, num_timesteps, obs_dim]

        Returns: [*continuous_shape, *discrete_shape, *shape]
        """
        # Extract
        discrete_states = discrete_latent
        continuous_states = continuous_latent
        shape = obs.shape[:-2]
        discrete_shape = discrete_states.shape[: -(len(shape) + 1)]
        continuous_shape = continuous_states.shape[: -(len(discrete_shape) + len(shape) + 2)]
        # num_elements = cmws.util.get_num_elements(shape)
        # num_discrete_elements = cmws.util.get_num_elements(discrete_shape)
        num_continuous_elements = cmws.util.get_num_elements(continuous_shape)

        # Expand discrete
        # [*continuous_shape, *discrete_shape, *shape, num_timesteps]
        discrete_states_expanded = (
            discrete_states.reshape(-1, self.num_timesteps)[None]
            .expand(num_continuous_elements, -1, self.num_timesteps)
            .view([*continuous_shape, *discrete_shape, *shape, self.num_timesteps])
        )
        return self.log_prob((discrete_states_expanded, continuous_states), obs)

    @torch.no_grad()
    def sample(self, sample_shape=[]):
        """Sample from p(s_{1:T}, z_{1:T}, x_{1:T})

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
        """Sample from p(x_{1:T} | s_{1:T}, z_{1:T})

        Args
            latent:
                discrete_states [*shape, num_timesteps]
                continuous_states [*shape, num_timesteps, continuous_dim]
            sample_shape

        Returns
            obs [*sample_shape, *shape, num_timesteps, obs_dim]
        """
        discrete_states, continuous_states = latent
        return torch.stack(
            [
                self.obs_dist(
                    discrete_states[..., timestep], continuous_states[..., timestep, :]
                ).sample(sample_shape)
                for timestep in range(self.num_timesteps)
            ],
            -2,
        )


class Guide(nn.Module):
    """Guide/Recognition model for the Switching Linear Dynamical Model
    https://github.com/lindermanlab/ssm/blob/db6d73670cb9ff0ea6ddf24926297aa4c706a3d3/notebooks/3%20Switching%20Linear%20Dynamical%20System.ipynb

    Args
        num_states (int): K
        continuous_dim (int): D
        obs_dim (int): N
    """

    def __init__(self, num_states, continuous_dim, obs_dim, num_timesteps):
        super().__init__()
        self.num_states = num_states
        self.continuous_dim = continuous_dim
        self.obs_dim = obs_dim
        self.num_timesteps = num_timesteps
        self.discrete_logits = nn.Parameter(torch.randn(self.num_timesteps, self.num_states))
        self.continuous_locs = nn.Parameter(torch.randn(self.num_timesteps, self.continuous_dim))
        self.continuous_log_scales = nn.Parameter(
            torch.randn(self.num_timesteps, self.continuous_dim)
        )

    @property
    def device(self):
        return self.discrete_logits.device

    @property
    def discrete_states_dist(self):
        """q(s_{1:T})
        """
        return torch.distributions.Independent(
            torch.distributions.Categorical(logits=self.discrete_logits),
            reinterpreted_batch_ndims=1,
        )

    @property
    def continuous_states_dist(self):
        """q(z_{1:T})"""
        return torch.distributions.Independent(
            torch.distributions.Normal(self.continuous_locs, self.continuous_log_scales.exp()),
            reinterpreted_batch_ndims=2,
        )

    def log_prob(self, obs, latent):
        """
        Args
            obs [*shape, num_timesteps, obs_dim]
            latent:
                discrete_states [*sample_shape, *shape, num_timesteps]
                continuous_states [*sample_shape, *shape, num_timesteps, continuous_dim]

        Returns [*sample_shape, *shape]
        """
        discrete_states, continuous_states = latent
        return self.log_prob_discrete(obs, discrete_states) + self.log_prob_continuous(
            obs, discrete_states, continuous_states
        )

    def sample(self, obs, sample_shape=[]):
        """z ~ q(z | x)

        Args
            obs [*shape, num_timesteps, obs_dim]

        Returns
            discrete_states [*sample_shape, *shape, num_timesteps]
            continuous_states [*sample_shape, *shape, num_timesteps, continuous_dim]
        """
        discrete_states = self.sample_discrete(obs, sample_shape)
        continuous_states = self.sample_continuous(obs, discrete_states)
        return discrete_states, continuous_states

    def sample_discrete(self, obs, sample_shape=[]):
        """s_{1:T} ~ q(s_{1:T} | x_{1:T})

        Args
            obs [*shape, num_timesteps, obs_dim]
            sample_shape

        Returns
            discrete_states [*sample_shape, *shape, num_timesteps]
        """
        # Extract
        shape = obs.shape[:-2]
        return self.discrete_states_dist.sample([*sample_shape, *shape])

    def _sample_continuous(self, reparam, obs, discrete_latent, sample_shape=[]):
        """z_{1:T} ~ q(z_{1:T} | x_{1:T}, s_{1:T})

        Args
            reparam (bool)
            obs [*shape, num_timesteps, obs_dim]
            discrete_latent (discrete_states) [*discrete_shape, *shape, num_timesteps]
            sample_shape

        Returns
            continuous_states
                [*sample_shape, *discrete_shape, *shape, num_timesteps, continuous_dim]
        """
        # Extract
        shape = obs.shape[:-2]
        discrete_shape = discrete_latent.shape[: -(1 + len(shape))]
        if reparam:
            return self.continuous_states_dist.rsample([*sample_shape, *discrete_shape, *shape])
        else:
            return self.continuous_states_dist.sample([*sample_shape, *discrete_shape, *shape])

    def sample_continuous(self, obs, discrete_latent, sample_shape=[]):
        """z_{1:T} ~ q(z_{1:T} | x_{1:T}, s_{1:T})

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
        """z_{1:T} ~ q(z_{1:T} | x_{1:T}, s_{1:T}) (reparameterized)

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
        return self.discrete_states_dist.log_prob(discrete_latent)

    def log_prob_continuous(self, obs, discrete_latent, continuous_latent):
        """log q(z_c | z_d, x)

        Args
            obs [*shape, num_timesteps]
            discrete_latent (discrete_states) [*discrete_shape, *shape, num_timesteps]
            continuous_latent (discrete_states)
                [*continuous_shape, *discrete_shape, *shape, num_timesteps, continuous_dim]

        Returns [*continuous_shape, *discrete_shape, *shape]
        """
        return self.continuous_states_dist.log_prob(continuous_latent)
