import torch
import torch.nn as nn
import torch.nn.functional as F
import cmws
import cmws.examples.timeseries.util as timeseries_util
import cmws.examples.timeseries.lstm_util as lstm_util


class GenerativeModel(nn.Module):
    """
    """

    def __init__(self, max_num_chars, lstm_hidden_dim):
        super().__init__()
        self.max_num_chars = max_num_chars
        self.lstm_hidden_dim = lstm_hidden_dim

        # Prior for the expression (discrete)
        self.expression_lstm = nn.LSTM(timeseries_util.vocabulary_size, self.lstm_hidden_dim)
        self.expression_extractor = nn.Linear(
            self.lstm_hidden_dim, timeseries_util.vocabulary_size + 1
        )

        # Prior for the gp params (continuous)
        self.gp_params_lstm = nn.LSTM(
            timeseries_util.gp_params_dim + self.lstm_hidden_dim, self.lstm_hidden_dim
        )
        self.gp_params_extractor = nn.Linear(
            self.lstm_hidden_dim, 2 * timeseries_util.gp_params_dim
        )

    @property
    def device(self):
        return next(iter(self.expression_lstm.parameters())).device

    @property
    def expression_dist(self):
        """p(z_d)"""
        return lstm_util.TimeseriesDistribution(
            "discrete",
            timeseries_util.vocabulary_size,
            None,
            self.expression_lstm,
            self.expression_extractor,
            lstm_eos=True,
            max_num_timesteps=self.max_num_chars,
        )

    def get_expression_embedding(self, raw_expression, eos):
        """
        Args:
            raw_expression [*shape, max_num_chars]
            eos [*shape, max_num_chars]

        Returns: [*shape, lstm_hidden_dim]
        """
        # Extract
        shape = raw_expression.shape[:-1]
        num_elements = cmws.util.get_num_elements(shape)

        # Flatten
        raw_expression_flattened = raw_expression.view(-1, self.max_num_chars)
        eos_flattened = eos.view(-1, self.max_num_chars)

        # Compute num chars
        # [num_elements]
        num_chars_flattened = lstm_util.get_num_timesteps(eos_flattened)

        lstm_input = []
        for element_id in range(num_elements):
            num_chars_b = num_chars_flattened[element_id]
            lstm_input.append(
                F.one_hot(
                    raw_expression_flattened[element_id, :num_chars_b].long(),
                    num_classes=timeseries_util.vocabulary_size,
                ).float()
            )
        lstm_input = nn.utils.rnn.pack_sequence(lstm_input, enforce_sorted=False)

        # Run LSTM
        # [1, num_elements, lstm_hidden_size]
        _, (h, _) = self.expression_lstm(lstm_input)
        return h[0].view(*[*shape, self.lstm_hidden_dim])

    def get_num_base_kernels(self, raw_expression, eos):
        """
        Args:
            raw_expression [*shape, max_num_chars]
            eos [*shape, max_num_chars]

        Returns: [*shape]
        """
        # Extract
        shape = raw_expression.shape[:-1]
        num_elements = cmws.util.get_num_elements(shape)

        # Flatten
        raw_expression_flattened = raw_expression.view(-1, self.max_num_chars)
        eos_flattened = eos.view(-1, self.max_num_chars)

        # Compute num timesteps
        # [num_elements]
        num_timesteps_flattened = lstm_util.get_num_timesteps(eos_flattened)

        result = []
        for element_id in range(num_elements):
            result.append(
                timeseries_util.count_base_kernels(
                    raw_expression_flattened[element_id, : num_timesteps_flattened[element_id]]
                )
            )
        return torch.tensor(result, device=self.device).long().view(*shape)

    def latent_log_prob(self, latent):
        """Prior log p(z)

        Args:
            latent:
                raw_expression [*shape, max_num_chars]
                eos [*shape, max_num_chars]
                raw_gp_params [*shape, max_num_chars, gp_params_dim]

        Returns: [*shape]
        """
        # Extract
        raw_expression, eos, raw_gp_params = latent
        shape = raw_expression.shape[:-1]

        # Flatten
        raw_expression_flattened = raw_expression.view(-1, self.max_num_chars)
        eos_flattened = eos.view(-1, self.max_num_chars)
        raw_gp_params_flattened = raw_gp_params.view(
            -1, self.max_num_chars, timeseries_util.gp_params_dim
        )

        # Expression log prob
        expression_log_prob = self.expression_dist.log_prob(
            raw_expression_flattened, eos=eos_flattened
        ).view(*shape)

        # GP params log prob
        num_base_kernels_flattened = self.get_num_base_kernels(
            raw_expression_flattened, eos_flattened
        )

        # Compute expression embedding
        expression_embedding_flattened = self.get_expression_embedding(
            raw_expression_flattened, eos_flattened
        )
        gp_params_log_prob = (
            lstm_util.TimeseriesDistribution(
                "continuous",
                timeseries_util.gp_params_dim,
                expression_embedding_flattened,
                self.gp_params_lstm,
                self.gp_params_extractor,
                lstm_eos=False,
                max_num_timesteps=self.max_num_chars,
            )
            .log_prob(raw_gp_params_flattened, num_timesteps=num_base_kernels_flattened)
            .view(*shape)
        )

        return expression_log_prob + gp_params_log_prob

    def latent_sample(self, sample_shape=[]):
        """Sample from p(z)

        Args
            sample_shape

        Returns
            latent:
                raw_expression [*sample_shape, max_num_chars]
                eos [*sample_shape, max_num_chars]
                raw_gp_params [*sample_shape, max_num_chars, gp_params_dim]
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
                raw_gp_params [*sample_shape, *shape, max_num_chars, gp_params_dim]

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
                raw_gp_params
                [*continuous_shape, *discrete_shape, *shape, max_num_chars, gp_params_dim]

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
                raw_gp_params [*sample_shape, max_num_chars, gp_params_dim]

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
                raw_gp_params [*sample_shape, *shape, max_num_chars, gp_params_dim]

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
            raw_gp_params [*sample_shape, *shape, max_num_chars, gp_params_dim]
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
                raw_gp_params [*sample_shape, *discrete_shape, *shape, max_num_chars, gp_params_dim]
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

            continuous_latent (raw_gp_params, gp_params_dim)
                [*continuous_shape, *discrete_shape, *shape, max_num_chars]

        Returns [*continuous_shape, *discrete_shape, *shape]
        """
        raise NotImplementedError()
