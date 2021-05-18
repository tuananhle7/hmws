import torch
import torch.nn as nn
import torch.nn.functional as F
import cmws
import cmws.examples.timeseries.util as timeseries_util
import cmws.examples.timeseries.data as timeseries_data
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
        # self.gp_params_extractor.weight.data.fill_(0)
        # self.gp_params_extractor.bias.data.fill_(0)

    @property
    def device(self):
        return next(iter(self.expression_lstm.parameters())).device

    @property
    def expression_dist(self):
        """p(z_d) distribution of batch_shape [] and event_shape ([max_num_chars], [max_num_chars])
        """
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
        return h[0].view([*shape, self.lstm_hidden_dim])

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
        raw_expression_flattened = raw_expression.reshape(-1, self.max_num_chars)
        eos_flattened = eos.reshape(-1, self.max_num_chars)

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
        return torch.tensor(result, device=self.device).long().view(shape)

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
        raw_expression_flattened = raw_expression.reshape(-1, self.max_num_chars)
        eos_flattened = eos.reshape(-1, self.max_num_chars)
        raw_gp_params_flattened = raw_gp_params.view(
            -1, self.max_num_chars, timeseries_util.gp_params_dim
        )

        # Expression log prob
        expression_log_prob = self.expression_dist.log_prob(
            raw_expression_flattened, eos=eos_flattened
        ).view(shape)

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
            .view(shape)
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
        # Sample discrete
        raw_expression, eos = self.expression_dist.sample(sample_shape)

        # Flatten
        raw_expression_flattened = raw_expression.view(-1, self.max_num_chars)
        eos_flattened = eos.view(-1, self.max_num_chars)

        # Sample continuous
        # -- Compute embedding
        expression_embedding_flattened = self.get_expression_embedding(
            raw_expression_flattened, eos_flattened
        )
        # -- Compute num base kernels
        num_base_kernels_flattened = self.get_num_base_kernels(
            raw_expression_flattened, eos_flattened
        )

        raw_gp_params = (
            lstm_util.TimeseriesDistribution(
                "continuous",
                timeseries_util.gp_params_dim,
                expression_embedding_flattened,
                self.gp_params_lstm,
                self.gp_params_extractor,
                lstm_eos=False,
                max_num_timesteps=self.max_num_chars,
            )
            .sample(num_timesteps=num_base_kernels_flattened)
            .view([*sample_shape, self.max_num_chars, timeseries_util.gp_params_dim])
        )
        return raw_expression, eos, raw_gp_params

    def discrete_latent_sample(self, sample_shape=[]):
        """Sample from p(z_d)

        Args
            sample_shape

        Returns
            latent:
                raw_expression [*sample_shape, max_num_chars]
                eos [*sample_shape, max_num_chars]
        """
        return self.expression_dist.sample(sample_shape)

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
        # Extract
        raw_expression, eos, raw_gp_params = latent
        shape = obs.shape[:-1]
        sample_shape = raw_expression.shape[: -(len(shape) + 1)]
        num_elements = cmws.util.get_num_elements(shape)
        num_samples = cmws.util.get_num_elements(sample_shape)

        # Log p(z)
        latent_log_prob = self.latent_log_prob(latent)

        # Log p(x | z)
        # -- Create x_1 and x_2
        x_1 = torch.linspace(-2, 2, steps=timeseries_data.num_timesteps, device=self.device)[
            None, :, None
        ].expand(1, timeseries_data.num_timesteps, 1)
        x_2 = torch.linspace(-2, 2, steps=timeseries_data.num_timesteps, device=self.device)[
            None, None, :
        ].expand(1, 1, timeseries_data.num_timesteps)

        # -- Create identity matrix
        identity_matrix = torch.eye(timeseries_data.num_timesteps, device=self.device).float()

        # -- Flatten
        raw_expression_flattened = raw_expression.reshape(num_samples, num_elements, -1)
        eos_flattened = eos.reshape(num_samples, num_elements, -1)
        raw_gp_params_flattened = raw_gp_params.view(
            num_samples, num_elements, self.max_num_chars, -1
        )
        obs_flattened = obs.reshape(num_elements, -1)

        # -- Compute num timesteps
        num_timesteps_flattened = lstm_util.get_num_timesteps(eos_flattened)

        # -- Compute num base kernels
        num_base_kernels_flattened = self.get_num_base_kernels(
            raw_expression_flattened, eos_flattened
        )

        # -- Compute covariance matrices
        covariance_matrix = []
        zero_obs_prob = torch.zeros(
            (num_samples, num_elements), dtype=torch.bool, device=self.device
        )
        for sample_id in range(num_samples):
            for element_id in range(num_elements):
                try:
                    raw_expression_se = raw_expression_flattened[sample_id, element_id][
                        : num_timesteps_flattened[sample_id, element_id]
                    ]
                    covariance_matrix_se = timeseries_util.Kernel(
                        timeseries_util.get_expression(raw_expression_se),
                        raw_gp_params_flattened[
                            sample_id,
                            element_id,
                            : num_base_kernels_flattened[sample_id, element_id],
                        ],
                    )(x_1, x_2)[0]
                except timeseries_util.ParsingError:
                    covariance_matrix_se = identity_matrix
                    zero_obs_prob[sample_id, element_id] = True

                covariance_matrix.append(covariance_matrix_se)
        covariance_matrix = torch.stack(covariance_matrix).view(
            num_samples, num_elements, timeseries_data.num_timesteps, timeseries_data.num_timesteps
        )

        # -- Expand obs
        obs_expanded = obs_flattened[None].expand(num_samples, num_elements, -1)

        # -- Create mean
        loc = torch.zeros(
            (num_samples, num_elements, timeseries_data.num_timesteps), device=self.device
        )

        # -- Compute obs log prob
        try:
            obs_log_prob = cmws.util.get_multivariate_normal_dist(
                loc, covariance_matrix=covariance_matrix
            ).log_prob(obs_expanded)
        except Exception as e:
            raise RuntimeError(f"MVN log prob error: {e}")

        # -- Mask out zero log probs
        obs_log_prob[zero_obs_prob] = -1e6

        # -- Reshape
        obs_log_prob = obs_log_prob.view([*sample_shape, *shape])

        # Add penalty factor based on length
        length_penalty = 10 * num_timesteps_flattened.view([*sample_shape, *shape])

        return latent_log_prob + obs_log_prob - length_penalty

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
        # Extract
        raw_expression, eos = discrete_latent
        raw_gp_params = continuous_latent
        shape = obs.shape[:-1]
        discrete_shape = raw_expression.shape[: -(len(shape) + 1)]
        continuous_shape = raw_gp_params.shape[: -(len(discrete_shape) + len(shape) + 2)]
        # num_elements = cmws.util.get_num_elements(shape)
        # num_discrete_elements = cmws.util.get_num_elements(discrete_shape)
        num_continuous_elements = cmws.util.get_num_elements(continuous_shape)

        # Expand discrete
        # [*continuous_shape, *discrete_shape, *shape, max_num_chars]
        raw_expression_expanded = (
            raw_expression.reshape(-1, self.max_num_chars)[None]
            .expand(num_continuous_elements, -1, self.max_num_chars)
            .view([*continuous_shape, *discrete_shape, *shape, self.max_num_chars])
        )
        # [*continuous_shape, *discrete_shape, *shape, max_num_chars]
        eos_expanded = (
            eos.reshape(-1, self.max_num_chars)[None]
            .expand(num_continuous_elements, -1, self.max_num_chars)
            .view([*continuous_shape, *discrete_shape, *shape, self.max_num_chars])
        )
        return self.log_prob((raw_expression_expanded, eos_expanded, raw_gp_params), obs)

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
                raw_expression [*shape, max_num_chars]
                eos [*shape, max_num_chars]
                raw_gp_params [*shape, max_num_chars, gp_params_dim]
            sample_shape

        Returns
            obs [*sample_shape, *shape, num_timesteps]
        """
        # Extract
        raw_expression, eos, raw_gp_params = latent
        shape = raw_expression.shape[:-1]
        num_elements = cmws.util.get_num_elements(shape)

        # Create x_1 and x_2
        x_1 = torch.linspace(-2, 2, steps=timeseries_data.num_timesteps, device=self.device)[
            None, :, None
        ].expand(1, timeseries_data.num_timesteps, 1)
        x_2 = torch.linspace(-2, 2, steps=timeseries_data.num_timesteps, device=self.device)[
            None, None, :
        ].expand(1, 1, timeseries_data.num_timesteps)

        # Create identity matrix
        identity_matrix = torch.eye(timeseries_data.num_timesteps, device=self.device).float()

        # Flatten
        raw_expression_flattened = raw_expression.reshape(num_elements, -1)
        eos_flattened = eos.reshape(num_elements, -1)
        raw_gp_params_flattened = raw_gp_params.view(num_elements, self.max_num_chars, -1)

        # Compute num timesteps
        num_timesteps_flattened = lstm_util.get_num_timesteps(eos_flattened)

        # Compute num base kernels
        num_base_kernels_flattened = self.get_num_base_kernels(
            raw_expression_flattened, eos_flattened
        )

        # Compute covariance matrices
        covariance_matrix = []
        for element_id in range(num_elements):
            try:
                raw_expression_se = raw_expression_flattened[element_id][
                    : num_timesteps_flattened[element_id]
                ]
                covariance_matrix_se = timeseries_util.Kernel(
                    timeseries_util.get_expression(raw_expression_se),
                    raw_gp_params_flattened[element_id, : num_base_kernels_flattened[element_id],],
                )(x_1, x_2)[0]
            except timeseries_util.ParsingError:
                covariance_matrix_se = identity_matrix * 1e-6

            covariance_matrix.append(covariance_matrix_se)
        covariance_matrix = torch.stack(covariance_matrix).view(
            *[*shape, timeseries_data.num_timesteps, timeseries_data.num_timesteps]
        )

        # Create mean
        loc = torch.zeros(*[*shape, timeseries_data.num_timesteps], device=self.device)

        # Sample obs
        try:
            obs = cmws.util.get_multivariate_normal_dist(
                loc, covariance_matrix=covariance_matrix
            ).sample(sample_shape)
        except Exception as e:
            raise RuntimeError(f"MVN sample error: {e}")
            
        return obs

    @torch.no_grad()
    def sample_obs_predictions(self, latent, obs, sample_shape=[]):
        """Sample from p(x' | z, x)

        Args
            latent:
                raw_expression [*shape, max_num_chars]
                eos [*shape, max_num_chars]
                raw_gp_params [*shape, max_num_chars, gp_params_dim]
            obs [*shape, num_timesteps]
            sample_shape

        Returns
            obs_predictions [*sample_shape, *shape, num_timesteps]
        """
        # Extract
        raw_expression, eos, raw_gp_params = latent
        shape = raw_expression.shape[:-1]
        num_elements = cmws.util.get_num_elements(shape)

        # Create x_1 and x_2
        x_old = torch.linspace(-2, 2, steps=timeseries_data.num_timesteps, device=self.device)
        x_new = x_old + (x_old[-1] - x_old[0]) + (x_old[1] - x_old[0])
        x = torch.cat([x_old, x_new])
        joint_num_timesteps = len(x)
        x_1 = x[None, :, None].expand(1, joint_num_timesteps, 1)
        x_2 = x[None, None, :].expand(1, 1, joint_num_timesteps)

        # Create identity matrix
        identity_matrix = torch.eye(joint_num_timesteps, device=self.device).float()

        # Flatten
        raw_expression_flattened = raw_expression.reshape(num_elements, -1)
        eos_flattened = eos.reshape(num_elements, -1)
        raw_gp_params_flattened = raw_gp_params.view(num_elements, self.max_num_chars, -1)

        # Compute num chars
        num_chars_flattened = lstm_util.get_num_timesteps(eos_flattened)

        # Compute num base kernels
        num_base_kernels_flattened = self.get_num_base_kernels(
            raw_expression_flattened, eos_flattened
        )

        # Compute covariance matrices
        covariance_matrix = []
        for element_id in range(num_elements):
            try:
                raw_expression_se = raw_expression_flattened[element_id][
                    : num_chars_flattened[element_id]
                ]
                covariance_matrix_se = timeseries_util.Kernel(
                    timeseries_util.get_expression(raw_expression_se),
                    raw_gp_params_flattened[element_id, : num_base_kernels_flattened[element_id],],
                )(x_1, x_2)[0]
            except timeseries_util.ParsingError:
                covariance_matrix_se = identity_matrix * 1e-6

            covariance_matrix.append(covariance_matrix_se)
        covariance_matrix = torch.stack(covariance_matrix).view(
            *[*shape, joint_num_timesteps, joint_num_timesteps]
        )

        # Create mean
        loc = torch.zeros(*[*shape, joint_num_timesteps], device=self.device)

        # Sample obs
        try:
            joint_dist = cmws.util.get_multivariate_normal_dist(
                loc, covariance_matrix=covariance_matrix
            )
            predictive_dist = cmws.util.condition_mvn(joint_dist, obs)
            obs_predictions = predictive_dist.sample(sample_shape)
        except Exception as e:
            raise RuntimeError(f"MVN sample predictions error: {e}")

        return obs_predictions


class Guide(nn.Module):
    """
    """

    def __init__(self, max_num_chars, lstm_hidden_dim):
        super().__init__()
        self.max_num_chars = max_num_chars
        self.lstm_hidden_dim = lstm_hidden_dim

        # Obs embedder
        self.obs_embedder = nn.LSTM(1, self.lstm_hidden_dim)
        self.obs_embedding_dim = self.lstm_hidden_dim

        # Expression embedder
        self.expression_embedder = nn.LSTM(timeseries_util.vocabulary_size, self.lstm_hidden_dim)
        self.expression_embedding_dim = self.lstm_hidden_dim

        # Recognition model for the expression (discrete)
        self.expression_lstm = nn.LSTM(
            self.obs_embedding_dim + timeseries_util.vocabulary_size, self.lstm_hidden_dim
        )
        self.expression_extractor = nn.Linear(
            self.lstm_hidden_dim, timeseries_util.vocabulary_size + 1
        )

        # Recognition model for the gp params (continuous)
        self.gp_params_lstm = nn.LSTM(
            self.obs_embedding_dim + timeseries_util.gp_params_dim + self.lstm_hidden_dim,
            self.lstm_hidden_dim,
        )
        self.gp_params_extractor = nn.Linear(
            self.lstm_hidden_dim, 2 * timeseries_util.gp_params_dim
        )
        # self.gp_params_extractor.weight.data.fill_(0)
        # self.gp_params_extractor.bias.data.fill_(0)

    @property
    def device(self):
        return next(iter(self.expression_lstm.parameters())).device

    def get_obs_embedding(self, obs):
        """
        Args
            obs [*shape, num_timesteps]

        Returns [*shape, obs_embedding_dim]
        """
        # Extract
        shape = obs.shape[:-1]
        num_timesteps = obs.shape[-1]

        # Flatten
        obs_flattened = obs.reshape(-1, num_timesteps)

        _, (h, c) = self.obs_embedder(obs_flattened.T.view(num_timesteps, -1, 1))
        return h.view([*shape, self.obs_embedding_dim])

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
        _, (h, _) = self.expression_embedder(lstm_input)
        return h[0].view([*shape, self.lstm_hidden_dim])

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
        return torch.tensor(result, device=self.device).long().view(shape)

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
        # Extract
        raw_expression, eos, raw_gp_params = latent
        shape = obs.shape[:-1]
        sample_shape = raw_expression.shape[: -(len(shape) + 1)]
        num_elements = cmws.util.get_num_elements(shape)
        num_samples = cmws.util.get_num_elements(sample_shape)

        # Compute obs embedding
        obs_embedding = self.get_obs_embedding(obs)

        # Log prob of discrete
        # -- Expand obs embedding
        # [num_samples * num_elements, obs_embedding_dim]
        obs_embedding_expanded = (
            obs_embedding[None]
            .expand(*[num_samples, *shape, -1])
            .reshape(num_samples * num_elements, -1)
        )
        # -- Flatten discrete latents
        raw_expression_flattened = raw_expression.view(-1, self.max_num_chars)
        eos_flattened = eos.view(-1, self.max_num_chars)

        discrete_log_prob = (
            lstm_util.TimeseriesDistribution(
                "discrete",
                timeseries_util.vocabulary_size,
                obs_embedding_expanded,
                self.expression_lstm,
                self.expression_extractor,
                lstm_eos=True,
                max_num_timesteps=self.max_num_chars,
            )
            .log_prob(raw_expression_flattened, eos_flattened)
            .view([*sample_shape, *shape])
        )

        # Log prob of continuous
        # -- Compute num base kernels
        # [num_samples * num_elements]
        num_base_kernels_flattened = self.get_num_base_kernels(
            raw_expression_flattened, eos_flattened
        )

        # -- Compute expression embedding
        # [num_samples * num_elements, expression_embedding_dim]
        expression_embedding_flattened = self.get_expression_embedding(
            raw_expression_flattened, eos_flattened
        )

        # -- Flatten continuous latent
        # [num_samples * num_elements, max_num_chars, gp_params_dim]
        raw_gp_params_flattened = raw_gp_params.view(
            -1, self.max_num_chars, timeseries_util.gp_params_dim
        )

        # -- Compute log prob
        continuous_log_prob = (
            lstm_util.TimeseriesDistribution(
                "continuous",
                timeseries_util.gp_params_dim,
                torch.cat([obs_embedding_expanded, expression_embedding_flattened], dim=1),
                self.gp_params_lstm,
                self.gp_params_extractor,
                lstm_eos=False,
                max_num_timesteps=self.max_num_chars,
            )
            .log_prob(raw_gp_params_flattened, num_timesteps=num_base_kernels_flattened)
            .view([*sample_shape, *shape])
        )

        return discrete_log_prob + continuous_log_prob

    def sample(self, obs, sample_shape=[]):
        """z ~ q(z | x)

        Args
            obs [*shape, num_timesteps]

        Returns
            raw_expression [*sample_shape, *shape, max_num_chars]
            eos [*sample_shape, *shape, max_num_chars]
            raw_gp_params [*sample_shape, *shape, max_num_chars, gp_params_dim]
        """
        # Extract
        shape = obs.shape[:-1]
        num_elements = cmws.util.get_num_elements(shape)
        num_samples = cmws.util.get_num_elements(sample_shape)

        # Compute obs embedding
        obs_embedding = self.get_obs_embedding(obs)

        # Sample discrete
        # -- Flatten obs embedding
        # [num_elements, obs_embedding_dim]
        obs_embedding_flattened = obs_embedding.view(num_elements, -1)

        # -- Sample
        raw_expression, eos = lstm_util.TimeseriesDistribution(
            "discrete",
            timeseries_util.vocabulary_size,
            obs_embedding_flattened,
            self.expression_lstm,
            self.expression_extractor,
            lstm_eos=True,
            max_num_timesteps=self.max_num_chars,
        ).sample(sample_shape)

        # -- Reshape
        raw_expression = raw_expression.view([*sample_shape, *shape, -1])
        eos = eos.view([*sample_shape, *shape, -1])

        # Sample continuous
        # -- Flatten discrete latents
        raw_expression_flattened = raw_expression.view(-1, self.max_num_chars)
        eos_flattened = eos.view(-1, self.max_num_chars)

        # -- Compute num base kernels
        # [num_samples * num_elements]
        num_base_kernels_flattened = self.get_num_base_kernels(
            raw_expression_flattened, eos_flattened
        )

        # -- Compute expression embedding
        # [num_samples * num_elements, expression_embedding_dim]
        expression_embedding_flattened = self.get_expression_embedding(
            raw_expression_flattened, eos_flattened
        )

        # -- Expand obs embedding
        # [num_samples * num_elements, obs_embedding_dim]
        obs_embedding_expanded = (
            obs_embedding_flattened[None]
            .expand(num_samples, num_elements, -1)
            .reshape(-1, self.obs_embedding_dim)
        )

        # -- Sample
        raw_gp_params = (
            lstm_util.TimeseriesDistribution(
                "continuous",
                timeseries_util.gp_params_dim,
                torch.cat([obs_embedding_expanded, expression_embedding_flattened], dim=1),
                self.gp_params_lstm,
                self.gp_params_extractor,
                lstm_eos=False,
                max_num_timesteps=self.max_num_chars,
            )
            .sample((), num_timesteps=num_base_kernels_flattened)
            .view([*sample_shape, *shape, self.max_num_chars, timeseries_util.gp_params_dim])
        )

        return raw_expression, eos, raw_gp_params

    def sample_discrete(self, obs, sample_shape=[]):
        """z_d ~ q(z_d | x)

        Args
            obs [*shape, num_timesteps]
            sample_shape

        Returns
            raw_expression [*sample_shape, *shape, max_num_chars]
            eos [*sample_shape, *shape, max_num_chars]
        """
        # Extract
        shape = obs.shape[:-1]
        num_elements = cmws.util.get_num_elements(shape)

        # Compute obs embedding
        obs_embedding = self.get_obs_embedding(obs)

        # Sample discrete
        # -- Flatten obs embedding
        # [num_elements, obs_embedding_dim]
        obs_embedding_flattened = obs_embedding.view(num_elements, -1)

        # -- Sample
        raw_expression, eos = lstm_util.TimeseriesDistribution(
            "discrete",
            timeseries_util.vocabulary_size,
            obs_embedding_flattened,
            self.expression_lstm,
            self.expression_extractor,
            lstm_eos=True,
            max_num_timesteps=self.max_num_chars,
        ).sample(sample_shape)

        # -- Reshape
        raw_expression = raw_expression.view([*sample_shape, *shape, -1])
        eos = eos.view([*sample_shape, *shape, -1])

        return raw_expression, eos

    def _sample_continuous(self, reparam, obs, discrete_latent, sample_shape=[]):
        """z_c ~ q(z_c | z_d, x)

        Args
            reparam (bool)
            obs [*shape, num_timesteps]
            discrete_latent
                raw_expression [*discrete_shape, *shape, max_num_chars]
                eos [*discrete_shape, *shape, max_num_chars]
            sample_shape

        Returns
            raw_gp_params [*sample_shape, *discrete_shape, *shape, max_num_chars, gp_params_dim]
        """
        # Extract
        shape = obs.shape[:-1]
        num_elements = cmws.util.get_num_elements(shape)
        raw_expression, eos = discrete_latent
        discrete_shape = raw_expression.shape[: -(len(shape) + 1)]
        num_discrete_elements = cmws.util.get_num_elements(discrete_shape)
        num_samples = cmws.util.get_num_elements(sample_shape)

        # Compute obs embedding
        obs_embedding = self.get_obs_embedding(obs)

        # Sample continuous
        # -- Flatten discrete latents
        raw_expression_flattened = raw_expression.reshape(-1, self.max_num_chars)
        eos_flattened = eos.reshape(-1, self.max_num_chars)

        # -- Compute num base kernels
        # [num_discrete_elements * num_elements]
        num_base_kernels_flattened = self.get_num_base_kernels(
            raw_expression_flattened, eos_flattened
        )

        # -- Expand num base kernels
        num_base_kernels_expanded = num_base_kernels_flattened[None].expand(num_samples, -1)

        # -- Compute expression embedding
        # [num_discrete_elements * num_elements, expression_embedding_dim]
        expression_embedding_flattened = self.get_expression_embedding(
            raw_expression_flattened, eos_flattened
        )

        # -- Expand obs embedding
        # [num_discrete_elements * num_elements, obs_embedding_dim]
        obs_embedding_expanded = (
            obs_embedding.view(1, num_elements, -1)
            .expand(num_discrete_elements, num_elements, -1)
            .reshape(-1, self.obs_embedding_dim)
        )

        # -- Sample
        raw_gp_params_dist = lstm_util.TimeseriesDistribution(
            "continuous",
            timeseries_util.gp_params_dim,
            torch.cat([obs_embedding_expanded, expression_embedding_flattened], dim=1),
            self.gp_params_lstm,
            self.gp_params_extractor,
            lstm_eos=False,
            max_num_timesteps=self.max_num_chars,
        )
        if reparam:
            raw_gp_params = raw_gp_params_dist.rsample(
                (num_samples,), num_timesteps=num_base_kernels_expanded
            )
        else:
            raw_gp_params = raw_gp_params_dist.sample(
                (num_samples,), num_timesteps=num_base_kernels_expanded
            )
        return raw_gp_params.view(
            *[
                *sample_shape,
                *discrete_shape,
                *shape,
                self.max_num_chars,
                timeseries_util.gp_params_dim,
            ]
        )

    def sample_continuous(self, obs, discrete_latent, sample_shape=[]):
        """z_c ~ q(z_c | z_d, x) (NOT reparameterized)

        Args
            obs [*shape, num_timesteps]
            discrete_latent
                raw_expression [*discrete_shape, *shape, max_num_chars]
                eos [*discrete_shape, *shape, max_num_chars]
            sample_shape

        Returns
            raw_gp_params [*sample_shape, *discrete_shape, *shape, max_num_chars, gp_params_dim]
        """
        return self._sample_continuous(False, obs, discrete_latent, sample_shape=sample_shape)

    def rsample_continuous(self, obs, discrete_latent, sample_shape=[]):
        """z_c ~ q(z_c | z_d, x) (reparameterized)

        Args
            obs [*shape, num_timesteps]
            discrete_latent
                raw_expression [*discrete_shape, *shape, max_num_chars]
                eos [*discrete_shape, *shape, max_num_chars]
            sample_shape

        Returns
            raw_gp_params [*sample_shape, *discrete_shape, *shape, max_num_chars, gp_params_dim]
        """
        return self._sample_continuous(True, obs, discrete_latent, sample_shape=sample_shape)

    def log_prob_discrete(self, obs, discrete_latent):
        """log q(z_d | x)

        Args
            obs [*shape, num_timesteps]
            discrete_latent
                raw_expression [*discrete_shape, *shape, max_num_chars]
                eos [*discrete_shape, *shape, max_num_chars]

        Returns [*discrete_shape, *shape]
        """
        # Extract
        raw_expression, eos = discrete_latent
        shape = obs.shape[:-1]
        discrete_shape = raw_expression.shape[: -(len(shape) + 1)]
        num_elements = cmws.util.get_num_elements(shape)
        num_discrete_elements = cmws.util.get_num_elements(discrete_shape)

        # Compute obs embedding
        obs_embedding = self.get_obs_embedding(obs)

        # Log prob of discrete
        # -- Expand obs embedding
        # [num_discrete_elements * num_elements, obs_embedding_dim]
        obs_embedding_expanded = (
            obs_embedding[None]
            .expand(*[num_discrete_elements, *shape, -1])
            .reshape(num_discrete_elements * num_elements, -1)
        )
        # -- Flatten discrete latents
        raw_expression_flattened = raw_expression.view(-1, self.max_num_chars)
        eos_flattened = eos.view(-1, self.max_num_chars)

        return (
            lstm_util.TimeseriesDistribution(
                "discrete",
                timeseries_util.vocabulary_size,
                obs_embedding_expanded,
                self.expression_lstm,
                self.expression_extractor,
                lstm_eos=True,
                max_num_timesteps=self.max_num_chars,
            )
            .log_prob(raw_expression_flattened, eos_flattened)
            .view([*discrete_shape, *shape])
        )

    def log_prob_continuous(self, obs, discrete_latent, continuous_latent):
        """log q(z_c | z_d, x)

        Args
            obs [*shape, num_timesteps]
            discrete_latent
                raw_expression [*discrete_shape, *shape, max_num_chars]
                eos [*discrete_shape, *shape, max_num_chars]

            continuous_latent (raw_gp_params)
                [*continuous_shape, *discrete_shape, *shape, max_num_chars, gp_params_dim]

        Returns [*continuous_shape, *discrete_shape, *shape]
        """
        # Extract
        raw_expression, eos = discrete_latent
        raw_gp_params = continuous_latent
        shape = obs.shape[:-1]
        discrete_shape = raw_expression.shape[: -(len(shape) + 1)]
        continuous_shape = raw_gp_params.shape[: -(len(discrete_shape) + len(shape) + 2)]
        num_elements = cmws.util.get_num_elements(shape)
        num_discrete_elements = cmws.util.get_num_elements(discrete_shape)
        num_continuous_elements = cmws.util.get_num_elements(continuous_shape)

        # Compute obs embedding
        obs_embedding = self.get_obs_embedding(obs)

        # Log prob of continuous
        # -- Expand obs embedding
        # [num_continuous_elements * num_discrete_elements * num_elements, obs_embedding_dim]
        obs_embedding_expanded = (
            obs_embedding[None]
            .expand(*[num_continuous_elements * num_discrete_elements, *shape, -1])
            .reshape(num_continuous_elements * num_discrete_elements * num_elements, -1)
        )
        # -- Flatten discrete latents
        # [num_continuous_elements * num_discrete_elements * num_elements, max_num_chars]
        raw_expression_flattened = (
            raw_expression[None]
            .expand(*[num_continuous_elements, *discrete_shape, *shape, -1])
            .reshape(-1, self.max_num_chars)
        )
        eos_flattened = (
            eos[None]
            .expand(*[num_continuous_elements, *discrete_shape, *shape, -1])
            .reshape(-1, self.max_num_chars)
        )

        # -- Compute num base kernels
        # [num_continuous_elements * num_discrete_elements * num_elements]
        num_base_kernels_flattened = self.get_num_base_kernels(
            raw_expression_flattened, eos_flattened
        )

        # -- Compute expression embedding
        # [num_continuous_elements * num_discrete_elements * num_elements, expression_embedding_dim]
        expression_embedding_flattened = self.get_expression_embedding(
            raw_expression_flattened, eos_flattened
        )

        # -- Flatten continuous latent
        # [num_continuous_elements * num_discrete_elements * num_elements, max_num_chars,
        #  gp_params_dim]
        raw_gp_params_flattened = raw_gp_params.view(
            -1, self.max_num_chars, timeseries_util.gp_params_dim
        )

        # -- Compute log prob
        return (
            lstm_util.TimeseriesDistribution(
                "continuous",
                timeseries_util.gp_params_dim,
                torch.cat([obs_embedding_expanded, expression_embedding_flattened], dim=1),
                self.gp_params_lstm,
                self.gp_params_extractor,
                lstm_eos=False,
                max_num_timesteps=self.max_num_chars,
            )
            .log_prob(raw_gp_params_flattened, num_timesteps=num_base_kernels_flattened)
            .view([*continuous_shape, *discrete_shape, *shape])
        )
