import torch
import torch.nn as nn


def step_lstm(lstm, input_, h_0_c_0=None):
    """LSTMCell-like API for LSTM.
    Args:
        lstm: nn.LSTM
        input_: [batch_size, input_size]
        h_0_c_0: None or
            h_0: [num_layers, batch_size, hidden_size]
            c_0: [num_layers, batch_size, hidden_size]
    Returns:
        output: [batch_size, hidden_size]
        h_1_c_1:
            h_1: [num_layers, batch_size, hidden_size]
            c_1: [num_layers, batch_size, hidden_size]
    """
    output, h_1_c_1 = lstm(input_[None], h_0_c_0)
    return output[0], h_1_c_1


class TimeseriesDistribution:
    """p(x_{1:T}, eos_{1:T} | embedding) where x_t ∈ R

    Args
        embedding: [batch_size, embedding_dim]
        lstm: nn.Module object
        linear: nn.Module mapping from LSTM hidden state to per-timestep distribution params
        lstm_eos: bool; models length if True
        num_mixtures (int; default 1)
    Returns distribution with batch_shape [batch_size] and event_shape [num_timesteps]
        where num_timesteps is variable
    """

    def __init__(
        self, embedding, lstm, linear, lstm_eos, num_mixtures=1, max_num_timesteps_sample=200,
    ):
        self.lstm = lstm
        self.linear = linear
        self.lstm_eos = lstm_eos
        self.num_mixtures = num_mixtures
        self.max_num_timesteps_sample = max_num_timesteps_sample
        self.embedding = embedding
        self.batch_size = self.embedding.shape[0]

    def sample(self, sample_shape=torch.Size(), num_timesteps=None, warning_enabled=True):
        """
        Args:
            sample_shape:
            num_timesteps: None or [*sample_shape, batch_size]
                NOTE: if provided, will sample for num_timesteps regardless of whether
                lstm_eos is True or not. if not provided, the length will be dicated by
                the end-of-signal indicator.
        Returns:
            x: list of length num_samples * batch_size where
                x[i] is a tensor [num_timesteps_i] and
                num_samples = prod(*sample_shape)
        """
        device = self.embedding.device
        num_samples = torch.tensor(sample_shape).prod().long().item()
        if num_timesteps is None:  # use EOS to end
            if not self.lstm_eos:
                raise RuntimeError(
                    "num_timesteps is not given and LSTM doesn't support end-of-signal indicator "
                    "sampling. Don't know how end sampling."
                )
            max_num_timesteps = self.max_num_timesteps_sample
        else:  # use num_timesteps to end
            if self.lstm_eos and warning_enabled:
                print(
                    "WARNING: num_timesteps is given and LSTM supports end-of-signal indicator."
                    " Sampling will ignore EOS and end based on num_timesteps."
                )
            max_num_timesteps = num_timesteps.max()

        # [num_samples * batch_size, 2 * cluster_embedding_dim]
        embedding_expanded = (
            self.embedding[None]
            .expand(num_samples, self.batch_size, -1)
            .contiguous()
            .view(num_samples * self.batch_size, -1)
        )

        # will be a list of length max_num_timesteps
        x = []
        if num_timesteps is None:  # use EOS to end
            eos = []

        # [num_samples * batch_size, 1 + 2 * cluster_embedding_dim]
        lstm_input = torch.cat(
            [torch.zeros((num_samples * self.batch_size, 1), device=device), embedding_expanded],
            dim=1,
        )
        hc = None
        for timestep in range(max_num_timesteps):
            # [num_samples * batch_size, lstm_hidden_size], ([...], [...])
            lstm_output, hc = step_lstm(self.lstm, lstm_input, hc)

            # if num_mixtures == 1: [num_samples * batch_size, 2 + 1 or 2]
            # else: [num_samples * batch_size, 3 * num_mixtures + 1 or 3 * num_mixtures]
            linear_out = self.linear(lstm_output)

            if self.num_mixtures == 1:
                # [num_samples * batch_size]
                x_loc = linear_out[..., 0]
                x_scale = linear_out[..., 1].exp()

                # batch_shape [num_samples * batch_size], event_shape []
                x_dist = torch.distributions.Normal(loc=x_loc, scale=x_scale)
            else:
                # [num_samples * batch_size, num_mixtures]
                x_mixture_logits = linear_out[..., : self.num_mixtures]
                # [num_samples * batch_size, num_mixtures]
                x_loc = linear_out[..., self.num_mixtures : 2 * self.num_mixtures]
                # [num_samples * batch_size, num_mixtures]
                x_scale = linear_out[..., 2 * self.num_mixtures : 3 * self.num_mixtures].exp()

                # batch_shape [num_samples * batch_size], event_shape []
                x_dist = torch.distributions.MixtureSameFamily(
                    torch.distributions.Categorical(logits=x_mixture_logits),
                    torch.distributions.Normal(x_loc, scale=x_scale),
                )
            # sample
            # [num_samples * batch_size]
            x.append(x_dist.sample())

            if num_timesteps is None:  # use EOS to end
                eos_logit = linear_out[..., -1]
                # [num_samples * batch_size]
                eos.append(torch.distributions.Bernoulli(logits=eos_logit).sample())

            # assemble lstm_input
            # [num_samples * batch_size, 1 + 2 * cluster_embedding_dim]
            lstm_input = torch.cat([x[-1][:, None], embedding_expanded], dim=1)

            if num_timesteps is None:  # use EOS to end
                # end early if all eos are 1
                if (torch.stack(eos).sum(0) > 0).all():
                    max_num_timesteps = timestep + 1
                    break

        # [max_num_timesteps, num_samples, batch_size]
        x = torch.stack(x).view(max_num_timesteps, num_samples, self.batch_size)
        if num_timesteps is None:  # use EOS to end
            # [max_num_timesteps, num_samples, batch_size]
            eos = torch.stack(eos).view(max_num_timesteps, num_samples, self.batch_size)

        # thing to be returned
        x_final = []
        for sample_id in range(num_samples):
            for batch_id in range(self.batch_size):
                if num_timesteps is None:  # use EOS to end
                    # [max_num_timesteps]
                    eos_single = eos[:, sample_id, batch_id]
                    if (eos_single == 0).all():
                        num_timesteps_sb = len(eos_single)
                    else:
                        num_timesteps_sb = (eos_single == 1).nonzero(as_tuple=False)[0].item() + 1
                else:  # use num_timesteps to end
                    num_timesteps_sb = num_timesteps.view(num_samples, self.batch_size)[
                        sample_id, batch_id
                    ]

                result = x[:num_timesteps_sb, sample_id, batch_id]
                x_final.append(result)

        return x_final

    def log_prob(self, x):
        """
        Args:
            x: list of length batch_size
                x[b] is a tensor [num_timesteps_b]
        Returns: tensor [batch_size]
        """
        device = self.embedding.device
        batch_size = len(x)

        # Downsample x
        downsampled_x = []

        # Build LSTM input
        # [batch_size]
        zero_length_x = torch.zeros((batch_size,), device=device, dtype=torch.bool)
        lstm_input = []
        for batch_id in range(batch_size):
            if len(x[batch_id]) == 0:
                zero_length_x[batch_id] = True
                x_b = torch.tensor([0.0], device=device)
                # print(f"Warning: x[{batch_id}] has length 0. Setting its log_prob to 0.")
            else:
                x_b = x[batch_id]

            downsampled_x.append(x_b)

            num_timesteps = len(x_b)
            # [num_timesteps, cluster_embedding_dim]
            embedding_expanded = self.embedding[batch_id][None].expand(num_timesteps, -1)
            # [num_timesteps]
            shifted_x = torch.cat([torch.tensor([0.0], device=device), x_b[:-1]])
            # [num_timesteps, 1 (x_t) + cluster_embedding_dim]
            lstm_input.append(torch.cat([shifted_x[:, None], embedding_expanded], dim=1))
        lstm_input = nn.utils.rnn.pack_sequence(lstm_input, enforce_sorted=False)

        # Run LSTM
        # [*, batch_size, lstm_hidden_size]
        output, _ = self.lstm(lstm_input)
        # [max_num_timesteps, batch_size, lstm_hidden_size]
        padded_output, _ = nn.utils.rnn.pad_packed_sequence(output)
        max_num_timesteps = padded_output.shape[0]

        # Extract distribution params for x_t
        # if num_mixtures == 1: [max_num_timesteps, batch_size, 2 + 1 or 2]
        # else: [max_num_timesteps, batch_size, 3 * num_mixtures + 1 or 3 * num_mixtures]
        linear_out = self.linear(padded_output.view(max_num_timesteps * batch_size, -1)).view(
            max_num_timesteps, batch_size, -1
        )

        if self.num_mixtures == 1:
            # [max_num_timesteps, batch_size]
            x_loc = linear_out[..., 0]
            # [max_num_timesteps, batch_size]
            x_scale = linear_out[..., 1].exp()

            # batch_shape [max_num_timesteps, batch_size], event_shape []
            x_dist = torch.distributions.Normal(x_loc, scale=x_scale)
        else:
            # [max_num_timesteps, batch_size, num_mixtures]
            x_mixture_logits = linear_out[..., : self.num_mixtures]
            # [max_num_timesteps, batch_size, num_mixtures]
            x_loc = linear_out[..., self.num_mixtures : 2 * self.num_mixtures]
            # [max_num_timesteps, batch_size, num_mixtures]
            x_scale = linear_out[..., 2 * self.num_mixtures : 3 * self.num_mixtures].exp()

            # batch_shape [max_num_timesteps, batch_size], event_shape []
            x_dist = torch.distributions.MixtureSameFamily(
                torch.distributions.Categorical(logits=x_mixture_logits),
                torch.distributions.Normal(x_loc, scale=x_scale),
            )

        # Evaluate log prob
        # [max_num_timesteps, batch_size]
        x_padded = nn.utils.rnn.pad_sequence(downsampled_x)
        # [max_num_timesteps, batch_size]
        x_log_prob = x_dist.log_prob(x_padded)

        if self.lstm_eos:
            # [max_num_timesteps, batch_size]
            eos_logit = linear_out[..., -1]

            eos_padded = []
            for batch_id in range(batch_size):
                num_timesteps = len(downsampled_x[batch_id])
                if num_timesteps == 0:
                    eos = torch.tensor([1.0], device=device)
                else:
                    eos = torch.zeros((num_timesteps,), device=device)
                    eos[-1] = 1.0
                eos_padded.append(eos)
            # [max_num_timesteps, batch_size]
            eos_padded = nn.utils.rnn.pad_sequence(eos_padded)
            # [max_num_timesteps, batch_size]
            eos_log_prob = torch.distributions.Bernoulli(logits=eos_logit).log_prob(eos_padded)

        # this will be a list of length bath_size
        log_prob = []
        for batch_id in range(batch_size):
            num_timesteps = len(downsampled_x[batch_id])
            if self.lstm_eos:
                log_prob_b = (x_log_prob[:, batch_id] + eos_log_prob[:, batch_id])[
                    :num_timesteps
                ].sum()
            else:
                log_prob_b = x_log_prob[:, batch_id][:num_timesteps].sum()
            log_prob.append(log_prob_b)
        return torch.stack(log_prob) * (1 - zero_length_x.float())
