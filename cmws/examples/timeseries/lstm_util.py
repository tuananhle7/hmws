import torch
import torch.nn as nn
import torch.nn.functional as F
import cmws.util


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


def get_num_timesteps(eos):
    """
    Args
        eos [*shape, max_num_timesteps]

    Returns [*shape]
    """
    # Extract
    shape = eos.shape[:-1]
    max_num_timesteps = eos.shape[-1]
    num_elements = cmws.util.get_num_elements(shape)
    device = eos.device

    # Flatten
    eos_flattened = eos.reshape(-1, max_num_timesteps)

    num_timesteps = []
    for element_id in range(num_elements):
        if torch.all(eos_flattened[element_id] == 0):
            num_timesteps.append(max_num_timesteps)
        else:
            num_timesteps.append(
                (eos_flattened[element_id] == 1).nonzero(as_tuple=False)[0].item() + 1
            )
    return torch.tensor(num_timesteps, device=device).view(*shape)


class TimeseriesDistribution:
    """p(x_{1:T}, eos_{1:T} | embedding) where x_t ∈ Z or R^d
    batch_shape [batch_size]
                or
                []
    and
    event_shape ([max_num_timesteps(, x_dim)], [max_num_timesteps])
                or
                [max_num_timesteps(, x_dim)]

    Args
        x_type (str) continuous or discrete
        num_classes_or_dim (int)
        embedding: [batch_size, embedding_dim] or None
        lstm: nn.Module object
        linear: nn.Module mapping from LSTM hidden state to per-timestep distribution params
        lstm_eos: bool; models length if True
        max_num_timesteps (int; default 200)
    Returns distribution with batch_shape [batch_size] and event_shape [num_timesteps]
        where num_timesteps is variable
    """

    def __init__(
        self, x_type, num_classes_or_dim, embedding, lstm, linear, lstm_eos, max_num_timesteps=200,
    ):
        self.x_type = x_type
        self.num_classes_or_dim = num_classes_or_dim
        if x_type == "continuous":
            self.x_dim = num_classes_or_dim
        elif x_type == "discrete":
            self.num_classes = num_classes_or_dim
        else:
            raise ValueError()
        self.lstm = lstm
        self.linear = linear
        self.lstm_eos = lstm_eos
        self.max_num_timesteps = max_num_timesteps
        self.embedding = embedding
        if self.embedding is not None:
            self.batch_size = self.embedding.shape[0]

    @property
    def device(self):
        return next(iter(self.lstm.parameters())).device

    def _sample(self, reparam, sample_shape=torch.Size(), num_timesteps=None, warning_enabled=True):
        """
        Args:
            reparam (bool)
            sample_shape:
            num_timesteps: None or [*sample_shape, batch_size]
                NOTE: if provided, will sample for num_timesteps regardless of whether
                lstm_eos is True or not. if not provided, the length will be dicated by
                the end-of-signal indicator.
        Returns:
            x [*sample_shape(, batch_size), max_num_timesteps(, x_dim)]

            Optionally
            eos [*sample_shape(, batch_size), max_num_timesteps]
        """
        if reparam:
            assert self.x_type == "continuous"
        num_samples = cmws.util.get_num_elements(sample_shape)
        if num_timesteps is None:  # use EOS to end
            if not self.lstm_eos:
                raise RuntimeError(
                    "num_timesteps is not given and LSTM doesn't support end-of-signal indicator "
                    "sampling. Don't know how end sampling."
                )
        else:  # use num_timesteps to end
            if self.lstm_eos and warning_enabled:
                print(
                    "WARNING: num_timesteps is given and LSTM supports end-of-signal indicator."
                    " Sampling will ignore EOS and end based on num_timesteps."
                )

        if self.embedding is not None:
            # [num_samples * batch_size, embedding_dim]
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

        if self.embedding is not None:
            # [num_samples * batch_size, num_classes_or_dim + embedding_dim]
            lstm_input = torch.cat(
                [
                    torch.zeros(
                        (num_samples * self.batch_size, self.num_classes_or_dim),
                        device=self.device,
                    ),
                    embedding_expanded,
                ],
                dim=1,
            )
        else:
            # [num_samples, num_classes_or_dim + embedding_dim]
            lstm_input = torch.zeros((num_samples, self.num_classes_or_dim), device=self.device,)
        hc = None
        for timestep in range(self.max_num_timesteps):
            # [num_samples( * batch_size), lstm_hidden_size], ([...], [...])
            lstm_output, hc = step_lstm(self.lstm, lstm_input, hc)

            # [num_samples( * batch_size), num_classes + 1 or num_classes]
            linear_out = self.linear(lstm_output)

            if self.x_type == "discrete":
                # [num_samples( * batch_size), num_classes]
                x_logits = linear_out[..., : self.num_classes]

                # batch_shape [num_samples( * batch_size)], event_shape []
                x_dist = torch.distributions.Categorical(logits=x_logits)
            elif self.x_type == "continuous":
                # [num_samples( * batch_size), x_dim]
                x_loc = linear_out[..., : self.x_dim]

                # [num_samples( * batch_size), x_dim]
                x_scale = linear_out[..., self.x_dim : 2 * self.x_dim].exp()

                # batch_shape [num_samples( * batch_size)], event_shape [x_dim]
                x_dist = torch.distributions.Independent(
                    torch.distributions.Normal(loc=x_loc, scale=x_scale),
                    reinterpreted_batch_ndims=1,
                )

            # sample
            # [num_samples( * batch_size)] or [num_samples( * batch_size), x_dim]
            if reparam:
                x.append(x_dist.rsample())
            else:
                x.append(x_dist.sample())

            if num_timesteps is None:  # use EOS to end
                eos_logit = linear_out[..., -1]
                # [num_samples( * batch_size)]
                eos.append(torch.distributions.Bernoulli(logits=eos_logit).sample())

            # assemble lstm_input
            if self.x_type == "discrete":
                if self.embedding is not None:
                    # [num_samples * batch_size, num_classes + embedding_dim]
                    lstm_input = torch.cat(
                        [F.one_hot(x[-1], num_classes=self.num_classes), embedding_expanded], dim=1
                    )
                else:
                    # [num_samples, num_classes + embedding_dim]
                    lstm_input = F.one_hot(x[-1], num_classes=self.num_classes).float()
            elif self.x_type == "continuous":
                if self.embedding is not None:
                    # [num_samples * batch_size, x_dim + embedding_dim]
                    lstm_input = torch.cat([x[-1], embedding_expanded], dim=1)
                else:
                    # [num_samples, x_dim + embedding_dim]
                    lstm_input = x[-1]

            # if num_timesteps is None:  # use EOS to end
            #     # end early if all eos are 1
            #     if (torch.stack(eos).sum(0) > 0).all():
            #         # max_num_timesteps = timestep + 1
            #         break

        if self.embedding is not None:
            if self.x_type == "discrete":
                # [max_num_timesteps, num_samples, batch_size]
                x = torch.stack(x).view(self.max_num_timesteps, num_samples, self.batch_size)
            elif self.x_type == "continuous":
                # [max_num_timesteps, num_samples, batch_size, x_dim]
                x = torch.stack(x).view(
                    self.max_num_timesteps, num_samples, self.batch_size, self.x_dim
                )

            if num_timesteps is None:  # use EOS to end
                # [max_num_timesteps, num_samples, batch_size]
                eos = torch.stack(eos).view(self.max_num_timesteps, num_samples, self.batch_size)

            # thing to be returned
            x_final = []
            eos_final = []
            for sample_id in range(num_samples):
                for batch_id in range(self.batch_size):
                    if num_timesteps is None:  # use EOS to end
                        # [max_num_timesteps]
                        eos_single = eos[:, sample_id, batch_id]
                        if (eos_single == 0).all():
                            num_timesteps_sb = len(eos_single)
                        else:
                            num_timesteps_sb = (eos_single == 1).nonzero(as_tuple=False)[
                                0
                            ].item() + 1

                        eos_final.append(
                            F.one_hot(
                                torch.tensor(num_timesteps_sb - 1, device=self.device).long(),
                                num_classes=self.max_num_timesteps,
                            )
                        )
                    else:  # use num_timesteps to end
                        num_timesteps_sb = num_timesteps.view(num_samples, self.batch_size)[
                            sample_id, batch_id
                        ]

                    if self.x_type == "discrete":
                        result = torch.zeros((self.max_num_timesteps,), device=self.device)
                        result[:num_timesteps_sb] = x[:num_timesteps_sb, sample_id, batch_id]
                    elif self.x_type == "continuous":
                        result = torch.zeros(
                            (self.max_num_timesteps, self.x_dim), device=self.device
                        )
                        result[:num_timesteps_sb] = x[:num_timesteps_sb, sample_id, batch_id]
                    x_final.append(result)

            if self.x_type == "discrete":
                x_final = (
                    torch.stack(x_final)
                    .view(*[*sample_shape, self.batch_size, self.max_num_timesteps])
                    .long()
                )
            elif self.x_type == "continuous":
                x_final = torch.stack(x_final).view(
                    *[*sample_shape, self.batch_size, self.max_num_timesteps, self.x_dim]
                )
            if num_timesteps is None:
                eos_final = (
                    torch.stack(eos_final).view(*[*sample_shape, self.batch_size, -1]).long()
                )
                return x_final, eos_final
            else:
                return x_final
        else:
            if self.x_type == "discrete":
                # [max_num_timesteps, num_samples]
                x = torch.stack(x).view(self.max_num_timesteps, num_samples)
            elif self.x_type == "continuous":
                # [max_num_timesteps, num_samples, x_dim]
                x = torch.stack(x).view(self.max_num_timesteps, num_samples, self.x_dim)

            if num_timesteps is None:  # use EOS to end
                # [max_num_timesteps, num_samples]
                eos = torch.stack(eos).view(self.max_num_timesteps, num_samples)

            # thing to be returned
            x_final = []
            eos_final = []
            for sample_id in range(num_samples):
                if num_timesteps is None:  # use EOS to end
                    # [max_num_timesteps]
                    eos_single = eos[:, sample_id]
                    if (eos_single == 0).all():
                        num_timesteps_sb = len(eos_single)
                    else:
                        num_timesteps_sb = (eos_single == 1).nonzero(as_tuple=False)[0].item() + 1

                    eos_final.append(
                        F.one_hot(
                            torch.tensor(num_timesteps_sb - 1, device=self.device).long(),
                            num_classes=self.max_num_timesteps,
                        )
                    )
                else:  # use num_timesteps to end
                    num_timesteps_sb = num_timesteps.view(num_samples)[sample_id]

                if self.x_type == "discrete":
                    result = torch.zeros((self.max_num_timesteps,), device=self.device)
                    result[:num_timesteps_sb] = x[:num_timesteps_sb, sample_id]
                elif self.x_type == "continuous":
                    result = torch.zeros((self.max_num_timesteps, self.x_dim), device=self.device)
                    result[:num_timesteps_sb] = x[:num_timesteps_sb, sample_id]
                x_final.append(result)

            if self.x_type == "discrete":
                x_final = torch.stack(x_final).view(*[*sample_shape, self.max_num_timesteps]).long()
            elif self.x_type == "continuous":
                x_final = torch.stack(x_final).view(
                    *[*sample_shape, self.max_num_timesteps, self.x_dim]
                )
            if num_timesteps is None:
                eos_final = torch.stack(eos_final).view(*[*sample_shape, -1]).long()
                return x_final, eos_final
            else:
                return x_final

    def sample(self, sample_shape=torch.Size(), num_timesteps=None, warning_enabled=True):
        """NOT reparameterized

        Args:
            sample_shape:
            num_timesteps: None or [*sample_shape, batch_size]
                NOTE: if provided, will sample for num_timesteps regardless of whether
                lstm_eos is True or not. if not provided, the length will be dicated by
                the end-of-signal indicator.
        Returns:
            x [*sample_shape(, batch_size), max_num_timesteps(, x_dim)]

            Optionally
            eos [*sample_shape(, batch_size), max_num_timesteps]
        """
        return self._sample(
            False,
            sample_shape=sample_shape,
            num_timesteps=num_timesteps,
            warning_enabled=warning_enabled,
        )

    def rsample(self, sample_shape=torch.Size(), num_timesteps=None, warning_enabled=True):
        """Reparameterized

        Args:
            sample_shape:
            num_timesteps: None or [*sample_shape, batch_size]
                NOTE: if provided, will sample for num_timesteps regardless of whether
                lstm_eos is True or not. if not provided, the length will be dicated by
                the end-of-signal indicator.
        Returns:
            x [*sample_shape(, batch_size), max_num_timesteps(, x_dim)]

            Optionally
            eos [*sample_shape(, batch_size), max_num_timesteps]
        """
        return self._sample(
            True,
            sample_shape=sample_shape,
            num_timesteps=num_timesteps,
            warning_enabled=warning_enabled,
        )

    def log_prob(self, x, eos=None, num_timesteps=None):
        """
        Args:
            x [batch_size, max_num_timesteps(, x_dim)]

            Optional
            eos [batch_size, max_num_timesteps]
            num_timesteps [batch_size]

        Returns: tensor [batch_size]
        """
        batch_size = len(x)
        if self.embedding is not None:
            assert batch_size == self.batch_size

        if num_timesteps is None:
            num_timesteps = get_num_timesteps(eos)

        # Downsample x
        x_seq = []

        # Build LSTM input
        # [batch_size]
        zero_length_x = torch.zeros((batch_size,), device=self.device, dtype=torch.bool)
        lstm_input = []
        for batch_id in range(batch_size):
            num_timesteps_b = num_timesteps[batch_id]
            if num_timesteps_b == 0:
                zero_length_x[batch_id] = True
                if self.x_type == "discrete":
                    x_b = torch.tensor([0.0], device=self.device)
                elif self.x_type == "continuous":
                    x_b = torch.zeros((1, self.x_dim), device=self.device)
                num_timesteps_b += 1
                # print(f"Warning: x[{batch_id}] has length 0. Setting its log_prob to 0.")
            else:
                x_b = x[batch_id, :num_timesteps_b]

            x_seq.append(x_b)

            if self.embedding is not None:
                # [num_timesteps_b, embedding_dim]
                embedding_expanded = self.embedding[batch_id][None].expand(num_timesteps_b, -1)
            if self.x_type == "discrete":
                # [num_timesteps_b, num_classes]
                shifted_x = torch.cat(
                    [
                        torch.zeros((1, self.num_classes), device=self.device),
                        F.one_hot(x_b[:-1].long(), num_classes=self.num_classes),
                    ]
                )
            elif self.x_type == "continuous":
                # [num_timesteps_b, x_dim]
                shifted_x = torch.cat([torch.zeros((1, self.x_dim), device=self.device), x_b[:-1]])

            if self.embedding is not None:
                # [num_timesteps_b, num_classes_or_dim + embedding_dim]
                lstm_input.append(torch.cat([shifted_x, embedding_expanded], dim=1))
            else:
                # [num_timesteps_b, num_classes_or_dim]
                lstm_input.append(shifted_x)
        lstm_input = nn.utils.rnn.pack_sequence(lstm_input, enforce_sorted=False)

        # Run LSTM
        # [*, batch_size, lstm_hidden_size]
        output, _ = self.lstm(lstm_input)
        # [max_num_timesteps, batch_size, lstm_hidden_size]
        padded_output, _ = nn.utils.rnn.pad_packed_sequence(
            output, total_length=self.max_num_timesteps
        )

        # Extract distribution params for x_t
        # [max_num_timesteps, batch_size, num_classes or 2*x_dim + 1 or num_classes or 2*x_dim]
        linear_out = self.linear(padded_output.view(self.max_num_timesteps * batch_size, -1)).view(
            self.max_num_timesteps, batch_size, -1
        )

        if self.x_type == "discrete":
            # [max_num_timesteps, batch_size, num_classes]
            x_logits = linear_out[..., : self.num_classes]

            # batch_shape [max_num_timesteps, batch_size], event_shape []
            x_dist = torch.distributions.Categorical(logits=x_logits)
        elif self.x_type == "continuous":
            # [max_num_timesteps, batch_size, x_dim]
            x_loc = linear_out[..., : self.x_dim]
            # [max_num_timesteps, batch_size, x_dim]
            x_scale = linear_out[..., self.x_dim : 2 * self.x_dim].exp()

            # batch_shape [max_num_timesteps, batch_size], event_shape [x_dim]
            x_dist = torch.distributions.Independent(
                torch.distributions.Normal(loc=x_loc, scale=x_scale), reinterpreted_batch_ndims=1
            )

        # Evaluate log prob
        if self.x_type == "discrete":
            # [max_num_timesteps, batch_size]
            x_log_prob = x_dist.log_prob(x.T)
        elif self.x_type == "continuous":
            # [max_num_timesteps, batch_size]
            x_log_prob = x_dist.log_prob(x.permute(1, 0, 2))

        if self.lstm_eos:
            # [max_num_timesteps, batch_size]
            eos_logit = linear_out[..., -1]

            # [max_num_timesteps, batch_size]
            eos_log_prob = torch.distributions.Bernoulli(logits=eos_logit).log_prob(eos.T.float())

        # this will be a list of length batch_size
        log_prob = []
        for batch_id in range(batch_size):
            num_timesteps_b = num_timesteps[batch_id]
            if self.lstm_eos:
                log_prob_b = (x_log_prob[:, batch_id] + eos_log_prob[:, batch_id])[
                    :num_timesteps_b
                ].sum()
            else:
                log_prob_b = x_log_prob[:, batch_id][:num_timesteps_b].sum()
            log_prob.append(log_prob_b)
        return torch.stack(log_prob) * (1 - zero_length_x.float())
