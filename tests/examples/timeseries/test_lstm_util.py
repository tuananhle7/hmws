import torch
import torch.nn as nn
import cmws.examples.timeseries.lstm_util as lstm_util
import cmws


def test_timeseries_distribution_shape():
    device = cmws.util.get_device()
    batch_size, embedding_dim, hidden_dim = 2, 3, 4
    input_dim = 1 + embedding_dim
    linear_out_dim = 2 + 1
    sample_shape = [5, 6]
    max_num_timesteps = 7

    # Timeseries Distribution args
    lstm = nn.LSTM(input_dim, hidden_dim).to(device)
    linear = nn.Linear(hidden_dim, linear_out_dim).to(device)
    lstm_eos = True

    # Sample
    embedding = torch.randn(batch_size, embedding_dim, device=device)
    timeseries_distribution = lstm_util.TimeseriesDistribution(
        embedding, lstm, linear, lstm_eos, max_num_timesteps
    )
    x, eos = timeseries_distribution.sample(sample_shape)
    assert list(x.shape) == sample_shape + [batch_size, max_num_timesteps]
    assert list(eos.shape) == sample_shape + [batch_size, max_num_timesteps]

    # Log prob
    x = x.view(-1, max_num_timesteps)
    eos = eos.view(-1, max_num_timesteps)
    batch_size = x.shape[0]
    embedding = torch.randn(batch_size, embedding_dim, device=device)
    timeseries_distribution = lstm_util.TimeseriesDistribution(
        embedding, lstm, linear, lstm_eos, max_num_timesteps
    )
    lp = timeseries_distribution.log_prob(x, eos)
    assert list(lp.shape) == [batch_size]
