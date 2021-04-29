import torch
import torch.nn as nn
import cmws.examples.timeseries.lstm_util as lstm_util
import cmws.util


def test_timeseries_distribution_shape():
    device = "cpu"
    batch_size, embedding_dim, hidden_dim = 2, 3, 4
    input_dim = 1 + embedding_dim
    linear_out_dim = 2 + 1
    sample_shape = (5, 6)
    num_samples = cmws.util.get_num_elements(sample_shape)

    # Timeseries Distribution args
    lstm = nn.LSTM(input_dim, hidden_dim).to(device)
    linear = nn.Linear(hidden_dim, linear_out_dim).to(device)
    lstm_eos = True

    # Sample
    embedding = torch.randn(batch_size, embedding_dim, device=device)
    timeseries_distribution = lstm_util.TimeseriesDistribution(embedding, lstm, linear, lstm_eos)
    x = timeseries_distribution.sample(sample_shape)
    assert len(x) == num_samples * batch_size

    # Log prob
    batch_size = len(x)
    embedding = torch.randn(batch_size, embedding_dim, device=device)
    timeseries_distribution = lstm_util.TimeseriesDistribution(embedding, lstm, linear, lstm_eos)
    lp = timeseries_distribution.log_prob(x)
    assert list(lp.shape) == [batch_size]
