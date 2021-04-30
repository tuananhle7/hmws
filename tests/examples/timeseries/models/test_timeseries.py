import torch
from cmws.examples.timeseries.models.timeseries import GenerativeModel
import cmws.examples.timeseries.util as timeseries_util


def test_generative_model_latent_log_prob_dims():
    shape = [2, 3]
    max_num_chars, lstm_hidden_dim = 4, 5

    generative_model = GenerativeModel(max_num_chars, lstm_hidden_dim)
    raw_expression = torch.randint(timeseries_util.vocabulary_size, size=shape + [max_num_chars])
    eos = torch.randint(2, size=shape + [max_num_chars])
    raw_gp_params = torch.randn(*[*shape, max_num_chars, timeseries_util.gp_params_dim])

    log_prob = generative_model.latent_log_prob((raw_expression, eos, raw_gp_params))

    assert list(log_prob.shape) == shape


def test_generative_model_latent_sample_dims():
    sample_shape = [2, 3]
    max_num_chars, lstm_hidden_dim = 4, 5

    generative_model = GenerativeModel(max_num_chars, lstm_hidden_dim)
    raw_expression, eos, raw_gp_params = generative_model.latent_sample(sample_shape)

    assert list(raw_expression.shape) == sample_shape + [max_num_chars]
    assert list(eos.shape) == sample_shape + [max_num_chars]
    assert list(raw_gp_params.shape) == sample_shape + [
        max_num_chars,
        timeseries_util.gp_params_dim,
    ]
