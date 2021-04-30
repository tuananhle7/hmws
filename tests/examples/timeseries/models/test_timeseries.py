import torch
from cmws.examples.timeseries.models.timeseries import GenerativeModel
import cmws.examples.timeseries.util as timeseries_util
import cmws.examples.timeseries.data as timeseries_data


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


def test_generative_model_discrete_latent_sample_dims():
    sample_shape = [2, 3]
    max_num_chars, lstm_hidden_dim = 4, 5

    generative_model = GenerativeModel(max_num_chars, lstm_hidden_dim)
    raw_expression, eos = generative_model.discrete_latent_sample(sample_shape)

    assert list(raw_expression.shape) == sample_shape + [max_num_chars]
    assert list(eos.shape) == sample_shape + [max_num_chars]


def test_generative_model_log_prob_dims():
    shape = [2, 3]
    sample_shape = [4, 5]
    max_num_chars, lstm_hidden_dim = 6, 7

    generative_model = GenerativeModel(max_num_chars, lstm_hidden_dim)

    # Create latent
    raw_expression = torch.randint(
        timeseries_util.vocabulary_size, size=sample_shape + shape + [max_num_chars]
    )
    eos = torch.randint(2, size=sample_shape + shape + [max_num_chars])
    raw_gp_params = torch.randn(
        *[*sample_shape, *shape, max_num_chars, timeseries_util.gp_params_dim]
    )
    latent = (raw_expression, eos, raw_gp_params)

    # Create obs
    obs = torch.randn(*[*shape, timeseries_data.num_timesteps])

    # Compute log prob
    log_prob = generative_model.log_prob(latent, obs)

    assert list(log_prob.shape) == sample_shape + shape


def test_generative_model_log_prob_discrete_continuous_dims():
    shape = [2, 3]
    discrete_shape = [4]
    continuous_shape = [5]
    max_num_chars, lstm_hidden_dim = 6, 7

    generative_model = GenerativeModel(max_num_chars, lstm_hidden_dim)

    # Create latent
    raw_expression = torch.randint(
        timeseries_util.vocabulary_size, size=discrete_shape + shape + [max_num_chars]
    )
    eos = torch.randint(2, size=discrete_shape + shape + [max_num_chars])
    raw_gp_params = torch.randn(
        *[*continuous_shape, *discrete_shape, *shape, max_num_chars, timeseries_util.gp_params_dim]
    )
    discrete_latent = (raw_expression, eos)
    continuous_latent = raw_gp_params

    # Create obs
    obs = torch.randn(*[*shape, timeseries_data.num_timesteps])

    # Compute log prob
    log_prob = generative_model.log_prob_discrete_continuous(
        discrete_latent, continuous_latent, obs
    )

    assert list(log_prob.shape) == continuous_shape + discrete_shape + shape


def test_generative_model_sample_dims():
    sample_shape = [2, 3]
    max_num_chars, lstm_hidden_dim = 4, 5

    generative_model = GenerativeModel(max_num_chars, lstm_hidden_dim)
    latent, obs = generative_model.sample(sample_shape)
    raw_expression, eos, raw_gp_params = latent

    assert list(raw_expression.shape) == sample_shape + [max_num_chars]
    assert list(eos.shape) == sample_shape + [max_num_chars]
    assert list(raw_gp_params.shape) == sample_shape + [
        max_num_chars,
        timeseries_util.gp_params_dim,
    ]
    assert list(obs.shape) == sample_shape + [timeseries_data.num_timesteps]
