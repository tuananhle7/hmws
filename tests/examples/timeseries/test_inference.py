import torch
import cmws.examples.timeseries.inference
import cmws.examples.timeseries.util as timeseries_util
import cmws.examples.timeseries.data as timeseries_data
from cmws.examples.timeseries.models.timeseries import GenerativeModel, Guide


def test_get_elbo_single_dim():
    max_num_chars, lstm_hidden_dim = 4, 5

    generative_model = GenerativeModel(max_num_chars, lstm_hidden_dim)
    guide = Guide(max_num_chars, lstm_hidden_dim)

    # Create obs
    obs = torch.randn(timeseries_data.num_timesteps)

    # Create latent
    raw_expression = torch.randint(timeseries_util.vocabulary_size, size=[max_num_chars])
    eos = torch.randint(2, size=[max_num_chars])
    discrete_latent = (raw_expression, eos)

    elbo_single = cmws.examples.timeseries.inference.get_elbo_single(
        discrete_latent, obs, generative_model, guide
    )

    assert list(elbo_single.shape) == []


def test_svi_single_dim():
    max_num_chars, lstm_hidden_dim, num_iterations = 4, 5, 6

    generative_model = GenerativeModel(max_num_chars, lstm_hidden_dim)
    guide = Guide(max_num_chars, lstm_hidden_dim)

    # Create obs
    obs = torch.randn(timeseries_data.num_timesteps)

    # Create latent
    raw_expression = torch.randint(timeseries_util.vocabulary_size, size=[max_num_chars])
    eos = torch.randint(2, size=[max_num_chars])
    discrete_latent = (raw_expression, eos)

    continuous_latent, log_prob = cmws.examples.timeseries.inference.svi_single(
        num_iterations, obs, discrete_latent, generative_model, guide
    )

    assert list(continuous_latent.shape) == [max_num_chars, timeseries_util.gp_params_dim]
    assert list(log_prob.shape) == []


def test_svi_dim():
    max_num_chars, lstm_hidden_dim, num_iterations = 4, 5, 6

    generative_model = GenerativeModel(max_num_chars, lstm_hidden_dim)
    guide = Guide(max_num_chars, lstm_hidden_dim)

    for shape in [[], [2], [2, 3]]:
        # Create obs
        obs = torch.randn(*[*shape, timeseries_data.num_timesteps])

        # Create latent
        raw_expression = torch.randint(
            timeseries_util.vocabulary_size, size=shape + [max_num_chars]
        )
        eos = torch.randint(2, size=shape + [max_num_chars])
        discrete_latent = (raw_expression, eos)

        continuous_latent, log_prob = cmws.examples.timeseries.inference.svi(
            num_iterations, obs, discrete_latent, generative_model, guide
        )

        assert list(continuous_latent.shape) == shape + [
            max_num_chars,
            timeseries_util.gp_params_dim,
        ]
        assert list(log_prob.shape) == shape


def test_svi_importance_sampling_dim():
    max_num_chars, lstm_hidden_dim, num_particles, num_svi_iterations = 4, 5, 6, 7

    generative_model = GenerativeModel(max_num_chars, lstm_hidden_dim)
    guide = Guide(max_num_chars, lstm_hidden_dim)

    for shape in [[], [2], [2, 3]]:
        # Create obs
        obs = torch.randn(*[*shape, timeseries_data.num_timesteps])

        (
            (raw_expression, eos, raw_gp_params),
            log_weight,
        ) = cmws.examples.timeseries.inference.svi_importance_sampling(
            num_particles, num_svi_iterations, obs, generative_model, guide
        )

        assert list(raw_expression.shape) == [num_particles] + shape + [max_num_chars]
        assert list(eos.shape) == [num_particles] + shape + [max_num_chars]
        assert list(raw_gp_params.shape) == [num_particles] + shape + [
            max_num_chars,
            timeseries_util.gp_params_dim,
        ]
        assert list(log_weight.shape) == [num_particles] + shape
