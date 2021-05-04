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
