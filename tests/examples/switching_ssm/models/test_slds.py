import torch
from cmws.examples.switching_ssm.models.slds import GenerativeModel, Guide


def test_generative_model_latent_log_prob_dims():
    num_states, continuous_dim, obs_dim, num_timesteps = 5, 2, 10, 100
    generative_model = GenerativeModel(num_states, continuous_dim, obs_dim, num_timesteps)

    for shape in [[], [2], [2, 3]]:
        discrete_states = torch.randint(num_states, size=shape + [num_timesteps])
        continuous_states = torch.randn(*[*shape, num_timesteps, continuous_dim])

        log_prob = generative_model.latent_log_prob((discrete_states, continuous_states))

        assert list(log_prob.shape) == shape
