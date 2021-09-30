import torch
from cmws.examples.switching_ssm.models.slds import GenerativeModel, Guide


def test_generative_model_latent_log_prob_dims():
    num_states, continuous_dim, obs_dim, num_timesteps = 5, 2, 10, 7
    generative_model = GenerativeModel(num_states, continuous_dim, obs_dim, num_timesteps)

    for shape in [[], [2], [2, 3]]:
        discrete_states = torch.randint(num_states, size=shape + [num_timesteps])
        continuous_states = torch.randn(*[*shape, num_timesteps, continuous_dim])

        log_prob = generative_model.latent_log_prob((discrete_states, continuous_states))

        assert list(log_prob.shape) == shape


def test_generative_model_latent_sample_dims():
    num_states, continuous_dim, obs_dim, num_timesteps = 5, 2, 10, 7
    generative_model = GenerativeModel(num_states, continuous_dim, obs_dim, num_timesteps)

    for sample_shape in [[], [2, 3]]:
        discrete_states, continuous_states = generative_model.latent_sample(sample_shape)

        assert list(discrete_states.shape) == sample_shape + [num_timesteps]
        assert list(continuous_states.shape) == sample_shape + [num_timesteps, continuous_dim]


def test_generative_model_discrete_latent_sample_dims():
    num_states, continuous_dim, obs_dim, num_timesteps = 5, 2, 10, 7
    generative_model = GenerativeModel(num_states, continuous_dim, obs_dim, num_timesteps)

    for sample_shape in [[], [2, 3]]:
        discrete_states = generative_model.discrete_latent_sample(sample_shape)

        assert list(discrete_states.shape) == sample_shape + [num_timesteps]


def test_generative_model_obs_log_prob_dims():
    num_states, continuous_dim, obs_dim, num_timesteps = 5, 2, 10, 7
    generative_model = GenerativeModel(num_states, continuous_dim, obs_dim, num_timesteps)

    for sample_shape in [[], [4], [4, 5]]:
        for shape in [[], [2], [2, 3]]:
            # Create latent
            discrete_states = torch.randint(num_states, size=sample_shape + shape + [num_timesteps])
            continuous_states = torch.randn(*[*sample_shape, *shape, num_timesteps, continuous_dim])
            latent = (discrete_states, continuous_states)

            # Create obs
            obs = torch.randn(*[*shape, num_timesteps, obs_dim])

            # Compute log prob
            log_prob = generative_model.obs_log_prob(latent, obs)

            assert list(log_prob.shape) == sample_shape + shape


def test_generative_model_log_prob_dims():
    num_states, continuous_dim, obs_dim, num_timesteps = 5, 2, 10, 7
    generative_model = GenerativeModel(num_states, continuous_dim, obs_dim, num_timesteps)

    for sample_shape in [[], [4], [4, 5]]:
        for shape in [[], [2], [2, 3]]:
            # Create latent
            discrete_states = torch.randint(num_states, size=sample_shape + shape + [num_timesteps])
            continuous_states = torch.randn(*[*sample_shape, *shape, num_timesteps, continuous_dim])
            latent = (discrete_states, continuous_states)

            # Create obs
            obs = torch.randn(*[*shape, num_timesteps, obs_dim])

            # Compute log prob
            log_prob = generative_model.log_prob(latent, obs)

            assert list(log_prob.shape) == sample_shape + shape


def test_generative_model_log_prob_discrete_continuous_dims():
    num_states, continuous_dim, obs_dim, num_timesteps = 5, 2, 10, 7
    generative_model = GenerativeModel(num_states, continuous_dim, obs_dim, num_timesteps)

    for shape in [[], [2, 3]]:
        for discrete_shape in [[], [4]]:
            for continuous_shape in [[], [5]]:
                # Create latent
                discrete_states = torch.randint(
                    num_states, size=discrete_shape + shape + [num_timesteps]
                )
                continuous_states = torch.randn(
                    *[*continuous_shape, *discrete_shape, *shape, num_timesteps, continuous_dim]
                )

                # Create obs
                obs = torch.randn(*[*shape, num_timesteps, obs_dim])

                # Compute log prob
                log_prob = generative_model.log_prob_discrete_continuous(
                    discrete_states, continuous_states, obs
                )

                assert list(log_prob.shape) == continuous_shape + discrete_shape + shape


def test_generative_model_sample_dims():
    num_states, continuous_dim, obs_dim, num_timesteps = 5, 2, 10, 7
    generative_model = GenerativeModel(num_states, continuous_dim, obs_dim, num_timesteps)

    for sample_shape in [[], [2, 3]]:
        latent, obs = generative_model.sample(sample_shape)
        discrete_states, continuous_states = latent

        assert list(discrete_states.shape) == sample_shape + [num_timesteps]
        assert list(continuous_states.shape) == sample_shape + [num_timesteps, continuous_dim]
        assert list(obs.shape) == sample_shape + [num_timesteps, obs_dim]


def test_generative_model_sample_obs_dims():
    num_states, continuous_dim, obs_dim, num_timesteps = 5, 2, 10, 7
    generative_model = GenerativeModel(num_states, continuous_dim, obs_dim, num_timesteps)

    for sample_shape in [[], [4], [4, 5]]:
        for shape in [[], [2], [2, 3]]:
            discrete_states = torch.randint(num_states, size=shape + [num_timesteps])
            continuous_states = torch.randn(*[*shape, num_timesteps, continuous_dim])
            obs = generative_model.sample_obs((discrete_states, continuous_states), sample_shape)

            assert list(obs.shape) == sample_shape + shape + [num_timesteps, obs_dim]


def test_guide_sample_discrete_dims():
    num_states, continuous_dim, obs_dim, num_timesteps = 5, 2, 10, 7
    guide = Guide(num_states, continuous_dim, obs_dim, num_timesteps)

    for shape in [[], [2, 3]]:
        for sample_shape in [[], [4, 5]]:
            # Create obs
            obs = torch.randn(*[*shape, num_timesteps, obs_dim])

            # Sample
            discrete_states = guide.sample_discrete(obs, sample_shape)

            assert list(discrete_states.shape) == sample_shape + shape + [num_timesteps]


def test_guide_sample_continuous_dims():
    num_states, continuous_dim, obs_dim, num_timesteps = 5, 2, 10, 7
    guide = Guide(num_states, continuous_dim, obs_dim, num_timesteps)
    for shape in [[], [2, 3]]:
        for discrete_shape in [[], [4]]:
            for sample_shape in [[], [5]]:
                # Create obs
                obs = torch.randn(*[*shape, num_timesteps, obs_dim])

                # Create latent
                discrete_latent = torch.randint(
                    num_states, size=discrete_shape + shape + [num_timesteps]
                )

                # Sample
                continuous_latent = guide.sample_continuous(obs, discrete_latent, sample_shape)

                assert list(continuous_latent.shape) == sample_shape + discrete_shape + shape + [
                    num_timesteps,
                    continuous_dim,
                ]
