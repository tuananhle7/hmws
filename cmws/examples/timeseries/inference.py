"""
SVI and IS inference for the timeseries model
"""
import copy
import itertools
from tqdm import tqdm
import torch
import cmws
import cmws.examples.timeseries.data as timeseries_data


def get_elbo_single(discrete_latent, obs, generative_model, guide):
    """E_q(z_c | z_d, x)[log p(z_d, z_c, x) - log q(z_c | z_d, x)]

    Args
        obs [num_timesteps]
        discrete_latent
            raw_expression [max_num_chars]
            eos [max_num_chars]
        generative_model
        guide

    Returns []
    """
    # Reparameterized sampling of z_c ~ q(z_c | z_d, x)
    # [max_num_chars, gp_params_dim]
    continuous_latent = guide.rsample_continuous(obs, discrete_latent)

    # Assemble latent
    latent = discrete_latent[0], discrete_latent[1], continuous_latent

    # Compute log p
    generative_model_log_prob = generative_model.log_prob(latent, obs)

    # Compute log q
    guide_log_prob = guide.log_prob(obs, latent)

    return generative_model_log_prob - guide_log_prob


def svi_single(num_iterations, obs, discrete_latent, generative_model, guide):
    """
    Args
        num_iterations (int)
        obs [num_timesteps]
        discrete_latent
            raw_expression [max_num_chars]
            eos [max_num_chars]
        generative_model
        guide

    Returns
        continuous_latent
            raw_gp_params [max_num_chars, gp_params_dim]
        log_prob [] the value of the svi-optimized log q(z_c | z_d, x)
    """
    cmws.util.logging.info("SVI optimization")

    # Copy guide
    guide_copy = copy.deepcopy(guide)

    # Initialize optimizer
    optimizer = torch.optim.Adam(
        itertools.chain(
            guide_copy.gp_params_lstm.parameters(),
            guide_copy.gp_params_extractor.parameters(),
            guide_copy.obs_embedder.parameters(),
            guide_copy.expression_embedder.parameters(),
        )
    )

    # Optimization loop
    for iteration in tqdm(range(num_iterations)):
        optimizer.zero_grad()
        loss = get_elbo_single(discrete_latent, obs, generative_model, guide_copy)
        loss.backward()
        optimizer.step()

    # Sample
    continuous_latent = guide_copy.sample_continuous(obs, discrete_latent)

    # Log prob
    log_prob = guide_copy.log_prob_continuous(obs, discrete_latent, continuous_latent)

    return continuous_latent, log_prob


def svi(num_iterations, obs, discrete_latent, generative_model, guide):
    """
    Args
        num_iterations (int)
        obs [*shape, num_timesteps]
        discrete_latent
            raw_expression [*shape, max_num_chars]
            eos [*shape, max_num_chars]
        generative_model
        guide

    Returns
        continuous_latent
            raw_gp_params [*shape, max_num_chars, gp_params_dim]
        log_prob [*shape] the value of the svi-optimized log q(z_c | z_d, x)
    """
    # Extract
    shape = obs.shape[:-1]
    num_elements = cmws.util.get_num_elements(shape)
    raw_expression, eos = discrete_latent

    # Flatten
    obs_flattened = obs.reshape(-1, timeseries_data.num_timesteps)
    raw_expression_flattened = raw_expression.view(-1, generative_model.max_num_chars)
    eos_flattened = eos.view(-1, generative_model.max_num_chars)

    # Compute in a loop
    continuous_latent, log_prob = [], []
    for element_id in range(num_elements):
        obs_e = obs_flattened[element_id]
        discrete_latent_e = (raw_expression_flattened[element_id], eos_flattened[element_id])
        continuous_latent_e, log_prob_e = svi_single(
            num_iterations, obs_e, discrete_latent_e, generative_model, guide
        )
        continuous_latent.append(continuous_latent_e)
        log_prob.append(log_prob_e)

    return (
        torch.stack(continuous_latent).view([*shape, generative_model.max_num_chars, -1]),
        torch.stack(log_prob).view(shape),
    )


def svi_importance_sampling(num_particles, num_svi_iterations, obs, generative_model, guide):
    """Do stochastic VI on continuous latents and use the optimized distribution to do
    importance sampling.

    Args
        num_particles (int)
        num_svi_iterations (int)
        obs [*shape, num_timesteps]
        generative_model
        guide

    Returns
        latent
            raw_expression [num_particles, *shape, max_num_chars]
            eos [num_particles, *shape, max_num_chars]
            raw_gp_params [num_particles, *shape, max_num_chars, gp_params_dim]
        log_weight [num_particles, *shape]
    """
    # Extract
    shape = obs.shape[:-1]

    # Sample discrete latent
    # [num_particles, *shape, ...]
    discrete_latent = guide.sample_discrete(obs, [num_particles])

    # Sample and score svi-optimized q(z_c | z_d, x)
    # -- Expand obs
    # [num_particles, *shape, num_timesteps]
    obs_expanded = obs[None].expand([num_particles, *shape, timeseries_data.num_timesteps])

    # -- Sample and score q(z_c | z_d, x)
    # [num_particles, *shape, ...], [num_particles, *shape]
    continuous_latent, continuous_latent_log_prob = svi(
        num_svi_iterations, obs_expanded, discrete_latent, generative_model, guide
    )

    # Combine latents
    latent = discrete_latent[0], discrete_latent[1], continuous_latent

    # Score p
    # [num_particles, *shape]
    generative_model_log_prob = generative_model.log_prob(latent, obs)

    # Score q
    # [num_particles, *shape]
    guide_log_prob = guide.log_prob_discrete(obs, discrete_latent) + continuous_latent_log_prob

    # Compute weight
    log_weight = generative_model_log_prob - guide_log_prob

    return latent, log_weight
