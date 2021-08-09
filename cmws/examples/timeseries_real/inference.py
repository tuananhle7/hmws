"""
SVI and IS inference for the timeseries model
"""
import copy
import itertools
import math
import torch
import cmws
import cmws.examples.timeseries_real.data as timeseries_data
import cmws.losses
import numpy as np


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


@torch.enable_grad()
def svi_single(
    num_particles, num_iterations, obs, discrete_latent, generative_model, guide, verbose=False
):
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
    # Copy guide
    guide_copy = copy.deepcopy(guide)

    # Initialize optimizer
    optimizer = torch.optim.Adam(
        itertools.chain(
            guide_copy.gp_params_lstm.parameters(),
            guide_copy.gp_params_extractor.parameters(),
            guide_copy.obs_embedder.parameters(),
            guide_copy.expression_embedder.parameters(),
        ),
        lr=0.05,
    )

    # Optimization loop
    l = []
    discrete_latent_repeat = (
        discrete_latent[0].repeat(num_particles, 1),
        discrete_latent[1].repeat(num_particles, 1),
    )
    obs_repeat = obs.repeat(num_particles, 1)

    iteration = 0
    while True:
        print(".", end="", flush=True)
        optimizer.zero_grad()
        # import pdb; pdb.set_trace()
        # loss = -get_elbo_single(discrete_latent, obs, generative_model, guide_copy)
        # loss = -get_elbo_single(discrete_latent_repeat, obs_repeat, generative_model, guide_copy).mean()

        continuous_latent = guide.rsample_continuous(obs_repeat, discrete_latent_repeat)
        latent = discrete_latent_repeat[0], discrete_latent_repeat[1], continuous_latent
        generative_model_log_prob = generative_model.log_prob(latent, obs_repeat)
        guide_log_prob = guide.log_prob(obs_repeat, latent)
        losses = generative_model_log_prob - guide_log_prob
        loss = losses.logsumexp(0)  # IAWE
        loss.backward()
        optimizer.step()
        l.append(loss.item())
        # if iteration % 10 == 0:
        # print("{:.0f}".format(np.mean(l[-10:])), end="")
        iteration += 1
        if num_iterations is None:
            if iteration > 100 and np.mean(l[-100:-50]) < np.mean(l[-50:]):
                break
            if iteration > 500:
                break
        else:
            if iteration > num_iterations:
                break

    # Sample
    continuous_latent = guide_copy.sample_continuous(obs_repeat, discrete_latent_repeat)
    # Log prob
    log_prob = guide_copy.log_prob_continuous(obs_repeat, discrete_latent_repeat, continuous_latent)

    return continuous_latent.detach(), log_prob.detach()


def svi_single_par(args):
    element_id = args[0]
    print(f"[{element_id}]", end="", flush=True)
    return svi_single(*args[1:])


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
    raw_expression_flattened = raw_expression.reshape(-1, generative_model.max_num_chars)
    eos_flattened = eos.reshape(-1, generative_model.max_num_chars)

    # Compute in a loop
    continuous_latent, log_prob = [], []
    # for element_id in range(num_elements):
    #     obs_e = obs_flattened[element_id]
    #     discrete_latent_e = (raw_expression_flattened[element_id], eos_flattened[element_id])
    #     continuous_latent_e, log_prob_e = svi_single(
    #         num_iterations, obs_e, discrete_latent_e, generative_model, guide
    #     )
    #     continuous_latent.append(continuous_latent_e)
    #     log_prob.append(log_prob_e)

    num_particles = 20
    svi_single_args = []
    for element_id in range(num_elements):
        obs_e = obs_flattened[element_id]
        discrete_latent_e = (raw_expression_flattened[element_id], eos_flattened[element_id])
        svi_single_args.append(
            (
                element_id,
                num_particles,
                num_iterations,
                obs_e,
                discrete_latent_e,
                generative_model,
                guide,
            )
        )

    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
    pool = mp.Pool(8)
    print(f"Running SVI individually on {num_elements} elements")
    svi_single_outputs = pool.map(svi_single_par, svi_single_args)
    print()
    print("Complete.")

    for element_id in range(num_elements):
        continuous_latent_e, log_prob_e = svi_single_outputs[element_id]
        continuous_latent.append(continuous_latent_e)
        log_prob.append(log_prob_e)

    return (
        torch.stack(continuous_latent).view(
            [*shape, num_particles, generative_model.max_num_chars, -1]
        ),
        torch.stack(log_prob).view([*shape, num_particles]),
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


def svi_memory(num_svi_iterations, obs, obs_id, generative_model, guide, memory):
    """Do stochastic VI on continuous latents and use the optimized distribution to do
    importance sampling.

    Args
        num_svi_iterations (int)
        obs [batch_size, num_timesteps]
        obs_id [batch_size]
        generative_model
        guide
        memory

    Returns
        latent
            raw_expression [memory_size, batch_size, max_num_chars]
            eos [memory_size, batch_size, max_num_chars]
            raw_gp_params [memory_size, batch_size, max_num_chars, gp_params_dim]
        log_weight [memory_size, batch_size]
    """
    # Extract
    batch_size = obs.shape[0]
    memory_size = memory.size

    # Sample discrete latent
    # [memory_size, batch_size, ...]
    discrete_latent = memory.select(obs_id)

    # Sample and score svi-optimized q(z_c | z_d, x)
    # -- Expand obs
    # [memory_size, batch_size, num_timesteps]
    obs_expanded = obs[None].expand([memory_size, batch_size, timeseries_data.num_timesteps])

    # -- Sample and score q(z_c | z_d, x)
    # [memory_size, batch_size, ...], [memory_size, batch_size]
    continuous_latent, continuous_latent_log_prob = svi(
        num_svi_iterations, obs_expanded, discrete_latent, generative_model, guide
    )

    # Combine latents
    # latent = discrete_latent[0], discrete_latent[1], continuous_latent
    latent_expand = (
        discrete_latent[0][:, :, None, :].repeat(1, 1, 20, 1),
        discrete_latent[1][:, :, None, :].repeat(1, 1, 20, 1),
        continuous_latent,
    )
    # Score p
    # [memory_size, batch_size]
    # generative_model_log_prob = generative_model.log_prob(latent, obs)
    generative_model_log_prob = generative_model.log_prob(
        latent_expand, obs[:, None].repeat(1, 20, 1)
    )

    # Score q
    # [memory_size, batch_size]
    # guide_log_prob = -math.log(memory_size) + continuous_latent_log_prob
    guide_log_prob = -math.log(memory_size) + continuous_latent_log_prob - math.log(20)

    # Compute weight
    log_weight = generative_model_log_prob - guide_log_prob

    continuous_latent_best = continuous_latent.gather(
        dim=2,
        index=log_weight.argmax(-1, keepdim=True)[..., None, None].repeat(
            1, 1, 1, *continuous_latent.shape[3:]
        ),
    ).squeeze(2)
    latent = discrete_latent[0], discrete_latent[1], continuous_latent_best

    return latent, log_weight.logsumexp(2)


def importance_sample_memory(num_particles, obs, obs_id, generative_model, guide, memory):
    """
    Args
        num_particles
        num_svi_iterations
        obs [batch_size, num_timesteps]
        obs_id [batch_size]
        generative_model
        guide
        memory

    Returns
        latent
            raw_expression [memory_size, batch_size, max_num_chars]
            eos [memory_size, batch_size, max_num_chars]
            raw_gp_params [memory_size, batch_size, max_num_chars, gp_params_dim]
        log_marginal_joint [memory_size, batch_size]
    """

    # Sample discrete latent
    # [memory_size, batch_size, ...]
    discrete_latent = memory.select(obs_id)

    # [num_particles, *discrete_shape, *shape, ...]
    continuous_latent = guide.sample_continuous(obs, discrete_latent, [num_particles])

    # log q(c | d)
    # [num_particles, *discrete_shape, *shape]
    log_q_continuous = guide.log_prob_continuous(obs, discrete_latent, continuous_latent)

    # log p(d, c, x)
    # [num_particles, *discrete_shape, *shape]
    log_p = generative_model.log_prob_discrete_continuous(discrete_latent, continuous_latent, obs)

    log_marginal_joint = torch.logsumexp(log_p - log_q_continuous, dim=0) - math.log(num_particles)

    # Sample from importance weighted posterior over continuous latents
    idx_resample = torch.distributions.Categorical(
        logits=(log_p - log_q_continuous).permute(1, 2, 0)
    ).sample()[None, ..., None, None]
    continuous_latent_resample = continuous_latent.gather(
        dim=0, index=idx_resample.expand(continuous_latent[:1].shape)
    )[0]

    # Combine latents
    latent = discrete_latent[0], discrete_latent[1], continuous_latent_resample
    return latent, log_marginal_joint


def importance_sample_memory_3(
    num_particles,
    num_svi_iterations,
    obs,
    obs_id,
    generative_model,
    guide,
    memory,
    num_continuous_optim_iterations,
    continuous_optim_lr,
):
    """
    Args
        num_particles
        num_svi_iterations
        obs [batch_size, num_timesteps]
        obs_id [batch_size]
        generative_model
        guide
        memory

    Returns
        latent
            raw_expression [memory_size, batch_size, max_num_chars]
            eos [memory_size, batch_size, max_num_chars]
            raw_gp_params [memory_size, batch_size, max_num_chars, gp_params_dim]
        log_marginal_joint [memory_size, batch_size]
    """
    # Sample discrete latent
    # [memory_size, batch_size, ...]
    discrete_latent = memory.select(obs_id)

    # COMPUTE SCORES s_i = log p(d_i, x) for i  {1, ..., (R + M)}
    # -- c ~ q(c | d, x)
    # [num_particles, memory_size, batch_size, ...]
    _continuous_latent = guide.sample_continuous(obs, discrete_latent, [num_particles])

    # OPTIMIZE CONTINUOUS LATENT
    _continuous_latent = cmws.losses.optimize_continuous(
        generative_model,
        obs,
        discrete_latent,
        _continuous_latent,
        num_iterations=num_continuous_optim_iterations,
        lr=continuous_optim_lr,
    )

    # -- log q(c | d)
    # [num_particles, memory_size, batch_size]
    _log_q_continuous = guide.log_prob_continuous(obs, discrete_latent, _continuous_latent,)

    # -- log p(d, c, x)
    # [num_particles, memory_size, batch_size]
    _log_p = generative_model.log_prob_discrete_continuous(discrete_latent, _continuous_latent, obs)

    # [memory_size, batch_size]
    log_marginal_joint = torch.logsumexp(_log_p - _log_q_continuous, dim=0) - math.log(
        num_particles
    )

    # Combine latents
    latent = discrete_latent[0], discrete_latent[1], _continuous_latent[0]

    return latent, log_marginal_joint


def importance_sample_memory_3(
    num_particles,
    num_svi_iterations,
    obs,
    obs_id,
    generative_model,
    guide,
    memory,
    num_continuous_optim_iterations,
    continuous_optim_lr,
):
    """
    Args
        num_particles
        num_svi_iterations
        obs [batch_size, num_timesteps]
        obs_id [batch_size]
        generative_model
        guide
        memory

    Returns
        latent
            raw_expression [memory_size, batch_size, max_num_chars]
            eos [memory_size, batch_size, max_num_chars]
            raw_gp_params [memory_size, batch_size, max_num_chars, gp_params_dim]
        log_marginal_joint [memory_size, batch_size]
    """
    # Sample discrete latent
    # [memory_size, batch_size, ...]
    discrete_latent = memory.select(obs_id)

    # COMPUTE SCORES s_i = log p(d_i, x) for i  {1, ..., (R + M)}
    # -- c ~ q(c | d, x)
    # [num_particles, memory_size, batch_size, ...]
    _continuous_latent = guide.sample_continuous(obs, discrete_latent, [num_particles])

    # OPTIMIZE CONTINUOUS LATENT
    _continuous_latent = cmws.losses.optimize_continuous(
        generative_model,
        obs,
        discrete_latent,
        _continuous_latent,
        num_iterations=num_continuous_optim_iterations,
        lr=continuous_optim_lr,
    )

    # -- log q(c | d)
    # [num_particles, memory_size, batch_size]
    _log_q_continuous = guide.log_prob_continuous(obs, discrete_latent, _continuous_latent,)

    # -- log p(d, c, x)
    # [num_particles, memory_size, batch_size]
    _log_p = generative_model.log_prob_discrete_continuous(discrete_latent, _continuous_latent, obs)

    # [memory_size, batch_size]
    log_marginal_joint = torch.logsumexp(_log_p - _log_q_continuous, dim=0) - math.log(
        num_particles
    )

    # Combine latents
    latent = discrete_latent[0], discrete_latent[1], _continuous_latent[0]

    return latent, log_marginal_joint
