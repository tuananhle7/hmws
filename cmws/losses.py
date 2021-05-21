import matplotlib.pyplot as plt
import math

import torch

import cmws
from cmws import util


def get_sleep_loss(generative_model, guide, num_samples):
    """
    Args:
        generative_model
        guide
        num_samples (int)

    Returns: [num_samples]
    """
    latent, obs = generative_model.sample(sample_shape=(num_samples,))
    return -guide.log_prob(obs, latent)


def get_elbo_loss(generative_model, guide, obs):
    """
    Args:
        generative_model
        guide
        obs [batch_size, *obs_dims]

    Returns: [batch_size]
    """
    guide_dist = guide.get_dist(obs)
    latent = guide_dist.rsample()
    guide_log_prob = guide_dist.log_prob(latent)
    generative_model_log_prob = generative_model.log_prob(latent, obs)
    elbo = generative_model_log_prob - guide_log_prob

    if torch.isnan(elbo).any():
        raise RuntimeError("nan")
    return -elbo


def get_iwae_loss(generative_model, guide, obs, num_particles):
    """
    Args:
        generative_model
        guide
        obs [batch_size, *obs_dims]
        num_particles

    Returns: [batch_size]
    """
    # Sample from guide
    # [num_particles, batch_size, ...]
    latent = guide.rsample(obs, (num_particles,))

    # Expand obs to [num_particles, batch_size, *obs_dims]
    batch_size = obs.shape[0]
    obs_dims = obs.shape[1:]
    obs_expanded = obs[None].expand(*[num_particles, batch_size, *obs_dims])

    # Evaluate log probs
    # [num_particles, batch_size]
    guide_log_prob = guide.log_prob(obs_expanded, latent)
    # [num_particles, batch_size]
    generative_model_log_prob = generative_model.log_prob(latent, obs_expanded)

    # Compute log weight
    # [num_particles, batch_size]
    log_weight = generative_model_log_prob - guide_log_prob

    # Estimate log marginal likelihood
    # [batch_size]
    log_p = torch.logsumexp(log_weight, dim=0) - math.log(num_particles)

    return log_p


def get_rws_loss(generative_model, guide, obs, num_particles, insomnia=1.0):
    """
    Args:
        generative_model
        guide
        obs [batch_size, *obs_dims]
        num_particles
        insomnia (float) 1.0 means Wake-Wake, 0.0 means Wake-Sleep,
            otherwise it's inbetween

    Returns: [batch_size]
    """
    # Sample from guide
    # [num_particles, batch_size, ...]
    latent = guide.sample(obs, (num_particles,))

    # Expand obs to [num_particles, batch_size, *obs_dims]
    batch_size = obs.shape[0]
    obs_dims = obs.shape[1:]
    obs_expanded = obs[None].expand(*[num_particles, batch_size, *obs_dims])

    # Evaluate log probs
    # [num_particles, batch_size]
    guide_log_prob = guide.log_prob(obs_expanded, latent)
    # [num_particles, batch_size]
    generative_model_log_prob = generative_model.log_prob(latent, obs_expanded)

    # Compute log weight
    # [num_particles, batch_size]
    log_weight = generative_model_log_prob - guide_log_prob
    # [num_particles, batch_size]
    normalized_weight = torch.softmax(log_weight, dim=0).detach()

    # Compute loss
    # --Compute generative model loss
    # [batch_size]
    generative_model_loss = -torch.sum(normalized_weight * generative_model_log_prob, dim=0)

    # --Compute guide loss
    # ----Compute guide wake loss
    if insomnia < 1.0:
        # [batch_size]
        guide_loss_sleep = (
            get_sleep_loss(generative_model, guide, num_particles * batch_size)
            .view(batch_size, num_particles)
            .mean(-1)
        )
    # ----Compute guide wake loss
    if insomnia > 0.0:
        # [batch_size]
        guide_loss_wake = -torch.sum(normalized_weight * guide_log_prob, dim=0)

    # ----Combine guide sleep and wake losses
    if insomnia == 0.0:
        guide_loss = guide_loss_sleep
    elif insomnia == 1.0:
        guide_loss = guide_loss_wake
    else:
        guide_loss = insomnia * guide_loss_wake + (1 - insomnia) * guide_loss_sleep

    # --Compute loss
    loss = generative_model_loss + guide_loss

    # Check nan
    if torch.isnan(loss).any():
        raise RuntimeError("nan")

    return loss


def get_vimco_loss(generative_model, guide, obs, num_particles):
    """Almost twice faster version of VIMCO loss (measured for batch_size = 24,
        num_particles = 1000). Inspired by Adam Kosiorek's implementation.

    Args:
        generative_model
        guide
        obs [batch_size, *obs_dims]
        num_particles: int

    Returns: [batch_size]
    """
    # Sample from guide
    # [num_particles, batch_size, ...]
    latent = guide.sample(obs, (num_particles,))

    # Expand obs to [num_particles, batch_size, *obs_dims]
    batch_size = obs.shape[0]
    obs_dims = obs.shape[1:]
    obs_expanded = obs[None].expand(*[num_particles, batch_size, *obs_dims])

    # Evaluate log probs
    # [num_particles, batch_size]
    guide_log_prob = guide.log_prob(obs_expanded, latent)
    # [num_particles, batch_size]
    generative_model_log_prob = generative_model.log_prob(latent, obs_expanded)

    # Compute log weight
    # [batch_size, num_particles]
    log_weight = generative_model_log_prob - guide_log_prob

    # shape [num_particles, batch_size]
    # log_weight_[b, k] = 1 / (K - 1) \sum_{\ell \neq k} \log w_{b, \ell}
    log_weight_ = (torch.sum(log_weight, dim=0, keepdim=True) - log_weight) / (num_particles - 1)

    # shape [batch_size, num_particles, num_particles]
    # temp[b, k, k_] =
    #     log_weight_[b, k]     if k == k_
    #     log_weight[b, k]      otherwise
    temp = log_weight.unsqueeze(-1) + torch.diag_embed(log_weight_ - log_weight)

    # this is the \Upsilon_{-k} term below equation 3
    # shape [batch_size, num_particles]
    control_variate = torch.logsumexp(temp, dim=1) - math.log(num_particles)

    log_evidence = torch.logsumexp(log_weight, dim=1) - math.log(num_particles)
    loss = -log_evidence - torch.sum(
        (log_evidence.unsqueeze(-1) - control_variate).detach() * guide_log_prob, dim=1
    )

    return loss


@torch.no_grad()
def get_log_p_and_kl(generative_model, guide, obs, num_particles):
    """Estimates log marginal likelihood and KL using importance sampling

    Args:
        generative_model
        guide
        obs [batch_size, *obs_dims]
        num_particles

    Returns: [batch_size]
    """
    # Sample from guide
    # [num_particles, batch_size, ...]
    latent = guide.sample(obs, (num_particles,))

    # Expand obs to [num_particles, batch_size, *obs_dims]
    batch_size = obs.shape[0]
    obs_dims = obs.shape[1:]
    obs_expanded = obs[None].expand(*[num_particles, batch_size, *obs_dims])

    # Evaluate log probs
    # [num_particles, batch_size]
    guide_log_prob = guide.log_prob(obs_expanded, latent)
    # [num_particles, batch_size]
    generative_model_log_prob = generative_model.log_prob(latent, obs_expanded)

    # Compute log weight
    # [num_particles, batch_size]
    log_weight = generative_model_log_prob - guide_log_prob

    # Estimate log marginal likelihood
    # [batch_size]
    log_p = torch.logsumexp(log_weight, dim=0) - math.log(num_particles)

    # Estimate ELBO
    # [batch_size]
    elbo = torch.mean(log_weight, dim=0)

    # Estimate KL
    # [batch_size]
    kl = log_p - elbo

    return log_p, kl


def get_log_marginal_joint(generative_model, guide, discrete_latent, obs, num_particles):
    """Estimate log p(z_d, x) using importance sampling

    Args:
        generative_model
        guide
        discrete_latent
            [*discrete_shape, *shape, *dims]

            OR

            [*discrete_shape, *shape, *dims1]
            ...
            [*discrete_shape, *shape, *dimsN]
        obs: tensor of shape [*shape, *obs_dims]
        num_particles

    Returns: [*discrete_shape, *shape]
    """
    # c ~ q(c | d, x)
    # [num_particles, *discrete_shape, *shape, ...]
    continuous_latent = guide.sample_continuous(obs, discrete_latent, [num_particles])

    # log q(c | d)
    # [num_particles, *discrete_shape, *shape]
    log_q_continuous = guide.log_prob_continuous(obs, discrete_latent, continuous_latent)

    # log p(d, c, x)
    # [num_particles, *discrete_shape, *shape]
    log_p = generative_model.log_prob_discrete_continuous(discrete_latent, continuous_latent, obs)

    return torch.logsumexp(log_p - log_q_continuous, dim=0) - math.log(num_particles)


def get_cmws_loss(
    generative_model, guide, memory, obs, obs_id, num_particles, num_proposals, insomnia=1.0
):
    """MWS loss for hybrid discrete-continuous models as described in
    https://www.overleaf.com/project/5dfd4bbac2914e0001efb29b

    Args:
        generative_model
        guide
        memory
        obs: tensor of shape [batch_size, *obs_dims]
        obs_id: long tensor of shape [batch_size]
        num_particles (int): number of particles used to marginalize continuous latents
        num_proposals (int): number of proposed elements to be considered as new memory

    Returns: [batch_size]
    """
    # SAMPLE d'_{1:R} ~ q(d | x)
    # [num_proposals, batch_size, ...]
    proposed_discrete_latent = guide.sample_discrete(obs, (num_proposals,))

    # ASSIGN d_{1:(R + M)} = CONCAT(d'_{1:R}, d_{1:M})
    # [memory_size + num_proposals, batch_size, ...]
    discrete_latent_concat = cmws.memory.concat(memory.select(obs_id), proposed_discrete_latent)

    # COMPUTE SCORES s_i = log p(d_i, x) for i  {1, ..., (R + M)}
    # [memory_size + num_proposals, batch_size]
    log_marginal_joint = get_log_marginal_joint(
        generative_model, guide, discrete_latent_concat, obs, num_particles
    )

    # ASSIGN d_{1:M} = TOP_K_UNIQUE(d_{1:(R + M)}, s_{1:(R + M)})
    # [memory_size, batch_size, ...]
    discrete_latent_selected, _ = cmws.memory.get_unique_and_top_k(
        discrete_latent_concat, log_marginal_joint, memory.size
    )

    # UPDATE MEMORY with d_{1:M}
    memory.update(obs_id, discrete_latent_selected)

    # CHECK UNIQUE
    # if not memory.is_unique(obs_id).all():
    #     raise RuntimeError("memory not unique")

    # SAMPLE CONTINUOUS c_i ~ q(c | d_i, x) for i in {1, ..., M}
    # [memory_size, batch_size, ...]
    continuous_latent = guide.sample_continuous(obs, discrete_latent_selected)

    # COMPUTE WEIGHT
    # --COMPUTE log p(d_i, c_i, x) for i in {1, ..., M}
    # [memory_size, batch_size]
    log_p = generative_model.log_prob(
        util.tensors_to_list(discrete_latent_selected) + util.tensors_to_list(continuous_latent),
        obs,
    )

    # --COMPUTE log q(c_i | d_i, x) for i in {1, ..., M}
    # [memory_size, batch_size]
    log_q_continuous = guide.log_prob_continuous(obs, discrete_latent_selected, continuous_latent)

    # --COMPUTE log (1 / M)
    # scalar
    log_uniform = -math.log(memory.size)

    # --COMPUTE weight w_i and normalize
    # [memory_size, batch_size]
    normalized_weight = torch.softmax(log_p - (log_uniform + log_q_continuous), dim=0).detach()

    # COMPUTE log q(d_i, c_i | x) for i in {1, ..., M}
    # [memory_size, batch_size]
    log_q = guide.log_prob(
        obs,
        util.tensors_to_list(discrete_latent_selected) + util.tensors_to_list(continuous_latent),
    )

    # COMPUTE losses
    # --Compute generative_model loss
    generative_model_loss = -(log_p * normalized_weight).sum(dim=0)

    # --Compute guide loss
    # ----Compute guide wake loss
    batch_size = obs.shape[0]
    if insomnia < 1.0:
        # [batch_size]
        guide_loss_sleep = (
            get_sleep_loss(generative_model, guide, num_particles * batch_size)
            .view(batch_size, num_particles)
            .mean(-1)
        )
    # ----Compute guide CMWS loss
    if insomnia > 0.0:
        # [batch_size]
        guide_loss_cmws = -(log_q * normalized_weight).sum(dim=0)

    # ----Combine guide sleep and CMWS losses
    if insomnia == 0.0:
        guide_loss = guide_loss_sleep
    elif insomnia == 1.0:
        guide_loss = guide_loss_cmws
    else:
        guide_loss = insomnia * guide_loss_cmws + (1 - insomnia) * guide_loss_sleep

    return generative_model_loss + guide_loss


def get_cmws_2_loss(
    generative_model, guide, memory, obs, obs_id, num_particles, num_proposals, insomnia=1.0
):
    """Use reweighted training for q(z_c | z_d, x)

    Args:
        generative_model
        guide
        memory
        obs: tensor of shape [batch_size, *obs_dims]
        obs_id: long tensor of shape [batch_size]
        num_particles (int): number of particles used to marginalize continuous latents
        num_proposals (int): number of proposed elements to be considered as new memory

    Returns: [batch_size]
    """
    # Extract
    batch_size = obs.shape[0]

    # SAMPLE d'_{1:R} ~ q(d | x)
    # [num_proposals, batch_size, ...]
    proposed_discrete_latent = guide.sample_discrete(obs, (num_proposals,))

    # ASSIGN d_{1:(R + M)} = CONCAT(d'_{1:R}, d_{1:M})
    # [memory_size + num_proposals, batch_size, ...]
    discrete_latent_concat = cmws.memory.concat(memory.select(obs_id), proposed_discrete_latent)

    # COMPUTE SCORES s_i = log p(d_i, x) for i  {1, ..., (R + M)}
    # -- c ~ q(c | d, x)
    # [num_particles, memory_size + num_proposals, batch_size, ...]
    _continuous_latent = guide.sample_continuous(obs, discrete_latent_concat, [num_particles])

    # -- log q(c | d)
    # [num_particles, memory_size + num_proposals, batch_size]
    _log_q_continuous = guide.log_prob_continuous(obs, discrete_latent_concat, _continuous_latent)

    # -- log p(d, c, x)
    # [num_particles, memory_size + num_proposals, batch_size]
    _log_p = generative_model.log_prob_discrete_continuous(
        discrete_latent_concat, _continuous_latent, obs
    )

    # [memory_size + num_proposals, batch_size]
    log_marginal_joint = torch.logsumexp(_log_p - _log_q_continuous, dim=0) - math.log(
        num_particles
    )

    # ASSIGN d_{1:M} = TOP_K_UNIQUE(d_{1:(R + M)}, s_{1:(R + M)})
    # [memory_size, batch_size, ...], [memory_size, batch_size]
    discrete_latent_selected, _, indices = cmws.memory.get_unique_and_top_k(
        discrete_latent_concat, log_marginal_joint, memory.size, return_indices=True
    )

    # SELECT log q(c | d, x) and log p(d, c, x)
    # [num_particles, memory_size, batch_size]
    _log_q_continuous = torch.gather(
        _log_q_continuous, 1, indices[None].expand(num_particles, memory.size, batch_size)
    )
    # [num_particles, memory_size, batch_size]
    _log_p = torch.gather(_log_p, 1, indices[None].expand(num_particles, memory.size, batch_size))

    # COMPUTE WEIGHT
    # [num_particles, memory_size, batch_size]
    _log_weight = _log_p - _log_q_continuous

    # COMPUTE log q(d_i | x) for i in {1, ..., M}
    # [memory_size, batch_size]
    _log_q_discrete = guide.log_prob_discrete(obs, discrete_latent_selected,)

    # UPDATE MEMORY with d_{1:M}
    memory.update(obs_id, discrete_latent_selected)

    # CHECK UNIQUE
    # if not memory.is_unique(obs_id).all():
    #     raise RuntimeError("memory not unique")

    # COMPUTE losses
    # --Compute generative model loss
    # [batch_size]
    generative_model_loss = -(
        (torch.softmax(_log_weight, dim=0).detach() * _log_p).sum(dim=0).mean(dim=0)
    )

    # --Compute guide loss
    # ----Compute guide wake loss
    batch_size = obs.shape[0]
    if insomnia < 1.0:
        # [batch_size]
        guide_loss_sleep = (
            get_sleep_loss(generative_model, guide, num_particles * batch_size)
            .view(batch_size, num_particles)
            .mean(-1)
        )
    # ----Compute guide CMWS loss
    if insomnia > 0.0:
        _log_weight_normalized = torch.softmax(_log_weight.view(-1, batch_size), dim=0).view(
            num_particles, memory.size, batch_size
        )
        # [batch_size]
        guide_loss_cmws = -(
            _log_weight_normalized.detach() * (_log_q_discrete[None] + _log_q_continuous)
        ).sum(dim=[0, 1])

    # ----Combine guide sleep and CMWS losses
    if insomnia == 0.0:
        guide_loss = guide_loss_sleep
    elif insomnia == 1.0:
        guide_loss = guide_loss_cmws
    else:
        guide_loss = insomnia * guide_loss_cmws + (1 - insomnia) * guide_loss_sleep

    return generative_model_loss + guide_loss


def optimize_continuous(
    generative_model, obs, discrete_latent, continuous_latent, num_iterations=10, lr=1e-3
):
    """
    Args:
        generative_model
        obs: tensor of shape [batch_size, *obs_dims]
        discrete_latent [*discrete_shape, batch_size, ...]
        continuous_latent [*continuous_shape, *discrete_shape, batch_size, ...]

    Returns:
        continuous_latent_new [*continuous_shape, *discrete_shape, batch_size, ...]
    """
    util.logging.info("Optimizing continuous latents")
    continuous_latent_delta = torch.zeros_like(continuous_latent, requires_grad=True)
    optimizer = torch.optim.Adam([continuous_latent_delta], lr=lr)
    losses = []
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        loss = -generative_model.log_prob_discrete_continuous(
            discrete_latent, continuous_latent + continuous_latent_delta, obs
        ).sum()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    plt.plot(losses, color="black")
    plt.show()
    return continuous_latent + continuous_latent_delta.detach()


def get_cmws_3_loss(
    generative_model,
    guide,
    memory,
    obs,
    obs_id,
    num_particles,
    num_proposals,
    insomnia=1.0,
    num_continuous_optim_iterations=10,
    continuous_optim_lr=1e-3,
):
    """Use reweighted training for q(z_c | z_d, x) + gradient descent on continuous latents

    Args:
        generative_model
        guide
        memory
        obs: tensor of shape [batch_size, *obs_dims]
        obs_id: long tensor of shape [batch_size]
        num_particles (int): number of particles used to marginalize continuous latents
        num_proposals (int): number of proposed elements to be considered as new memory
    Returns: [batch_size]
    """
    # Extract
    batch_size = obs.shape[0]

    # SAMPLE d'_{1:R} ~ q(d | x)
    # [num_proposals, batch_size, ...]
    proposed_discrete_latent = guide.sample_discrete(obs, (num_proposals,))

    # ASSIGN d_{1:(R + M)} = CONCAT(d'_{1:R}, d_{1:M})
    # [memory_size + num_proposals, batch_size, ...]
    discrete_latent_concat = cmws.memory.concat(memory.select(obs_id), proposed_discrete_latent)

    # COMPUTE SCORES s_i = log p(d_i, x) for i  {1, ..., (R + M)}
    # -- c ~ q(c | d, x)
    # [num_particles, memory_size + num_proposals, batch_size, ...]
    _continuous_latent = guide.sample_continuous(obs, discrete_latent_concat, [num_particles])

    # OPTIMIZE CONTINUOUS LATENT
    _continuous_latent = optimize_continuous(
        generative_model,
        obs,
        discrete_latent_concat,
        _continuous_latent,
        num_iterations=num_continuous_optim_iterations,
        lr=continuous_optim_lr,
    )

    # -- log q(c | d)
    # [num_particles, memory_size + num_proposals, batch_size]
    _log_q_continuous = guide.log_prob_continuous(obs, discrete_latent_concat, _continuous_latent,)

    # -- log p(d, c, x)
    # [num_particles, memory_size + num_proposals, batch_size]
    _log_p = generative_model.log_prob_discrete_continuous(
        discrete_latent_concat, _continuous_latent, obs
    )

    # [memory_size + num_proposals, batch_size]
    log_marginal_joint = torch.logsumexp(_log_p - _log_q_continuous, dim=0) - math.log(
        num_particles
    )

    # ASSIGN d_{1:M} = TOP_K_UNIQUE(d_{1:(R + M)}, s_{1:(R + M)})
    # [memory_size, batch_size, ...], [memory_size, batch_size]
    discrete_latent_selected, _, indices = cmws.memory.get_unique_and_top_k(
        discrete_latent_concat, log_marginal_joint, memory.size, return_indices=True
    )

    # SELECT log q(c | d, x) and log p(d, c, x)
    # [num_particles, memory_size, batch_size]
    _log_q_continuous = torch.gather(
        _log_q_continuous, 1, indices[None].expand(num_particles, memory.size, batch_size)
    )
    # [num_particles, memory_size, batch_size]
    _log_p = torch.gather(_log_p, 1, indices[None].expand(num_particles, memory.size, batch_size))

    # COMPUTE WEIGHT
    # [num_particles, memory_size, batch_size]
    _log_weight = _log_p - _log_q_continuous

    # COMPUTE log q(d_i | x) for i in {1, ..., M}
    # [memory_size, batch_size]
    _log_q_discrete = guide.log_prob_discrete(obs, discrete_latent_selected,)

    # UPDATE MEMORY with d_{1:M}
    memory.update(obs_id, discrete_latent_selected)

    # CHECK UNIQUE
    # if not memory.is_unique(obs_id).all():
    #     raise RuntimeError("memory not unique")

    # COMPUTE losses
    # --Compute generative model loss
    # [batch_size]
    generative_model_loss = -(
        (torch.softmax(_log_weight, dim=0).detach() * _log_p).sum(dim=0).mean(dim=0)
    )

    # --Compute guide loss
    # ----Compute guide wake loss
    batch_size = obs.shape[0]
    if insomnia < 1.0:
        # [batch_size]
        guide_loss_sleep = (
            get_sleep_loss(generative_model, guide, num_particles * batch_size)
            .view(batch_size, num_particles)
            .mean(-1)
        )
    # ----Compute guide CMWS loss
    if insomnia > 0.0:
        _log_weight_normalized = torch.softmax(_log_weight.view(-1, batch_size), dim=0).view(
            num_particles, memory.size, batch_size
        )
        # [batch_size]
        guide_loss_cmws = -(
            _log_weight_normalized.detach() * (_log_q_discrete[None] + _log_q_continuous)
        ).sum(dim=[0, 1])

    # ----Combine guide sleep and CMWS losses
    if insomnia == 0.0:
        guide_loss = guide_loss_sleep
    elif insomnia == 1.0:
        guide_loss = guide_loss_cmws
    else:
        guide_loss = insomnia * guide_loss_cmws + (1 - insomnia) * guide_loss_sleep

    return generative_model_loss + guide_loss


def get_cmws_4_loss(
    generative_model, guide, memory, obs, obs_id, num_particles, num_proposals, insomnia=1.0
):
    """Use reweighted training for q(z_c | z_d, x) but reweigh it individually for each memory element

    Args:
        generative_model
        guide
        memory
        obs: tensor of shape [batch_size, *obs_dims]
        obs_id: long tensor of shape [batch_size]
        num_particles (int): number of particles used to marginalize continuous latents
        num_proposals (int): number of proposed elements to be considered as new memory

    Returns: [batch_size]
    """
    # Extract
    batch_size = obs.shape[0]

    # SAMPLE d'_{1:R} ~ q(d | x)
    # [num_proposals, batch_size, ...]
    proposed_discrete_latent = guide.sample_discrete(obs, (num_proposals,))

    # ASSIGN d_{1:(R + M)} = CONCAT(d'_{1:R}, d_{1:M})
    # [memory_size + num_proposals, batch_size, ...]
    discrete_latent_concat = cmws.memory.concat(memory.select(obs_id), proposed_discrete_latent)

    # COMPUTE SCORES s_i = log p(d_i, x) for i  {1, ..., (R + M)}
    # -- c ~ q(c | d, x)
    # [num_particles, memory_size + num_proposals, batch_size, ...]
    _continuous_latent = guide.sample_continuous(obs, discrete_latent_concat, [num_particles])

    # -- log q(c | d)
    # [num_particles, memory_size + num_proposals, batch_size]
    _log_q_continuous = guide.log_prob_continuous(obs, discrete_latent_concat, _continuous_latent)

    # -- log p(d, c, x)
    # [num_particles, memory_size + num_proposals, batch_size]
    _log_p = generative_model.log_prob_discrete_continuous(
        discrete_latent_concat, _continuous_latent, obs
    )

    # [memory_size + num_proposals, batch_size]
    log_marginal_joint = torch.logsumexp(_log_p - _log_q_continuous, dim=0) - math.log(
        num_particles
    )

    # ASSIGN d_{1:M} = TOP_K_UNIQUE(d_{1:(R + M)}, s_{1:(R + M)})
    # [memory_size, batch_size, ...], [memory_size, batch_size]
    discrete_latent_selected, _, indices = cmws.memory.get_unique_and_top_k(
        discrete_latent_concat, log_marginal_joint, memory.size, return_indices=True
    )

    # SELECT log q(c | d, x) and log p(d, c, x)
    # [num_particles, memory_size, batch_size]
    _log_q_continuous = torch.gather(
        _log_q_continuous, 1, indices[None].expand(num_particles, memory.size, batch_size)
    )
    # [num_particles, memory_size, batch_size]
    _log_p = torch.gather(_log_p, 1, indices[None].expand(num_particles, memory.size, batch_size))

    # COMPUTE WEIGHT
    # [num_particles, memory_size, batch_size]
    _log_weight = _log_p - _log_q_continuous

    # COMPUTE log q(d_i | x) for i in {1, ..., M}
    # [memory_size, batch_size]
    _log_q_discrete = guide.log_prob_discrete(obs, discrete_latent_selected,)

    # UPDATE MEMORY with d_{1:M}
    memory.update(obs_id, discrete_latent_selected)

    # CHECK UNIQUE
    # if not memory.is_unique(obs_id).all():
    #     raise RuntimeError("memory not unique")

    # COMPUTE losses
    # --Compute generative model loss
    # [batch_size]
    generative_model_loss = -(
        (torch.softmax(_log_weight, dim=0).detach() * _log_p).sum(dim=0).mean(dim=0)
    )

    # --Compute guide loss
    # ----Compute guide wake loss
    batch_size = obs.shape[0]
    if insomnia < 1.0:
        # [batch_size]
        guide_loss_sleep = (
            get_sleep_loss(generative_model, guide, num_particles * batch_size)
            .view(batch_size, num_particles)
            .mean(-1)
        )
    # ----Compute guide CMWS loss
    if insomnia > 0.0:
        _log_weight_normalized = torch.softmax(_log_weight.view(-1, batch_size), dim=0).view(
            num_particles, memory.size, batch_size
        )
        # [batch_size]
        guide_loss_cmws = -(
            (_log_weight_normalized.detach() * _log_q_discrete[None]).sum(dim=[0, 1])
            + (torch.softmax(_log_weight, dim=0).detach() * _log_q_continuous)
            .sum(dim=0)
            .mean(dim=0)
        )

    # ----Combine guide sleep and CMWS losses
    if insomnia == 0.0:
        guide_loss = guide_loss_sleep
    elif insomnia == 1.0:
        guide_loss = guide_loss_cmws
    else:
        guide_loss = insomnia * guide_loss_cmws + (1 - insomnia) * guide_loss_sleep

    return generative_model_loss + guide_loss
