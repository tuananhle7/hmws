import math

import torch

import cmws


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
            [*shape, *dims]

            OR

            [*shape, *dims1]
            ...
            [*shape, *dimsN]
        obs: tensor of shape [*shape, *obs_dims]
        num_particles

    Returns: [*shape]
    """
    # c ~ q(c | d, x)
    # [num_particles, *shape, ...]
    continuous_latent = guide.sample_continuous(obs, discrete_latent, [num_particles])

    # log q(c | d)
    # [num_particles, *shape]
    log_q_continuous = guide.log_prob_continuous(obs, discrete_latent, continuous_latent)

    # log p(d, c, x)
    # [num_particles, *shape]
    log_p = generative_model.log_prob_discrete_continuous(discrete_latent, continuous_latent, obs)

    return torch.logsumexp(log_p - log_q_continuous, dim=0) - math.log(num_particles)


def get_cmws_loss(generative_model, guide, memory, obs, obs_id, num_particles, num_proposals):
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
    Returns:
        loss: scalar that we call .backward() on and step the optimizer
    """
    # SAMPLE d'_{1:R} ~ q(d | x)
    # [num_proposals, batch_size, ...]
    proposed_discrete_latent = guide.sample_discrete(obs, (num_proposals,))

    # ASSIGN d_{1:(R + M)} = CONCAT(d'_{1:R}, d_{1:M})
    # [batch_size, memory_size + num_proposals, ...]
    discrete_latent_concat = cmws.memory.concat(proposed_discrete_latent, memory.select(obs_id))

    # COMPUTE SCORES s_i = log p(d_i, x) for i  {1, ..., (R + M)}
    # [batch_size, memory_size + num_proposals]
    log_marginal_joint = get_log_marginal_joint(
        generative_model, guide, discrete_latent_concat, obs, num_particles
    )

    # ASSIGN d_{1:M} = TOP_K_UNIQUE(d_{1:(R + M)}, s_{1:(R + M)})
    # [batch_size, memory_size, ...]
    discrete_latent_selected, _ = cmws.memory.get_unique_and_top_k(
        discrete_latent_concat, log_marginal_joint, memory.size
    )

    # UPDATE MEMORY with d_{1:M}
    memory.update(obs_id, discrete_latent_selected)

    # SAMPLE CONTINUOUS c_i ~ q(c | d_i, x) for i in {1, ..., M}
    # [memory_size, batch_size, ...]
    continuous_latent = guide.sample_continuous(obs, discrete_latent_selected.T)

    # COMPUTE WEIGHT
    # --COMPUTE log p(d_i, c_i, x) for i in {1, ..., M}
    # [memory_size, batch_size]
    log_p = generative_model.log_prob((discrete_latent_selected, continuous_latent), obs)

    # --COMPUTE log q(c_i | d_i, x) for i in {1, ..., M}
    log_q_continuous = guide.log_prob_continuous(obs, discrete_latent_selected, continuous_latent)

    # --COMPUTE log (1 / M)
    log_uniform = -math.log(memory.size)

    # --COMPUTE weight w_i and normalize
    normalized_weight = torch.softmax(
        log_p - (log_uniform + log_q_continuous), dim=0
    ).detach()  # [memory_size, batch_size]

    # COMPUTE log q(d_i, c_i | x) for i in {1, ..., M}
    log_q = guide.log_prob(obs, (discrete_latent_selected, continuous_latent))

    # COMPUTE losses
    generative_model_loss = -(log_p * normalized_weight).sum(dim=0)
    guide_loss = -(log_q * normalized_weight).sum(dim=0)

    return generative_model_loss + guide_loss

    ###########################
    ###########################
    ###########################

    # memory_size, num_arcs = memory.shape[1:3]
    # batch_size, num_rows, num_cols = obs.shape

    # # Select ids_and_on_offs from memory
    # memory_ids_and_on_offs = memory[obs_id]  # [batch_size, memory_size, num_arcs, 2]
    # memory_ids_and_on_offs_transposed = memory_ids_and_on_offs.transpose(
    #     0, 1
    # ).contiguous()  # [memory_size, batch_size, num_arcs, 2]

    # # PROPOSE DISCRETE LATENT
    # # [num_proposals, batch_size, num_arcs, 2]
    # proposed_ids_and_on_offs = guide.sample_ids_and_on_offs(obs, num_proposals)

    # # UPDATE MEMORY
    # # [memory_size + num_proposals, batch_size, num_arcs, 2]
    # ids_and_on_offs = torch.cat(
    #     [memory_ids_and_on_offs_transposed, proposed_ids_and_on_offs], dim=0
    # )
    # ids_and_on_offs_log_p = get_log_marginal_joint(
    #     generative_model,
    #     guide,
    #     ids_and_on_offs,
    #     obs[None].expand(memory_size + num_proposals, batch_size, num_rows, num_cols),
    #     num_particles=num_particles,
    # )  # [memory_size + num_proposals, batch_size]

    # # KEEP TOP `memory_size` ACCORDING TO log_p
    # # -- 1) Sort log_ps
    # # [memory_size + num_proposals, batch_size],
    # # [memory_size + num_proposals, batch_size]
    # sorted_without_inf, indices_without_inf = ids_and_on_offs_log_p.sort(dim=0)

    # # -- 2) Replace non-unique values with -inf
    # # [memory_size + num_proposals, batch_size, num_arcs, 2]
    # # sorted_m[i, j, k, l] = ids_and_on_offs[indices_without_inf[i, j], j, k, l]
    # sorted_ids_and_on_offs = ids_and_on_offs.gather(
    #     0,
    #     indices_without_inf[..., None, None].expand(
    #         memory_size + num_proposals, batch_size, num_arcs, 2
    #     ),
    # )
    # # [memory_size + num_proposals - 1, batch_size]
    # is_same = (
    #     (sorted_ids_and_on_offs[1:] == sorted_ids_and_on_offs[:-1])
    #     .view(memory_size + num_proposals - 1, batch_size, -1)
    #     .all(dim=-1)
    # )
    # sorted_without_inf[1:].masked_fill_(is_same, float("-inf"))

    # # -- 3) choose the top k remaining (valid as long as two distinct latents don't have the same
    # #       logp)
    # # [memory_size + num_proposals, batch_size],
    # # [memory_size + num_proposals, batch_size]
    # sorted_with_inf, indices_with_inf = sorted_without_inf.sort(dim=0)

    # # [memory_size, batch_size]
    # indices = indices_without_inf.gather(0, indices_with_inf)[-memory_size:]

    # # [memory_size, batch_size, num_arcs, 2]
    # memory_ids_and_on_offs = torch.gather(
    #     ids_and_on_offs,
    #     0,
    #     indices[:, :, None, None].expand(memory_size, batch_size, num_arcs, 2).contiguous(),
    # )

    # # memory: [len(dataset), memory_size, num_arcs, 2]
    # # [batch_size, memory_size, num_arcs, 2]
    # memory[obs_id] = memory_ids_and_on_offs.transpose(0, 1).contiguous()

    # # COMPUTE LOSSES
    # guide_motor_noise_dist = guide.get_motor_noise_dist(
    #     memory_ids_and_on_offs, obs[None].expand(memory_size, batch_size, num_rows, num_cols)
    # )  # batch_shape [memory_size, batch_size]
    # motor_noise = guide_motor_noise_dist.sample()  # [memory_size, batch_size, num_arcs, 3]
    # log_q_continuous = guide_motor_noise_dist.log_prob(motor_noise)  # [memory_size, batch_size]
    # log_uniform = -math.log(memory_size)

    # log_p = generative_model.get_log_prob(
    #     (memory_ids_and_on_offs, motor_noise), obs
    # )  # [memory_size, batch_size]
    # log_q = guide.get_log_prob(
    #     (memory_ids_and_on_offs, motor_noise), obs
    # )  # [memory_size, batch_size]

    # normalized_weight = torch.softmax(
    #     log_p - (log_uniform + log_q_continuous), dim=0
    # ).detach()  # [memory_size, batch_size]

    # generative_model_loss = -(log_p * normalized_weight).sum(dim=0).mean()
    # guide_loss = -(log_q * normalized_weight).sum(dim=0).mean()

    # return generative_model_loss + guide_loss, generative_model_loss.item(), guide_loss.item()
