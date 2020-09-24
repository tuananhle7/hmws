import torch
import math
import util


def get_latent_and_log_weight_and_log_q(generative_model, guide, obs, obs_id=None, num_particles=1):
    """Samples latent and computes log weight and log prob of inference network.

    Args:
        generative_model: models.GenerativeModel object
        guide: models.Guide object
        obs: tensor of shape [batch_size, num_rows, num_cols]
        obs_id: long tensor of shape [batch_size]
        num_particles: int

    Returns:
        latent: tensor of shape [num_particles, batch_size, num_arcs, 2]
        log_weight: tensor of shape [batch_size, num_particles]
        log_q: tensor of shape [batch_size, num_particles]
    """

    latent_dist = guide.get_latent_dist(obs)
    latent = guide.sample_from_latent_dist(latent_dist, num_particles)
    log_p = generative_model.get_log_prob(latent, obs).transpose(0, 1)
    log_q = guide.get_log_prob_from_latent_dist(latent_dist, latent).transpose(0, 1)
    log_weight = log_p - log_q
    return latent, log_weight, log_q


def get_log_weight_and_log_q(generative_model, guide, obs, obs_id=None, num_particles=1):
    """Compute log weight and log prob of inference network.

    Args:
        generative_model: models.GenerativeModel object
        guide: models.Guide object
        obs: tensor of shape [batch_size, num_rows, num_cols]
        obs_id: long tensor of shape [batch_size]
        num_particles: int

    Returns:
        log_weight: tensor of shape [batch_size, num_particles]
        log_q: tensor of shape [batch_size, num_particles]
    """

    latent, log_weight, log_q = get_latent_and_log_weight_and_log_q(
        generative_model, guide, obs, obs_id, num_particles
    )
    return log_weight, log_q


def get_wake_theta_loss_from_log_weight(log_weight):
    """Args:
        log_weight: tensor of shape [batch_size, num_particles]

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """

    _, num_particles = log_weight.shape
    elbo = torch.mean(torch.logsumexp(log_weight, dim=1) - math.log(num_particles))
    return -elbo, elbo


def get_wake_phi_loss_from_log_weight_and_log_q(log_weight, log_q):
    """Returns:
        loss: scalar that we call .backward() on and step the optimizer.
    """
    normalized_weight = util.exponentiate_and_normalize(log_weight, dim=1)
    return torch.mean(-torch.sum(normalized_weight.detach() * log_q, dim=1))


def get_rws_loss(generative_model, guide, obs, obs_id, num_particles):
    latent_dist = guide.get_latent_dist(obs)
    latent = guide.sample_from_latent_dist(latent_dist, num_particles)
    log_p = generative_model.get_log_prob(latent, obs).transpose(0, 1)
    log_q = guide.get_log_prob_from_latent_dist(latent_dist, latent).transpose(0, 1)
    log_weight = log_p - log_q.detach()

    # wake theta
    wake_theta_loss, elbo = get_wake_theta_loss_from_log_weight(log_weight)

    # wake phi
    wake_phi_loss = get_wake_phi_loss_from_log_weight_and_log_q(log_weight, log_q)

    return wake_theta_loss + wake_phi_loss, wake_theta_loss.item(), wake_phi_loss.item()


def get_vimco_loss(generative_model, guide, obs, obs_id, num_particles):
    """Almost twice faster version of VIMCO loss (measured for batch_size = 24,
        num_particles = 1000). Inspired by Adam Kosiorek's implementation.

    Args:
        generative_model: models.GenerativeModel object
        guide: models.Guide object
        obs: tensor of shape [batch_size, num_rows, num_cols]
        obs_id: long tensor of shape [batch_size]
        num_particles: int

    Returns:

        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """
    log_weight, log_q = get_log_weight_and_log_q(
        generative_model, guide, obs, obs_id, num_particles
    )

    # shape [batch_size, num_particles]
    # log_weight_[b, k] = 1 / (K - 1) \sum_{\ell \neq k} \log w_{b, \ell}
    log_weight_ = (torch.sum(log_weight, dim=1, keepdim=True) - log_weight) / (num_particles - 1)

    # shape [batch_size, num_particles, num_particles]
    # temp[b, k, k_] =
    #     log_weight_[b, k]     if k == k_
    #     log_weight[b, k]      otherwise
    temp = log_weight.unsqueeze(-1) + torch.diag_embed(log_weight_ - log_weight)

    # this is the \Upsilon_{-k} term below equation 3
    # shape [batch_size, num_particles]
    control_variate = torch.logsumexp(temp, dim=1) - math.log(num_particles)

    log_evidence = torch.logsumexp(log_weight, dim=1) - math.log(num_particles)
    elbo = torch.mean(log_evidence)
    loss = -elbo - torch.mean(
        torch.sum((log_evidence.unsqueeze(-1) - control_variate).detach() * log_q, dim=1)
    )
    theta_loss = -elbo.item()
    phi_loss = theta_loss

    return loss, theta_loss, phi_loss


def get_mws_loss(generative_model, guide, memory, obs, obs_id, num_particles):
    """
    Args:
        generative_model: models.GenerativeModel object
        guide: models.Guide object
        memory: tensor of shape [num_data, memory_size, num_arcs, 2]
        obs: tensor of shape [batch_size, num_rows, num_cols]
        obs_id: long tensor of shape [batch_size]
        num_particles: int

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
        theta_loss: python float
        phi_loss: python float
        prior_loss, accuracy, novel_proportion, new_map: python float
    """
    memory_size = memory.shape[1]

    # Propose latents from inference network
    latent_dist = guide.get_latent_dist(obs)
    # [num_particles, batch_size, num_arcs, 2]
    latent = guide.sample_from_latent_dist(latent_dist, num_particles).detach()
    batch_size = latent.shape[1]

    # Evaluate log p of proposed latents
    # [num_particles, batch_size]
    log_prior, log_likelihood = generative_model.get_log_probss(latent, obs, obs_id)
    log_p = log_prior + log_likelihood

    # Select latents from memory
    memory_latent = memory[obs_id]  # [batch_size, memory_size, num_arcs, 2]
    memory_latent_transposed = memory_latent.transpose(
        0, 1
    ).contiguous()  # [memory_size, batch_size, num_arcs, 2]

    # Evaluate log p of memory latents
    memory_log_prior, memory_log_likelihood = generative_model.get_log_probss(
        memory_latent_transposed, obs, obs_id
    )  # [memory_size, batch_size]
    memory_log_p = memory_log_prior + memory_log_likelihood
    memory_log_likelihood = None  # don't need this anymore

    # Merge proposed latents and memory latents
    # [memory_size + num_particles, batch_size, num_arcs, 2]
    memory_and_latent = torch.cat([memory_latent_transposed, latent])

    # Evaluate log p of merged latents
    # [memory_size + num_particles, batch_size]
    memory_and_latent_log_p = torch.cat([memory_log_p, log_p])
    memory_and_latent_log_prior = torch.cat([memory_log_prior, log_prior])

    # Compute new map
    new_map = (log_p.max(dim=0).values > memory_log_p.max(dim=0).values).float().mean()

    # Sort log_ps, replace non-unique values with -inf, choose the top k remaining
    # (valid as long as two distinct latents don't have the same logp)
    sorted1, indices1 = memory_and_latent_log_p.sort(dim=0)
    is_same = sorted1[1:] == sorted1[:-1]
    novel_proportion = 1 - (is_same.float().sum() / num_particles / batch_size)
    sorted1[1:].masked_fill_(is_same, float("-inf"))
    sorted2, indices2 = sorted1.sort(dim=0)
    memory_log_p = sorted2[-memory_size:]
    memory_latent
    indices = indices1.gather(0, indices2)[-memory_size:]

    memory_latent = torch.gather(
        memory_and_latent,
        0,
        indices[:, :, None, None]
        .expand(memory_size, batch_size, generative_model.num_arcs, 2)
        .contiguous(),
    )
    memory_log_prior = torch.gather(memory_and_latent_log_prior, 0, indices)

    # Update memory
    # [batch_size, memory_size, num_arcs, 2]
    memory[obs_id] = memory_latent.transpose(0, 1).contiguous()

    # Compute losses
    dist = torch.distributions.Categorical(logits=memory_log_p.t())
    sampled_memory_id = dist.sample()  # [batch_size]
    sampled_memory_id_log_prob = dist.log_prob(sampled_memory_id)  # [batch_size]
    sampled_memory_latent = torch.gather(
        memory_latent,
        0,
        sampled_memory_id[None, :, None, None]
        .expand(1, batch_size, generative_model.num_arcs, 2)
        .contiguous(),
    )  # [1, batch_size, num_arcs, 2]

    log_p = torch.gather(memory_log_p, 0, sampled_memory_id[None])[0]  # [batch_size]
    log_q = guide.get_log_prob_from_latent_dist(latent_dist, sampled_memory_latent)[
        0
    ]  # [batch_size]

    prior_loss = -torch.gather(memory_log_prior, 0, sampled_memory_id[None, :]).mean().detach()
    theta_loss = -torch.mean(log_p - sampled_memory_id_log_prob.detach())  # []
    phi_loss = -torch.mean(log_q)  # []
    loss = theta_loss + phi_loss

    accuracy = None
    return (
        loss,
        theta_loss.item(),
        phi_loss.item(),
        prior_loss.item(),
        accuracy,
        novel_proportion.item(),
        new_map.item(),
    )


def get_log_marginal_joint(generative_model, guide, ids_and_on_offs, obs, num_particles):
    """
    Args:
        generative_model
        guide
        ids_and_on_offs [*shape, num_arcs, 2]
        obs: tensor of shape [*shape, num_rows, num_cols]
        num_particles

    Returns: [*shape]
    """
    shape = ids_and_on_offs.shape[:-2]
    num_samples = int(torch.tensor(shape).prod().long().item())
    num_arcs = ids_and_on_offs.shape[-2]
    num_rows, num_cols = obs.shape[-2:]

    # q(c | d, x)
    # batch_shape [*shape]
    guide_motor_noise_dist = guide.get_motor_noise_dist(ids_and_on_offs, obs)

    # c ~ q(c | d, x)
    # [num_particles, *shape, num_arcs, 3]
    motor_noise = guide_motor_noise_dist.sample((num_particles,))

    # log q(c | d)
    # [num_particles, *shape]
    log_q_continuous = guide_motor_noise_dist.log_prob(motor_noise)

    # log p(d, c)
    # [num_particles, *shape]
    log_p = generative_model.get_log_prob(
        (
            # [num_particles, num_samples, num_arcs, 2]
            ids_and_on_offs[None]
            .expand(*[num_particles, *shape, num_arcs, 2])
            .view(num_particles, num_samples, num_arcs, 2),
            # [num_particles, num_samples, num_arcs, 3]
            motor_noise.view(num_particles, num_samples, num_arcs, 3),
        ),
        # [num_samples, num_rows, num_cols]
        obs.reshape(num_samples, num_rows, num_cols),
    ).view(*[num_particles, *shape])

    return torch.logsumexp(log_p - log_q_continuous, dim=0) - math.log(num_particles)


def get_cmws_loss(generative_model, guide, memory, obs, obs_id, num_particles, num_proposals):
    """MWS loss for hybrid discrete-continuous models as described in
    https://www.overleaf.com/project/5dfd4bbac2914e0001efb29b

    Args:
        generative_model
        guide
        memory: tensor of shape [num_data, memory_size, num_arcs, 2]
        obs: tensor of shape [batch_size, num_rows, num_cols]
        obs_id: long tensor of shape [batch_size]
        num_particles (int): number of particles used to marginalize motor noie
        num_proposals (int): number of proposed elements to be considered as new memory

    Returns:
        loss: scalar that we call .backward() on and step the optimizer
    """

    memory_size, num_arcs = memory.shape[1:3]
    batch_size, num_rows, num_cols = obs.shape

    # Select ids_and_on_offs from memory
    memory_ids_and_on_offs = memory[obs_id]  # [batch_size, memory_size, num_arcs, 2]
    memory_ids_and_on_offs_transposed = memory_ids_and_on_offs.transpose(
        0, 1
    ).contiguous()  # [memory_size, batch_size, num_arcs, 2]

    # PROPOSE DISCRETE LATENT
    # [num_proposals, batch_size, num_arcs, 2]
    proposed_ids_and_on_offs = guide.sample_ids_and_on_offs(obs, num_proposals)

    # UPDATE MEMORY
    # [memory_size + num_proposals, batch_size, num_arcs, 2]
    ids_and_on_offs = torch.cat(
        [memory_ids_and_on_offs_transposed, proposed_ids_and_on_offs], dim=0
    )
    ids_and_on_offs_log_p = get_log_marginal_joint(
        generative_model,
        guide,
        ids_and_on_offs,
        obs[None].expand(memory_size + num_proposals, batch_size, num_rows, num_cols),
        num_particles=num_particles,
    )  # [memory_size + num_proposals, batch_size]

    # KEEP TOP `memory_size` ACCORDING TO log_p
    # -- 1) Sort log_ps
    # [memory_size + num_proposals, batch_size],
    # [memory_size + num_proposals, batch_size]
    sorted_without_inf, indices_without_inf = ids_and_on_offs_log_p.sort(dim=0)

    # -- 2) Replace non-unique values with -inf
    # [memory_size + num_proposals, batch_size, num_arcs, 2]
    # sorted_m[i, j, k, l] = ids_and_on_offs[indices_without_inf[i, j], j, k, l]
    sorted_ids_and_on_offs = ids_and_on_offs.gather(
        0,
        indices_without_inf[..., None, None].expand(
            memory_size + num_proposals, batch_size, num_arcs, 2
        ),
    )
    # [memory_size + num_proposals - 1, batch_size]
    is_same = (
        (sorted_ids_and_on_offs[1:] == sorted_ids_and_on_offs[:-1])
        .view(memory_size + num_proposals - 1, batch_size, -1)
        .all(dim=-1)
    )
    sorted_without_inf[1:].masked_fill_(is_same, float("-inf"))

    # -- 3) choose the top k remaining (valid as long as two distinct latents don't have the same
    #       logp)
    # [memory_size + num_proposals, batch_size],
    # [memory_size + num_proposals, batch_size]
    sorted_with_inf, indices_with_inf = sorted_without_inf.sort(dim=0)

    # [memory_size, batch_size]
    indices = indices_without_inf.gather(0, indices_with_inf)[-memory_size:]

    # [memory_size, batch_size, num_arcs, 2]
    memory_ids_and_on_offs = torch.gather(
        ids_and_on_offs,
        0,
        indices[:, :, None, None].expand(memory_size, batch_size, num_arcs, 2).contiguous(),
    )

    # memory: [len(dataset), memory_size, num_arcs, 2]
    # [batch_size, memory_size, num_arcs, 2]
    memory[obs_id] = memory_ids_and_on_offs.transpose(0, 1).contiguous()

    # COMPUTE LOSSES
    guide_motor_noise_dist = guide.get_motor_noise_dist(
        memory_ids_and_on_offs, obs[None].expand(memory_size, batch_size, num_rows, num_cols)
    )  # batch_shape [memory_size, batch_size]
    motor_noise = guide_motor_noise_dist.sample()  # [memory_size, batch_size, num_arcs, 3]
    log_q_continuous = guide_motor_noise_dist.log_prob(motor_noise)  # [memory_size, batch_size]
    log_uniform = -math.log(memory_size)

    log_p = generative_model.get_log_prob(
        (memory_ids_and_on_offs, motor_noise), obs
    )  # [memory_size, batch_size]
    log_q = guide.get_log_prob(
        (memory_ids_and_on_offs, motor_noise), obs
    )  # [memory_size, batch_size]

    normalized_weight = util.exponentiate_and_normalize(
        log_p - (log_uniform + log_q_continuous)
    ).detach()  # [memory_size, batch_size]

    generative_model_loss = -(log_p * normalized_weight).sum(dim=0).mean()
    guide_loss = -(log_q * normalized_weight).sum(dim=0).mean()

    return generative_model_loss + guide_loss, generative_model_loss.item(), guide_loss.item()


def get_sleep_loss(generative_model, guide, num_samples=1):
    """Returns:
        loss: scalar that we call .backward() on and step the optimizer.
    """

    latent, obs = generative_model.sample_latent_and_obs(num_samples=num_samples)
    return -torch.mean(guide.get_log_prob(latent, obs))
