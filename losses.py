import math
import util
import torch


def propose_discrete(sample_shape, support_size, device):
    """Returns: [*sample_shape]"""
    return torch.distributions.Categorical(logits=torch.zeros(support_size, device=device)).sample(
        sample_shape
    )


def get_mws_loss(generative_model, guide, memory, num_particles):
    """
    Args:
        generative_model: GenerativeModel object
        guide: Guide object
        memory:
            discrete: [memory_size]
            continuous: [memory_size]
        num_particles: int
    Returns:
        loss: scalar that we call .backward() on and step the optimizer
        memory
    """
    memory_size = memory[0].shape[0]

    # Propose latents
    discrete = propose_discrete(
        (num_particles,), generative_model.support_size, generative_model.device
    )  # [num_particles]
    continuous = guide.get_continuous_dist(discrete).sample()  # [num_particles]
    latent = (discrete, continuous)  # [num_particles], [num_particles]

    # Evaluate log p of proposed latents
    log_p = generative_model.log_prob(latent)  # [num_particles]

    # Evaluate log p of memory latents
    memory_log_p = generative_model.log_prob(memory)  # [memory_size]

    # Merge proposed latents and memory latents
    # [memory_size + num_particles], [memory_size + num_particles]
    memory_and_latent = [torch.cat([memory[i], latent[i]]) for i in range(2)]

    # Evaluate log p of merged latents
    # [memory_size + num_particles]
    memory_and_latent_log_p = torch.cat([memory_log_p, log_p])

    # Sort log_ps, replace non-unique values with -inf, choose the top k remaining
    # (valid as long as two distinct latents don't have the same logp)
    sorted1, indices1 = memory_and_latent_log_p.sort(dim=0)
    is_same = sorted1[1:] == sorted1[:-1]
    sorted1[1:].masked_fill_(is_same, float("-inf"))
    sorted2, indices2 = sorted1.sort(dim=0)
    memory_log_p = sorted2[-memory_size:]
    indices = indices1.gather(0, indices2)[-memory_size:]

    # [memory_size], [memory_size]
    memory_latent = [memory_and_latent[i][indices] for i in range(2)]

    # Update memory
    # [memory_size], [memory_size]
    for i in range(2):
        memory[i] = memory_latent[i]

    # Compute losses
    log_p = memory_log_p  # [memory_size]
    normalized_weight = util.exponentiate_and_normalize(log_p).detach()  # [memory_size]
    log_q = guide.log_prob(memory_latent)  # [memory_size]

    generative_model_loss = -(log_p * normalized_weight).sum(dim=0).mean()
    guide_loss = -(log_q * normalized_weight).sum(dim=0).mean()

    return generative_model_loss + guide_loss, memory


def get_log_marginal_joint_is(generative_model, guide, memory, num_particles):
    """Estimates log p(discrete, obs), marginalizing out continuous.

    Args:
        generative_model: GenerativeModel object
        guide: Guide object
        memory: [memory_size]
        num_particles: int
    Returns:
        log_marginal_joint: [memory_size]
    """

    # q(c | d)
    # batch_shape [memory_size]
    guide_continuous_dist = guide.get_continuous_dist(memory)

    # c ~ q(c | d)
    # [num_particles, memory_size]
    continuous = guide_continuous_dist.sample((num_particles,))

    # log q(c | d)
    # [num_particles, memory_size]
    log_q_continuous = guide_continuous_dist.log_prob(continuous)

    # [num_particles, memory_size]
    memory_expanded = memory[None].expand(num_particles, -1)

    # log p(d, c)
    # [num_particles, memory_size]
    log_p = generative_model.log_prob((memory_expanded, continuous))

    return torch.logsumexp(log_p - log_q_continuous, dim=0) - math.log(num_particles)


def get_log_marginal_joint_sgd(generative_model, guide, memory, num_iterations):
    """Estimates log p(discrete, obs) by evaluating max_continuous log p(discrete, continuous, obs)

    Args:
        generative_model: GenerativeModel object
        guide: Guide object
        memory: [memory_size]
        num_particles: int
    Returns:
        log_marginal_joint: [memory_size]
    """
    memory_size = len(memory)
    device = memory.device

    # c_init ~ q(c | d)
    continuous_init = guide.get_continuous_dist(memory).sample()  # [memory_size]
    continuous = torch.zeros((memory_size,), device=device)  # [memory_size]
    for i in range(memory_size):
        continuous_delta = torch.tensor(0.0, device=device, requires_grad=True)
        optimizer = torch.optim.SGD([continuous_delta], lr=0.1)

        for _ in range(num_iterations):
            loss = -generative_model.log_prob((memory[i], continuous_init[i] + continuous_delta))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        continuous[i] = continuous_init[i] + continuous_delta

    return generative_model.log_prob((memory, continuous))


def get_log_marginal_joint_exact(generative_model, memory):
    """Exactly evaluates log p(discrete, obs), marginalizing out continuous.

    Args:
        generative_model: GenerativeModel object
        memory: [memory_size]
    Returns:
        log_marginal_joint: [memory_size]
    """

    return generative_model.discrete_dist.log_prob(memory)


def get_log_marginal_joint(
    generative_model, guide, memory, num_particles=None, num_iterations=None
):
    """Evaluate / estimate log p(discrete, obs), marginalizing out continuous.

    Args:
        generative_model: GenerativeModel object
        guide: Guide object
        memory: [memory_size]
        num_particles (int): for estimating memory weights using importance sampling
        num_iterations (int): for estimating memory weights using SGD
    Returns:
        log_marginal_joint: [memory_size]
    """
    if num_particles is None and num_iterations is None:
        log_marginal_joint = get_log_marginal_joint_exact(generative_model, memory)
    elif num_particles is not None and num_iterations is not None:
        raise ValueError("both num_particles and num_iterations are not None")
    else:
        if num_particles is not None:
            log_marginal_joint = get_log_marginal_joint_is(
                generative_model, guide, memory, num_particles
            )
        elif num_iterations is not None:
            log_marginal_joint = get_log_marginal_joint_sgd(
                generative_model, guide, memory, num_iterations
            )
    return log_marginal_joint


def get_memory_log_weight(generative_model, guide, memory, num_particles=None, num_iterations=None):
    """Estimates normalized log weights associated with the approximation based on memory.

    sum_k weight_k memory_k ≅ p(discrete | obs)

    Args:
        generative_model: GenerativeModel object
        guide: Guide object
        memory: [memory_size]
        num_particles (int): for estimating memory weights using importance sampling
        num_iterations (int): for estimating memory weights using SGD
    Returns:
        memory_log_weight: [memory_size]
    """
    return util.lognormexp(
        get_log_marginal_joint(
            generative_model,
            guide,
            memory,
            num_particles=num_particles,
            num_iterations=num_iterations,
        )
    )


def get_replace_one_memory(memory, idx, replacement):
    """
    Args:
        memory: [memory_size]
        idx: int
        replacement: torch.long []

    Returns:
        replace_one_memory: [memory_size]
    """
    replace_one_memory = memory.clone().detach()
    replace_one_memory[idx] = replacement
    return replace_one_memory


def get_cmws_loss(
    generative_model, guide, memory, num_mc_samples, num_particles=None, num_iterations=None
):
    """MWS loss for hybrid discrete-continuous models as described in
    https://www.overleaf.com/project/5dfd4bbac2914e0001efb29b

    Args:
        generative_model: GenerativeModel object
        guide: Guide object
        memory: [memory_size]
        num_mc_samples (int): for estimating ELBOs for choosing memory elements
        num_particles (int): for estimating memory weights using importance sampling
        num_iterations (int): for estimating memory weights using SGD
    Returns:
        loss: scalar that we call .backward() on and step the optimizer
        memory
    """
    memory_size = memory.shape[0]

    # PROPOSE DISCRETE LATENT
    discrete = propose_discrete((), generative_model.support_size, generative_model.device)  # []
    while discrete in memory:
        discrete = propose_discrete(
            (), generative_model.support_size, generative_model.device
        )  # []

    # UPDATE MEMORY
    memory_and_latent = torch.cat([memory.clone().detach(), discrete[None]])  # [memory_size + 1]
    memory_and_latent_log_p = get_log_marginal_joint(
        generative_model,
        guide,
        memory_and_latent,
        num_particles=num_particles,
        num_iterations=num_iterations,
    )  # [memory_size + 1]

    # Keep top `memory_size` according to log_p
    worst_i = torch.argmin(memory_and_latent_log_p)
    j = 0
    for i in range(memory_size + 1):
        if i != worst_i:
            memory[j] = memory_and_latent[i].clone().detach()
            j += 1

    # UPDATE MEMORY WEIGHTS
    # [memory_size]
    memory_log_weight = get_memory_log_weight(
        generative_model, guide, memory, num_particles=num_particles, num_iterations=num_iterations
    )

    # COMPUTE LOSSES

    # ---- 1st version: reweigh using p(d, c) / q_M(d, c) ----
    # guide_continuous_dist = guide.get_continuous_dist(memory)  # batch_shape [memory_size]
    # continuous = guide_continuous_dist.sample()  # [memory_size]
    # continuous_log_prob = guide_continuous_dist.log_prob(continuous)  # [memory_size]

    # memory_log_q = memory_log_weight + continuous_log_prob  # [memory_size]
    # log_p = generative_model.log_prob((memory, continuous))  # [memory_size]
    # log_q = guide.log_prob((memory, continuous))  # [memory_size]

    # normalized_weight = util.exponentiate_and_normalize(
    #     log_p - memory_log_q
    # ).detach()  # [memory_size]

    # generative_model_loss = -(log_p * normalized_weight).sum(dim=0)
    # guide_loss = -(log_q * normalized_weight).sum(dim=0)

    # ---- 2nd version: sample a single sample from q_M(d, c) ----
    # memory_sample_id = torch.distributions.Categorical(logits=memory_log_weight).sample()  # []
    # memory_sample = memory[memory_sample_id]  # []
    # continuous_sample = guide.get_continuous_dist(memory_sample).sample()

    # generative_model_loss = -generative_model.log_prob((memory_sample, continuous_sample))  # []
    # guide_loss = -guide.log_prob((memory_sample, continuous_sample))  # []

    # ---- 3rd version: reweigh using q_M(d) p(c | d) / q_M(d, c) = p(c | d) / q(c | d) ----
    # guide_continuous_dist = guide.get_continuous_dist(memory)  # batch_shape [memory_size]
    # continuous = guide_continuous_dist.sample()  # [memory_size]
    # # log q(c | d)
    # log_q_continuous = guide_continuous_dist.log_prob(continuous)  # [memory_size]

    # # log p(c | d)
    # log_p_continuous = generative_model.get_continuous_dist(memory).log_prob(
    #     continuous
    # )  # [memory_size]

    # # log p(d, c)
    # log_p = generative_model.log_prob((memory, continuous))  # [memory_size]

    # # log q(d, c)
    # log_q = guide.log_prob((memory, continuous))  # [memory_size]

    # normalized_weight = util.exponentiate_and_normalize(
    #     log_p_continuous - log_q_continuous
    # ).detach()  # [memory_size]

    # generative_model_loss = -(log_p * normalized_weight).sum(dim=0)
    # guide_loss = -(log_q * normalized_weight).sum(dim=0)

    # ---- 4th version: reweigh using w = p(d, c) / Uniform(d) q(c | d) ----
    guide_continuous_dist = guide.get_continuous_dist(memory)  # batch_shape [memory_size]
    continuous = guide_continuous_dist.sample()  # [memory_size]
    log_q_continuous = guide_continuous_dist.log_prob(continuous)  # [memory_size]
    log_uniform = -torch.ones_like(memory) * math.log(memory_size)  # [memory_size]

    log_p = generative_model.log_prob((memory, continuous))  # [memory_size]
    log_q = guide.log_prob((memory, continuous))  # [memory_size]

    normalized_weight = util.exponentiate_and_normalize(
        log_p - (log_uniform + log_q_continuous)
    ).detach()  # [memory_size]

    generative_model_loss = -(log_p * normalized_weight).sum(dim=0)
    guide_loss = -(log_q * normalized_weight).sum(dim=0)

    # ---- 5th version: reweigh using w = q_M(d) p(c | d) / Uniform(d) q(c | d) ----
    # guide_continuous_dist = guide.get_continuous_dist(memory)  # batch_shape [memory_size]
    # continuous = guide_continuous_dist.sample()  # [memory_size]
    # log_q_continuous = guide_continuous_dist.log_prob(continuous)  # [memory_size]
    # log_uniform = -torch.ones_like(memory) * math.log(memory_size)  # [memory_size]
    # # [memory_size]
    # log_p_continuous = generative_model.get_continuous_dist(memory).log_prob(continuous)

    # log_p = generative_model.log_prob((memory, continuous))  # [memory_size]
    # log_q = guide.log_prob((memory, continuous))  # [memory_size]

    # normalized_weight = util.exponentiate_and_normalize(
    #     (memory_log_weight + log_p_continuous) - (log_uniform + log_q_continuous)
    # ).detach()  # [memory_size]

    # generative_model_loss = -(log_p * normalized_weight).sum(dim=0)
    # guide_loss = -(log_q * normalized_weight).sum(dim=0)

    return generative_model_loss + guide_loss, memory


def get_rws_loss(generative_model, guide, num_particles):
    latent = guide.sample([num_particles])
    log_p = generative_model.log_prob(latent)
    log_q = guide.log_prob(latent)

    log_weight = log_p - log_q.detach()

    normalized_weight_detached = util.exponentiate_and_normalize(log_weight).detach()

    guide_loss = -torch.sum(normalized_weight_detached * log_q)
    generative_model_loss = -torch.sum(normalized_weight_detached * log_p)

    return generative_model_loss + guide_loss


def get_elbo_loss(generative_model, guide, num_particles):
    latent = guide.sample([num_particles])
    log_p = generative_model.log_prob(latent)
    log_q = guide.log_prob(latent)

    log_weight = log_p - log_q
    return -(log_q * log_weight.detach() + log_weight).mean()
