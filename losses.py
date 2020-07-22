import numpy as np
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


def get_memory_elbo(generative_model, guide, memory, memory_log_weights, num_particles):
    """
    Args:
        generative_model: GenerativeModel object
        guide: Guide object
        memory: [memory_size]
        memory_log_weights: [memory_size]
        num_particles: int
    Returns:
        memory_elbo: scalar
    """
    # [memory_size]
    memory_log_weights_normalized = util.lognormexp(memory_log_weights)

    # batch_shape [memory_size]
    guide_continuous_dist = guide.get_continuous_dist(memory)

    # [num_particles, memory_size]
    continuous = guide_continuous_dist.sample((num_particles,))

    # [num_particles, memory_size]
    memory_expanded = memory[None].expand(num_particles, -1)

    # [num_particles, memory_size]
    log_p = generative_model.log_prob((memory_expanded, continuous))

    # [num_particles, memory_size]
    log_q_continuous = guide_continuous_dist.log_prob(continuous)

    return (
        memory_log_weights_normalized.exp()
        * (log_p - (memory_log_weights_normalized[None] + log_q_continuous)).mean(dim=0)
    ).sum()


def get_log_marginal_joint(generative_model, guide, memory, num_particles):
    """Estimates log p(discrete, obs), marginalizing out continuous.

    Args:
        generative_model: GenerativeModel object
        guide: Guide object
        memory: [memory_size]
        num_particles: int
    Returns:
        log_marginal_joint: [memory_size]
    """

    # batch_shape [memory_size]
    guide_continuous_dist = guide.get_continuous_dist(memory)

    # [num_particles, memory_size]
    continuous = guide_continuous_dist.sample((num_particles,))

    # [num_particles, memory_size]
    log_q_continuous = guide_continuous_dist.log_prob(continuous)

    # [num_particles, memory_size]
    memory_expanded = memory[None].expand(num_particles, -1)

    # [num_particles, memory_size]
    log_p = generative_model.log_prob((memory_expanded, continuous))

    return torch.logsumexp(log_p - log_q_continuous, dim=0) - math.log(num_particles)


def get_memory_log_weight(generative_model, guide, memory, num_particles):
    """Estimates normalized log weights associated with the approximation based on memory.

    sum_k weight_k memory_k ≅ p(discrete | obs)

    Args:
        generative_model: GenerativeModel object
        guide: Guide object
        memory: [memory_size]
        num_particles: int
    Returns:
        memory_log_weight: [memory_size]
    """
    return util.lognormexp(get_log_marginal_joint(generative_model, guide, memory, num_particles))


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


def get_cmws_loss(generative_model, guide, memory, num_particles):
    """MWS loss for hybrid discrete-continuous models as described in
    https://www.overleaf.com/project/5dfd4bbac2914e0001efb29b

    Args:
        generative_model: GenerativeModel object
        guide: Guide object
        memory: [memory_size]
        num_particles: int
    Returns:
        loss: scalar that we call .backward() on and step the optimizer
        memory
    """
    memory_size = memory.shape[0]

    # Propose discrete latent
    discrete = propose_discrete((), generative_model.support_size, generative_model.device)  # []
    while discrete in memory:
        discrete = propose_discrete(
            (), generative_model.support_size, generative_model.device
        )  # []

    # UPDATE MEMORY
    memory_log_weights = [get_memory_log_weight(generative_model, guide, memory, num_particles)]
    elbos = [
        get_memory_elbo(
            generative_model, guide, memory, memory_log_weights[-1], num_particles,
        ).item()
    ]
    for i in range(memory_size):
        replace_one_memory = get_replace_one_memory(memory, i, discrete)
        if len(torch.unique(replace_one_memory)) == memory_size:
            memory_log_weights.append(
                get_memory_log_weight(generative_model, guide, replace_one_memory, num_particles)
            )
            elbos.append(
                get_memory_elbo(
                    generative_model,
                    guide,
                    replace_one_memory,
                    memory_log_weights[-1],
                    num_particles,
                ).item()
            )
        else:
            # reject if memory elements are non-unique
            memory_log_weights.append(None)
            elbos.append(-math.inf)
    best_i = np.argmax(elbos)
    if best_i > 0:
        memory = get_replace_one_memory(memory, best_i - 1, discrete)
        print("---")
        print(f"discrete = {discrete}")
        print(f"elbos = {elbos}")
        print(f"best_i = {best_i}")
    print(f"memory = {memory}")

    # UPDATE MEMORY WEIGHTS
    memory_log_weight = memory_log_weights[best_i]

    # Compute losses
    guide_continuous_dist = guide.get_continuous_dist(memory)  # batch_shape [memory_size]
    continuous = guide_continuous_dist.sample()  # [memory_size]
    continuous_log_prob = guide_continuous_dist.log_prob(continuous)  # [memory_size]

    memory_log_q = memory_log_weight + continuous_log_prob  # [memory_size]
    log_p = generative_model.log_prob((memory, continuous))  # [memory_size]
    log_q = guide.log_prob((memory, continuous))  # [memory_size]

    normalized_weight = util.exponentiate_and_normalize(
        log_p - memory_log_q
    ).detach()  # [memory_size]

    generative_model_loss = -(log_p * normalized_weight).sum(dim=0)
    guide_loss = -(log_q * normalized_weight).sum(dim=0)

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
