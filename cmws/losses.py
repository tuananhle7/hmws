import math
import torch


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
