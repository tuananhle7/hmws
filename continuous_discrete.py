import seaborn as sns
import argparse
import numpy as np
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import util


class GenerativeModel:
    def __init__(self, support_size, device):
        self.support_size = support_size
        self.device = device
        # self.logits = torch.rand(support_size, device=device) - 0.5
        self.logits = torch.zeros(support_size, device=device)
        self.logits[-1] = 1
        self.discrete_dist = torch.distributions.Categorical(logits=self.logits)

        self.locs = torch.linspace(-2, 2, self.support_size, device=device)
        # self.locs = torch.zeros(support_size, device=device)
        self.scales = torch.ones(support_size, device=device)

    def get_continuous_dist(self, discrete):
        """
        Args:
            discrete: [*batch_shape]

        Returns: distribution of batch_shape [*batch_shape] and event_shape []
        """
        return torch.distributions.Normal(self.locs[discrete], self.scales[discrete])

    def log_prob(self, latent):
        """
        Args:
            latent:
                discrete: [*batch_shape]
                continuous: [*batch_shape]

        Return: [*batch_shape]
        """
        discrete, continuous = latent
        return self.discrete_dist.log_prob(discrete) + self.get_continuous_dist(discrete).log_prob(
            continuous
        )

    def plot(self, path):
        discrete = torch.arange(self.support_size, device=self.device)

        xs = torch.linspace(-self.support_size, self.support_size, steps=1000)

        continuous_dist = self.get_continuous_dist(discrete)
        # [support_size, len(xs)]
        probss = continuous_dist.log_prob(xs[:, None].expand(-1, self.support_size)).T.exp()

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        ax = axs[0]
        ax.bar(torch.arange(self.support_size), self.discrete_dist.probs)
        ax.set_ylabel("$p(z_d)$")

        ax = axs[1]
        for i, probs in enumerate(probss):
            ax.plot(xs, probs, label=f"$z_d = {i}$")
        ax.set_xlim(-self.support_size, self.support_size)

        ax.set_ylabel("$p(z_c | z_d)$")
        ax.legend()
        util.save_fig(fig, path)


class Guide(nn.Module):
    def __init__(self, support_size):
        super().__init__()
        self.support_size = support_size
        self.logits = nn.Parameter(torch.rand(support_size))

        # self.locs = nn.Parameter(torch.rand(support_size))
        self.locs = nn.Parameter(torch.rand(support_size) - 0.5)
        # self.locs = nn.Parameter(torch.zeros(support_size))

        self.register_buffer("scales", torch.ones(support_size))

    @property
    def device(self):
        return self.logits.device

    @property
    def discrete_dist(self):
        return torch.distributions.Categorical(logits=self.logits)

    def get_continuous_dist(self, discrete):
        """
        Args:
            discrete: [*batch_shape]

        Returns: distribution of batch_shape [*batch_shape] and event_shape []
        """
        return torch.distributions.Normal(self.locs[discrete], self.scales[discrete])

    def log_prob(self, latent):
        """
        Args:
            latent:
                discrete: [*batch_shape]
                continuous: [*batch_shape]

        Return: [*batch_shape]
        """
        discrete, continuous = latent
        return self.discrete_dist.log_prob(discrete) + self.get_continuous_dist(discrete).log_prob(
            continuous
        )

    def sample(self, sample_shape=torch.Size([])):
        """
        Args:
            sample_shape: list / tuple / torch.Size

        Returns:
            latent:
                discrete: [*sample_shape]
                continuous: [*sample_shape]
        """
        discrete = self.discrete_dist.sample(sample_shape).detach()
        continuous = self.get_continuous_dist(discrete).sample().detach()
        return discrete, continuous

    def plot(self, path):
        discrete = torch.arange(self.support_size, device=self.device)

        xs = torch.linspace(-self.support_size, self.support_size, steps=1000)

        continuous_dist = self.get_continuous_dist(discrete)
        # [support_size, len(xs)]
        probss = (
            continuous_dist.log_prob(xs[:, None].expand(-1, self.support_size)).T.exp().detach()
        )

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        ax = axs[0]
        ax.bar(torch.arange(self.support_size), self.discrete_dist.probs.detach())
        ax.set_ylabel("$q(z_d)$")

        ax = axs[1]
        for i, probs in enumerate(probss):
            ax.plot(xs, probs, label=f"$z_d = {i}$")
        ax.set_xlim(-self.support_size, self.support_size)

        ax.set_ylabel("$q(z_c | z_d)$")
        ax.legend()
        util.save_fig(fig, path)


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


def init_memory(memory_size, support_size, device):
    discrete = torch.distributions.Categorical(
        logits=torch.zeros((support_size,), device=device)
    ).sample((memory_size,))
    continuous = torch.randn((memory_size,), device=device)
    return [discrete, continuous]


def empirical_discrete_probs(data, support_size):
    discrete_probs = torch.zeros(support_size, device=data.device)
    for i in range(support_size):
        discrete_probs[i] = (data == i).sum()
    return discrete_probs / len(data)


def plot_memory(path, memory, support_size):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    ax = axs[0]
    # ax.hist(
    #     memory[0], bins=torch.linspace(-0.5, support_size + 0.5, support_size + 1), density=True
    # )
    ax.bar(torch.arange(support_size), empirical_discrete_probs(memory[0], support_size))
    ax.set_ylabel("$q_M(z_d)$")

    ax = axs[1]
    for i in range(support_size):
        if sum(memory[0] == i) > 0:
            sns.kdeplot(
                memory[1][memory[0] == i].detach().numpy(),
                ax=ax,
                label=f"$z_d = {i}$",
                color=f"C{i}",
            )
        # ax.hist(memory[1][memory[0] == i], density=True, label=f"$z_d = {i}$")
    ax.set_xlim(-support_size, support_size)
    ax.set_ylabel("$q_M(z_c | z_d)$")
    ax.legend()
    util.save_fig(fig, path)


def plot_continuous_memory(path, generative_model, guide, memory, num_particles):
    if len(torch.unique(memory)) != len(memory):
        raise RuntimeError("memory elements not unique")
    support_size = generative_model.support_size
    memory = torch.sort(memory)[0]

    # [memory_size]
    memory_log_weight = get_memory_log_weight(generative_model, guide, memory, num_particles)

    xs = torch.linspace(-support_size, support_size, steps=1000)
    discrete = memory.clone().detach()  # torch.arange(support_size, device=guide.device)
    continuous_dist = guide.get_continuous_dist(discrete)
    # [memory_size, len(xs)]
    probss = continuous_dist.log_prob(xs[:, None].expand(-1, len(memory))).T.exp().detach()

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    ax = axs[0]
    support = torch.arange(support_size)
    memory_prob = torch.zeros(support_size)
    memory_prob[memory] = util.exponentiate_and_normalize(memory_log_weight).detach()
    ax.bar(support, memory_prob)
    ax.set_ylabel("$q_M(z_d)$")

    ax = axs[1]
    for i, (memory_element, probs) in enumerate(zip(memory, probss)):
        ax.plot(xs, probs, label=f"$z_d = {memory_element}$", color=f"C{memory_element}")
    ax.set_xlim(-support_size, support_size)
    ax.set_ylabel("$q(z_c | z_d)$")

    ax.legend()
    util.save_fig(fig, path)


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
    # while discrete in memory:
    #     discrete = propose_discrete(
    #         (), generative_model.support_size, generative_model.device
    #     )  # []
    # print(f"discrete = {discrete}")

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


def train(generative_model, guide, algorithm, num_particles, num_iterations, memory=None):
    optimizer = torch.optim.Adam(guide.parameters())
    losses = []

    for i in range(num_iterations):
        if "mws" in algorithm:
            if algorithm == "mws":
                loss_fn = get_mws_loss
                # print(f"memory = {memory[0]}")
            elif algorithm == "cmws":
                loss_fn = get_cmws_loss
            loss, memory = loss_fn(generative_model, guide, memory, num_particles)
            # print(f"memory = {memory}")
        else:
            if algorithm == "rws":
                loss_fn = get_rws_loss
            elif algorithm == "elbo":
                loss_fn = get_elbo_loss
            loss = loss_fn(generative_model, guide, num_particles)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.detach().item())
        if i % 10 == 0:
            print(f"Iteration {i} | Loss {losses[-1]:.2f}")
    return losses, memory


def plot_losses(path, losses):
    fig, ax = plt.subplots(1, 1)
    ax.plot(losses)
    util.save_fig(fig, path)


def init_discrete_memory(memory_size, support_size, device):
    if memory_size > support_size:
        raise ValueError("memory_size > support_size")
    return torch.arange(memory_size, device=device)


def main(args):
    # general
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        args.cuda = True
    else:
        device = torch.device("cpu")
        args.cuda = False

    print(f"TRAINING {args.algorithm}")
    generative_model = GenerativeModel(args.support_size, device)
    guide = Guide(args.support_size)
    if args.algorithm == "mws":
        memory = init_memory(args.memory_size, args.support_size, device)
    elif args.algorithm == "cmws":
        memory = init_discrete_memory(args.memory_size, args.support_size, device)
    else:
        memory = None

    losses, memory = train(
        generative_model,
        guide,
        args.algorithm,
        args.num_particles,
        args.num_iterations,
        memory=memory,
    )

    base_dir = f"save/toy_model/{args.algorithm}"
    plot_losses(f"{base_dir}/losses.pdf", losses)
    generative_model.plot(f"{base_dir}/generative_model.pdf")
    guide.plot(f"{base_dir}/guide.pdf")
    if args.algorithm == "mws":
        plot_memory(f"{base_dir}/memory.pdf", memory, generative_model.support_size)
    elif args.algorithm == "cmws":
        plot_continuous_memory(
            f"{base_dir}/memory.pdf", generative_model, guide, memory, args.num_particles
        )


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # general
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--support-size", type=int, default=5, help=" ")
    parser.add_argument("--memory-size", type=int, default=3, help=" ")
    parser.add_argument("--num-particles", type=int, default=100, help=" ")
    parser.add_argument("--num-iterations", type=int, default=10000, help=" ")
    parser.add_argument(
        "--algorithm",
        default="rws",
        choices=["rws", "elbo", "mws", "cmws"],
        help="Learning/inference algorithm to use",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
