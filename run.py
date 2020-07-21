import torch
import util
import matplotlib.pyplot as plt


def topk(values, score, k):
    return values[torch.sort(score(values)).indices[-k:]]


def updated_memory(memory, propose, log_target, num_particles):
    memory_size = len(memory)
    particles = propose(num_particles)
    return topk(torch.unique(torch.cat([memory, particles])), log_target, memory_size)


class LogTargetDiscrete:
    def __init__(self, support_size):
        self.values = torch.rand(support_size)

    def __call__(self, latents):
        return self.values[latents]


class ProposeDiscrete:
    def __init__(self, support_size):
        self.support_size = support_size

    def __call__(self, num_particles):
        return torch.distributions.Categorical(logits=torch.zeros(self.support_size)).sample(
            (num_particles,)
        )


def log_target_continuous(latents):
    return torch.distributions.Normal(0, 1).log_prob(latents)


def propose_continuous(num_particles):
    return torch.distributions.Normal(0, 2).sample((num_particles,))


def plot_discrete_mws(path):
    support_size = 10
    log_target = LogTargetDiscrete(support_size)
    propose = ProposeDiscrete(support_size)

    memory_size = 3
    num_particles = memory_size

    memory = torch.arange(memory_size)

    num_updates = 10
    memories = [memory]
    for _ in range(num_updates):
        memories.append(updated_memory(memories[-1], propose, log_target, num_particles))

    fig, axs = plt.subplots(num_updates + 1, 1, figsize=(1 * 6, (num_updates + 1) * 4))
    for update_it in range(num_updates + 1):
        ax = axs[update_it]
        ax.set_title(f"Iteration {update_it}")
        ax.bar(
            torch.arange(support_size),
            util.lognormexp(log_target.values).exp(),
            alpha=0.5,
            label="true",
        )
        ax.bar(
            memories[update_it],
            util.lognormexp(log_target(memories[update_it])).exp(),
            alpha=0.5,
            label="memory-based approx.",
        )
        ax.legend()
    util.save_fig(fig, path)


def plot_continuous_mws(path):
    log_target = log_target_continuous
    propose = propose_continuous

    memory_size = 3
    num_particles = memory_size

    memory = torch.arange(memory_size).float()

    num_updates = 10
    memories = [memory]
    for _ in range(num_updates):
        memories.append(updated_memory(memories[-1], propose, log_target, num_particles))

    support = torch.linspace(-3, 3)
    fig, axs = plt.subplots(num_updates + 1, 1, figsize=(1 * 6, (num_updates + 1) * 4))
    for update_it in range(num_updates + 1):
        ax = axs[update_it]
        ax.set_title(f"Iteration {update_it}")
        ax.plot(support, log_target(support).exp(), label="true", color="C0")
        ax.bar(
            memories[update_it],
            util.lognormexp(log_target(memories[update_it])).exp(),
            width=0.2,
            label="memory-based approx.",
            color="C1",
        )
        ax.legend()
    util.save_fig(fig, path)


if __name__ == "__main__":
    plot_discrete_mws("save/discrete.png")
    plot_continuous_mws("save/continuous.png")
