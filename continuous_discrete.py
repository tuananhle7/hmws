import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import util


class GenerativeModel:
    def __init__(self, support_size, device):
        self.support_size = support_size
        self.device = device
        self.logits = torch.rand(support_size, device=device) - 0.5
        self.discrete_dist = torch.distributions.Categorical(logits=self.logits)

        self.locs = torch.arange(support_size, device=device)
        self.scales = torch.ones(support_size, device=device) * 0.1

    def get_continuous_dist(self, discrete):
        """
        Args:
            discrete: [batch_size]

        Returns: distribution of batch_shape [batch_size] and event_shape []
        """
        return torch.distributions.Normal(self.locs[discrete], self.scales[discrete])

    def log_prob(self, latent):
        """
        Args:
            latent:
                discrete: [batch_size]
                continuous: [batch_size]

        Return: [batch_size]
        """
        discrete, continuous = latent
        return self.discrete_dist.log_prob(discrete) + self.get_continuous_dist(discrete).log_prob(
            continuous
        )

    def plot(self, path):
        discrete = torch.arange(self.support_size, device=self.device)

        xs = torch.linspace(-1, self.support_size, steps=1000)

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

        ax.set_ylabel("$p(z_c | z_d)$")
        ax.legend()
        util.save_fig(fig, path)


class Guide(nn.Module):
    def __init__(self, support_size):
        super().__init__()
        self.support_size = support_size
        self.logits = nn.Parameter(torch.rand(support_size))

        self.locs = nn.Parameter(torch.rand(support_size))
        self.register_buffer("scales", torch.ones(support_size) * 0.1)

    @property
    def device(self):
        return self.logits.device

    @property
    def discrete_dist(self):
        return torch.distributions.Categorical(logits=self.logits)

    def get_continuous_dist(self, discrete):
        """
        Args:
            discrete: [batch_size]

        Returns: distribution of batch_shape [batch_size] and event_shape []
        """
        return torch.distributions.Normal(self.locs[discrete], self.scales[discrete])

    def log_prob(self, latent):
        """
        Args:
            latent:
                discrete: [batch_size]
                continuous: [batch_size]

        Return: [batch_size]
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

        xs = torch.linspace(-1, self.support_size, steps=1000)

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

        ax.set_ylabel("$q(z_c | z_d)$")
        ax.legend()
        util.save_fig(fig, path)


def get_rws_loss(generative_model, guide, num_particles):
    latent = guide.sample([num_particles])
    log_p = generative_model.log_prob(latent)
    log_q = guide.log_prob(latent)

    log_weight = log_p - log_q.detach()

    normalized_weight_detached = util.exponentiate_and_normalize(log_weight).detach()

    # wake phi
    wake_phi_loss = -torch.sum(normalized_weight_detached * log_q)

    # wake theta
    wake_theta_loss = -torch.sum(normalized_weight_detached * log_p)

    return wake_theta_loss + wake_phi_loss


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


def plot_memory(path, memory, support_size):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    ax = axs[0]
    ax.hist(
        memory[0], bins=torch.linspace(-0.5, support_size + 0.5, support_size + 1), density=True
    )
    ax.set_ylabel("$p(z_d)$")

    ax = axs[1]
    for i in range(support_size):
        ax.hist(memory[1][memory[0] == i], density=True, label=f"$z_d = {i}$")

    ax.set_ylabel("$p(z_c | z_d)$")
    ax.legend()
    util.save_fig(fig, path)


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
        loss: scalar that we call .backward() on and step the optimizer.
    """
    memory_size = memory[0].shape[0]

    # Propose latents from inference network
    latent = guide.sample([num_particles])  # [num_particles], [num_particles]

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

    theta_loss = -(log_p * normalized_weight).sum(dim=0).mean()
    phi_loss = -(log_q * normalized_weight).sum(dim=0).mean()

    return theta_loss + phi_loss


# TODO
def get_cmws_loss(generative_model, guide, memory, num_particles):
    """
    Args:
        generative_model: GenerativeModel object
        guide: Guide object
        memory: [memory_size]
        num_particles: int
    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
    """
    memory_size = memory.shape[0]

    # CONTINUE HERE

    # Propose latents from inference network
    latent = guide.sample([num_particles])  # [num_particles], [num_particles]

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

    theta_loss = -(log_p * normalized_weight).sum(dim=0).mean()
    phi_loss = -(log_q * normalized_weight).sum(dim=0).mean()

    return theta_loss + phi_loss


def train(generative_model, guide, algorithm, num_particles, num_iterations, memory=None):
    optimizer = torch.optim.Adam(guide.parameters())
    losses = []

    for i in range(num_iterations):
        if algorithm == "mws":
            loss = get_mws_loss(generative_model, guide, memory, num_particles)
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
    return losses


def plot_losses(path, losses):
    fig, ax = plt.subplots(1, 1)
    ax.plot(losses)
    util.save_fig(fig, path)


def main():
    device = "cpu"
    support_size = 5
    memory_size = 100
    num_particles = 100
    num_iterations = 10000
    algorithms = ["rws", "elbo", "mws"]

    for algorithm in algorithms:
        print(f"TRAINING {algorithm}")
        generative_model = GenerativeModel(support_size, device)
        guide = Guide(support_size)
        memory = init_memory(memory_size, support_size, device)

        losses = train(
            generative_model, guide, algorithm, num_particles, num_iterations, memory=memory
        )

        base_dir = f"save/toy_model/{algorithm}"
        plot_losses(f"{base_dir}/losses.pdf", losses)
        generative_model.plot(f"{base_dir}/generative_model.pdf")
        guide.plot(f"{base_dir}/guide.pdf")
        if algorithm == "mws":
            plot_memory(f"{base_dir}/memory.pdf", memory, generative_model.support_size)


if __name__ == "__main__":
    main()
