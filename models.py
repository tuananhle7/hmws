import torch.nn as nn
import torch
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

    def plot_discrete(self, ax):
        ax.bar(torch.arange(self.support_size), self.discrete_dist.probs.cpu())

    def plot_continuous(self, ax):
        discrete = torch.arange(self.support_size, device=self.device)
        xs = torch.linspace(-self.support_size, self.support_size, steps=1000, device=self.device)
        continuous_dist = self.get_continuous_dist(discrete)
        # [support_size, len(xs)]
        probss = continuous_dist.log_prob(xs[:, None].expand(-1, self.support_size)).T.exp()
        for i, probs in enumerate(probss):
            ax.plot(xs.cpu(), probs.cpu(), label=f"$z_d = {i}$", linewidth=2)
        ax.set_xlim(-self.support_size, self.support_size)

    def plot(self, path):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        ax = axs[0]
        self.plot_discrete(ax)
        ax.set_ylabel("$p(z_d)$")

        ax = axs[1]
        self.plot_continuous(ax)
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

    def plot_discrete(self, ax):
        ax.bar(torch.arange(self.support_size), self.discrete_dist.probs.cpu().detach())

    def plot_continuous(self, ax):
        discrete = torch.arange(self.support_size, device=self.device)

        xs = torch.linspace(-self.support_size, self.support_size, steps=1000, device=self.device)

        continuous_dist = self.get_continuous_dist(discrete)
        # [support_size, len(xs)]
        probss = (
            continuous_dist.log_prob(xs[:, None].expand(-1, self.support_size)).T.exp().detach()
        )
        for i, probs in enumerate(probss):
            ax.plot(xs.cpu(), probs.cpu(), label=f"$z_d = {i}$", linewidth=2)

    def plot(self, path):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        ax = axs[0]
        self.plot_discrete(ax)
        ax.set_ylabel("$q(z_d)$")

        ax = axs[1]
        self.plot_continuous(ax)
        ax.set_xlim(-self.support_size, self.support_size)
        ax.set_ylabel("$q(z_c | z_d)$")

        ax.legend()
        util.save_fig(fig, path)
