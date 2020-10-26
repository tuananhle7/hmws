import util
import numpy as np
import torch
import torch.nn as nn
import render


class GenerativeModel(nn.Module):
    def __init__(self, im_size=64):
        super().__init__()
        self.im_size = 64
        self.register_buffer("blank_canvas", torch.zeros((self.im_size, self.im_size)))

    @property
    def device(self):
        return self.blank_canvas.device

    def log_prob(self, latent, obs):
        """
        Args:
            latent
            obs

        Returns: [*shape]
        """
        raise NotImplementedError

    def sample(self, sample_shape=[]):
        """
        Args:
            sample_shape: list-like object (default [])

        Returns:
            latent [*sample_shape, 4]
            obs [*sample_shape, im_size, im_size]
        """
        # Sample latent
        one = torch.tensor(1.0, device=self.device)
        minus_one = -one
        min_x = torch.distributions.Uniform(minus_one, one).sample(sample_shape)
        max_x = torch.distributions.Uniform(min_x, one).sample()
        min_y = torch.distributions.Uniform(minus_one, one).sample(sample_shape)
        max_y = torch.distributions.Uniform(min_y, one).sample()
        latent = torch.stack([min_x, min_y, max_x, max_y], dim=-1)

        # Sample obs
        num_samples = int(np.prod(list(sample_shape)))
        latent_flattened = latent.view(-1, 4)
        obs = []
        for sample_id in range(num_samples):
            obs.append(render.render_rectangle(latent_flattened[sample_id], self.blank_canvas))
        obs = torch.stack(obs).view(*(*sample_shape, self.im_size, self.im_size))

        return latent, obs


class Guide(nn.Module):
    def __init__(self, im_size=64):
        super().__init__()
        self.im_size = im_size
        self.cnn = util.init_cnn(output_dim=16)
        self.cnn_features_dim = 400  # computed manually
        self.mlp = util.init_mlp(self.cnn_features_dim, 4 * 2, hidden_dim=100, num_layers=1)

    @property
    def device(self):
        return next(self.cnn.parameters()).device

    def get_cnn_features(self, obs):
        """
        Args:
            obs: [batch_size, im_size, im_size]

        Returns: [batch_size, cnn_features_dim]
        """
        batch_size = obs.shape[0]
        return self.cnn(obs[:, None]).view(batch_size, -1)

    def get_dist(self, obs):
        """
        Args:
            obs: [batch_size, im_size, im_size]

        Returns: dist with batch_size [batch_size] and event_shape [4]
        """
        cnn_features = self.get_cnn_features(obs)
        raw_loc, raw_scale = self.mlp(cnn_features).chunk(2, dim=-1)
        loc, scale = raw_loc, raw_scale.exp()
        return torch.distributions.Independent(
            torch.distributions.Normal(loc, scale), reinterpreted_batch_ndims=1
        )

    def log_prob(self, obs, latent):
        """
        Args:
            obs [batch_size, im_size, im_size]
            latent [batch_size, 4]

        Returns: [batch_size]
        """
        return self.get_dist(obs).log_prob(latent)

    def sample(self, obs, sample_shape=[]):
        """
        Args:
            obs [batch_size, im_size, im_size]

        Returns:
            latent [*sample_shape, batch_size, 4]
        """
        return self.get_dist(obs).sample()


if __name__ == "__main__":
    # Sample
    device = "cuda"
    generative_model = GenerativeModel().to(device)
    num_samples = 10
    latent, obs = generative_model.sample((num_samples,))

    import matplotlib.pyplot as plt
    import util

    fig, axs = plt.subplots(1, num_samples, figsize=(2 * num_samples, 2), sharex=True, sharey=True)
    for sample_id in range(num_samples):
        axs[sample_id].imshow(obs[sample_id].cpu(), cmap="Greys", vmin=0, vmax=1)
    util.save_fig(fig, "test.png")

    guide = Guide().to(device)
    log_prob = guide.log_prob(obs, latent)
    util.logging.info(f"guide log prob = {log_prob}")
