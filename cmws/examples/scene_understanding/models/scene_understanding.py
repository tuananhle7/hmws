import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from cmws import util
from cmws.examples.scene_understanding import render


class GenerativeModel(nn.Module):
    """
    """

    def __init__(
        self,
        num_grid_rows=3,
        num_grid_cols=3,
        num_primitives=5,
        max_num_blocks=3,
        im_size=32,
        obs_scale=1.0,
        obs_dist_type="normal",
    ):
        super().__init__()

        # Init
        self.num_grid_rows = num_grid_rows
        self.num_grid_cols = num_grid_cols
        self.num_primitives = num_primitives
        self.max_num_blocks = max_num_blocks
        self.num_channels = 3
        self.im_size = im_size
        self.obs_scale = obs_scale
        if obs_dist_type == "normal":
            self.obs_dist = torch.distributions.Normal
        elif obs_dist_type == "laplace":
            self.obs_dist = torch.distributions.Laplace

        # Primitive parameters (parameters of symbols)
        self.primitives = nn.ModuleList(
            [render.LearnableCube(f"{i}") for i in range(self.num_primitives)]
        )

        # Rendering parameters
        self.sigma = nn.Parameter(torch.randn(()))
        self.gamma = nn.Parameter(torch.randn(()))

    @property
    def device(self):
        return self.primitives[0].device

    @property
    def num_blocks_dist(self):
        """Prior distribution p(num_blocks)
        batch_shape [], event_shape [num_grid_rows, num_grid_cols]
        """
        return torch.distributions.Independent(
            torch.distributions.Categorical(
                logits=torch.ones(
                    (self.num_grid_rows, self.num_grid_cols, self.max_num_blocks + 1),
                    device=self.device,
                )
            ),
            reinterpreted_batch_ndims=2,
        )

    def latent_log_prob(self, latent):
        """Prior log p(z)

        Args:
            latent:
                num_blocks [*shape, num_grid_rows, num_grid_cols]
                stacking_program [*shape, num_grid_rows, num_grid_cols, max_num_blocks]
                raw_locations [*shape, num_grid_rows, num_grid_cols, max_num_blocks]

        Returns: [*shape]
        """
        # Extract
        num_blocks, stacking_program, raw_locations = latent

        # Log prob of num_blocks
        num_blocks_log_prob = self.num_blocks_dist.log_prob(num_blocks)

        # Log prob of stacking_program
        logits = torch.ones(
            (self.num_grid_rows, self.num_grid_cols, self.max_num_blocks, self.num_primitives),
            device=self.device,
        )
        stacking_program_log_prob = util.pad_tensor(
            torch.distributions.Categorical(logits=logits).log_prob(stacking_program), num_blocks, 0
        ).sum([-1, -2, -3])

        # Log prob of raw_locations
        # --Compute dist params
        loc = torch.zeros(
            (self.num_grid_rows, self.num_grid_cols, self.max_num_blocks), device=self.device
        )
        scale = torch.ones(
            (self.num_grid_rows, self.num_grid_cols, self.max_num_blocks), device=self.device
        )
        # --Compute log prob [*shape]
        raw_locations_log_prob = util.pad_tensor(
            torch.distributions.Normal(loc, scale).log_prob(raw_locations), num_blocks, 0,
        ).sum([-1, -2, -3])

        return num_blocks_log_prob + stacking_program_log_prob + raw_locations_log_prob

    def latent_sample(self, sample_shape=[]):
        """Sample from p(z)

        Args
            sample_shape

        Returns
            latent:
                num_blocks [*sample_shape, num_grid_rows, num_grid_cols]
                stacking_program [*sample_shape, num_grid_rows, num_grid_cols, max_num_blocks]
                raw_locations [*sample_shape, num_grid_rows, num_grid_cols, max_num_blocks]
        """
        # Sample discrete latents
        num_blocks, stacking_program = self.discrete_latent_sample(sample_shape)

        # Sample raw_locations
        # --Compute dist params
        loc = torch.zeros(
            self.num_grid_rows, self.num_grid_cols, self.max_num_blocks, device=self.device
        )
        scale = torch.ones(
            self.num_grid_rows, self.num_grid_cols, self.max_num_blocks, device=self.device
        )
        # --Compute log prob [*shape]
        raw_locations = util.pad_tensor(
            torch.distributions.Normal(loc, scale).sample(sample_shape), num_blocks, 0
        )

        return num_blocks, stacking_program, raw_locations

    def discrete_latent_sample(self, sample_shape=[]):
        """Sample from p(z_d)

        Args
            sample_shape

        Returns
            latent:
                num_blocks [*sample_shape, num_grid_rows, num_grid_cols]
                stacking_program [*sample_shape, num_grid_rows, num_grid_cols, max_num_blocks]
        """
        # Sample num_blocks
        num_blocks = self.num_blocks_dist.sample(sample_shape)

        # Sample stacking_program
        logits = torch.ones(
            (self.num_grid_rows, self.num_grid_cols, self.max_num_blocks, self.num_primitives),
            device=self.device,
        )
        stacking_program = util.pad_tensor(
            torch.distributions.Categorical(logits=logits).sample(sample_shape), num_blocks, 0
        )

        return num_blocks, stacking_program

    def get_obs_loc(self, latent):
        """Location parameter of p(x | z)

        Args:
            latent:
                num_blocks [*shape, num_grid_rows, num_grid_cols]
                stacking_program [*shape, num_grid_rows, num_grid_cols, max_num_blocks]
                raw_locations [*shape, num_grid_rows, num_grid_cols, max_num_blocks]

        Returns: [*shape, num_channels, im_size, im_size]
        """
        # Extract stuff
        num_blocks, stacking_program, raw_locations = latent

        # Compute stuff
        return render.render(
            self.primitives,
            num_blocks,
            stacking_program,
            raw_locations,
            im_size=self.im_size,
            sigma=self.sigma,
            gamma=self.gamma,
        )

    def log_prob(self, latent, obs):
        """Log joint probability of the generative model
        log p(z, x)

        Args:
            latent:
                num_blocks [*sample_shape, *shape, num_grid_rows, num_grid_cols]
                stacking_program
                    [*sample_shape, *shape, num_grid_rows, num_grid_cols, max_num_blocks]
                raw_locations [*sample_shape, *shape, num_grid_rows, num_grid_cols, max_num_blocks]
            obs [*shape, num_channels, im_size, im_size]

        Returns: [*sample_shape, *shape]
        """
        # TODO: tests still fail
        # p(z)
        latent_log_prob = self.latent_log_prob(latent)

        # p(x | z)
        obs_log_prob = torch.distributions.Independent(
            self.obs_dist(loc=self.get_obs_loc(latent), scale=self.obs_scale),
            reinterpreted_batch_ndims=3,
        ).log_prob(obs)

        return latent_log_prob + obs_log_prob

    def log_prob_discrete_continuous(self, discrete_latent, continuous_latent, obs):
        """Log joint probability of the generative model
        log p(z, x)

        Args:
            discrete_latent
                num_blocks [*discrete_shape, *shape, num_grid_rows, num_grid_cols]
                stacking_program [*discrete_shape, *shape, num_grid_rows, num_grid_cols,
                                  max_num_blocks]
            continuous_latent
                raw_locations [*continuous_shape, *discrete_shape, *shape, num_grid_rows,
                               num_grid_cols, max_num_blocks]
            obs [*shape, num_channels, im_size, im_size]

        Returns: [*continuous_shape, *discrete_shape, *shape]
        """
        # TODO: test
        # Extract
        num_blocks, stacking_program = discrete_latent
        raw_locations = continuous_latent
        shape = obs.shape[:-3]
        discrete_shape = num_blocks.shape[: -(len(shape) + 2)]
        continuous_shape = continuous_latent.shape[: -(3 + len(shape) + len(discrete_shape))]
        continuous_num_elements = util.get_num_elements(continuous_shape)

        # Expand discrete latent
        num_blocks_expanded = num_blocks[None].expand(
            *[
                continuous_num_elements,
                *discrete_shape,
                *shape,
                self.num_grid_rows,
                self.num_grid_cols,
            ]
        )
        stacking_program_expanded = stacking_program[None].expand(
            *[
                continuous_num_elements,
                *discrete_shape,
                *shape,
                self.num_grid_rows,
                self.num_grid_cols,
                self.max_num_blocks,
            ]
        )

        return self.log_prob((num_blocks_expanded, stacking_program_expanded, raw_locations), obs)

    @torch.no_grad()
    def sample(self, sample_shape=[]):
        """Sample from p(z, x)

        Args
            sample_shape

        Returns
            latent:
                num_blocks [*sample_shape, num_grid_rows, num_grid_cols]
                stacking_program [*sample_shape, num_grid_rows, num_grid_cols, max_num_blocks]
                raw_locations [*sample_shape, num_grid_rows, num_grid_cols, max_num_blocks]
            obs [*sample_shape, num_channels, im_size, im_size]
        """
        # p(z)
        latent = self.latent_sample(sample_shape)

        # p(x | z)
        obs = self.get_obs_loc(latent)

        return latent, obs


class Guide(nn.Module):
    """CNN going from image to order and locations
    """

    def __init__(self, num_primitives=3, max_num_blocks=3, im_size=32):
        super().__init__()

        # Init
        self.num_primitives = num_primitives
        self.max_num_blocks = max_num_blocks
        self.num_channels = 3
        self.im_size = im_size

        # Obs embedder
        self.obs_embedder = util.init_cnn(16, input_num_channels=3)
        self.obs_embedding_dim = 16  # determined by running the network forward
        # self.obs_embedding_dim = 400  # determined by running the network forward

        # Mapping from obs embedding to num_blocks params
        self.num_blocks_params_extractor = nn.Linear(self.obs_embedding_dim, self.max_num_blocks)

        # Mapping from obs embedding to stacking_program params
        self.stacking_program_params_extractor = nn.Linear(
            self.obs_embedding_dim, self.max_num_blocks * self.num_primitives
        )

        # Mapping from obs embedding to location params
        self.raw_location_params_extractor = nn.Linear(
            self.obs_embedding_dim, self.max_num_blocks * 2
        )

    @property
    def device(self):
        return next(self.obs_embedder.parameters()).device

    def get_dist_params(self, obs):
        """
        Args
            obs: [*shape, num_channels, im_size, im_size]

        Returns
            num_blocks_params [*shape, max_num_blocks]
            stacking_program_params [*shape, max_num_blocks, num_primitives]
            raw_locations_params
                loc [*shape, max_num_blocks]
                scale [*shape, max_num_blocks]
        """
        shape = obs.shape[:-3]
        obs_flattened = obs.reshape(-1, self.num_channels, self.im_size, self.im_size)

        # [num_elements, obs_embedding_dim]
        obs_embedding = self.obs_embedder(obs_flattened)

        # Num blocks
        num_blocks_params = self.num_blocks_params_extractor(obs_embedding).view(
            *shape, self.max_num_blocks
        )

        # Stacking program
        stacking_program_params = self.stacking_program_params_extractor(obs_embedding).view(
            *shape, self.max_num_blocks, self.num_primitives
        )

        # Raw location
        raw_location_params_extractor = self.raw_location_params_extractor(obs_embedding)
        loc = raw_location_params_extractor[..., : self.max_num_blocks].view(
            *shape, self.max_num_blocks
        )
        scale = (
            raw_location_params_extractor[..., self.max_num_blocks :]
            .exp()
            .view(*shape, self.max_num_blocks)
        )

        return num_blocks_params, stacking_program_params, (loc, scale)

    def log_prob(self, obs, latent):
        """
        Args
            obs [*shape, num_channels, im_size, im_size]
            latent:
                num_blocks [*sample_shape, *shape, num_grid_rows, num_grid_cols]
                stacking_program [*sample_shape, *shape, num_grid_rows, num_grid_cols,
                                  max_num_blocks]
                raw_locations [*sample_shape, *shape, num_grid_rows, num_grid_cols, max_num_blocks]

        Returns [*sample_shape, *shape]
        """
        # TODO
        # Extract
        num_blocks, stacking_program, raw_locations = latent

        # Compute params
        num_blocks_params, stacking_program_params, (loc, scale) = self.get_dist_params(obs)

        # log q(Num blocks | x)
        num_blocks_log_prob = util.CategoricalPlusOne(logits=num_blocks_params).log_prob(num_blocks)

        # log q(Stacking program | x)
        stacking_program_log_prob = util.pad_tensor(
            torch.distributions.Categorical(logits=stacking_program_params).log_prob(
                stacking_program
            ),
            num_blocks,
            0,
        ).sum(-1)

        # log q(Raw locations | x)
        raw_locations_log_prob = util.pad_tensor(
            torch.distributions.Normal(loc, scale).log_prob(raw_locations), num_blocks, 0,
        ).sum(-1)

        return num_blocks_log_prob + stacking_program_log_prob + raw_locations_log_prob

    def sample(self, obs, sample_shape=[]):
        """z ~ q(z | x)

        Args
            obs [*shape, num_channels, im_size, im_size]

        Returns
            num_blocks [*sample_shape, *shape, num_grid_rows, num_grid_cols]
            stacking_program [*sample_shape, *shape, num_grid_rows, num_grid_cols, max_num_blocks]
            raw_locations [*sample_shape, *shape, num_grid_rows, num_grid_cols, max_num_blocks]
        """
        # TODO
        # Compute params
        num_blocks_params, stacking_program_params, (loc, scale) = self.get_dist_params(obs)

        # Sample from q(Num blocks | x)
        num_blocks = util.CategoricalPlusOne(logits=num_blocks_params).sample(sample_shape)

        # Sample from q(Stacking program | x)
        stacking_program = util.pad_tensor(
            torch.distributions.Categorical(logits=stacking_program_params).sample(sample_shape),
            num_blocks,
            0,
        )

        # Sample from q(Raw locations | x)
        raw_locations = util.pad_tensor(
            torch.distributions.Normal(loc, scale).sample(sample_shape), num_blocks, 0
        )

        return num_blocks, stacking_program, raw_locations

    def sample_discrete(self, obs, sample_shape=[]):
        """z_d ~ q(z_d | x)

        Args
            obs [*shape, num_channels, im_size, im_size]
            sample_shape

        Returns
            num_blocks [*sample_shape, *shape, num_grid_rows, num_grid_cols]
            stacking_program [*sample_shape, *shape, num_grid_rows, num_grid_cols, max_num_blocks]
        """
        # TODO
        # Compute params
        num_blocks_params, stacking_program_params, _ = self.get_dist_params(obs)

        # Sample from q(Num blocks | x)
        num_blocks = util.CategoricalPlusOne(logits=num_blocks_params).sample(sample_shape)

        # Sample from q(Stacking program | x)
        stacking_program = util.pad_tensor(
            torch.distributions.Categorical(logits=stacking_program_params).sample(sample_shape),
            num_blocks,
            0,
        )

        return num_blocks, stacking_program

    def sample_continuous(self, obs, discrete_latent, sample_shape=[]):
        """z_c ~ q(z_c | z_d, x)

        Args
            obs [*shape, num_channels, im_size, im_size]
            discrete_latent
                num_blocks [*discrete_shape, *shape, num_grid_rows, num_grid_cols]
                stacking_program [*discrete_shape, *shape, num_grid_rows, num_grid_cols,
                                  max_num_blocks]
            sample_shape

        Returns
            raw_locations [*sample_shape, *discrete_shape, *shape, num_grid_rows, num_grid_cols,
                           max_num_blocks]
        """
        # TODO
        # Extract
        num_blocks, stacking_program = discrete_latent
        shape = obs.shape[:-3]
        discrete_shape = list(num_blocks.shape[: -len(shape)])
        num_elements = util.get_num_elements(sample_shape)

        # Compute params
        _, _, (loc, scale) = self.get_dist_params(obs)

        # Sample from q(Raw locations | x)
        raw_locations = util.pad_tensor(
            torch.distributions.Normal(loc, scale).sample(sample_shape + discrete_shape),
            num_blocks[None]
            .expand(*[num_elements, *discrete_shape, *shape])
            .view(*[*sample_shape, *discrete_shape, *shape]),
            0,
        )

        return raw_locations

    def log_prob_discrete(self, obs, discrete_latent):
        """log q(z_d | x)

        Args
            obs [*shape, num_channels, im_size, im_size]
            discrete_latent
                num_blocks [*discrete_shape, *shape, num_grid_rows, num_grid_cols]
                stacking_program
                    [*discrete_shape, *shape, num_grid_rows, num_grid_cols, max_num_blocks]

        Returns [*discrete_shape, *shape]
        """
        # TODO
        # Extract
        num_blocks, stacking_program = discrete_latent

        # Compute params
        num_blocks_params, stacking_program_params, _ = self.get_dist_params(obs)

        # log q(Num blocks | x)
        num_blocks_log_prob = util.CategoricalPlusOne(logits=num_blocks_params).log_prob(num_blocks)

        # log q(Stacking program | x)
        stacking_program_log_prob = util.pad_tensor(
            torch.distributions.Categorical(logits=stacking_program_params).log_prob(
                stacking_program
            ),
            num_blocks,
            0,
        ).sum(-1)

        return num_blocks_log_prob + stacking_program_log_prob

    def log_prob_continuous(self, obs, discrete_latent, continuous_latent):
        """log q(z_c | z_d, x)

        Args
            obs [*shape, num_channels, im_size, im_size]
            discrete_latent
                num_blocks [*discrete_shape, *shape, num_grid_rows, num_grid_cols]
                stacking_program
                    [*discrete_shape, *shape, num_grid_rows, num_grid_cols, max_num_blocks]
            continuous_latent (raw_locations)
                [*continuous_shape, *discrete_shape, *shape, num_grid_rows, num_grid_cols,
                 max_num_blocks]

        Returns [*continuous_shape, *discrete_shape, *shape]
        """
        # TODO
        # Extract
        num_blocks, stacking_program = discrete_latent
        raw_locations = continuous_latent
        shape = obs.shape[:-3]
        discrete_shape = num_blocks.shape[: -len(shape)]
        continuous_shape = continuous_latent.shape[: -(1 + len(shape) + len(discrete_shape))]
        continuous_num_elements = util.get_num_elements(continuous_shape)

        # Compute params
        _, _, (loc, scale) = self.get_dist_params(obs)

        # Expand num_blocks
        # [*continuous_shape, *discrete_shape, *shape]
        num_blocks_expanded = (
            num_blocks[None]
            .expand(*[continuous_num_elements, *discrete_shape, *shape])
            .view(*[*continuous_shape, *discrete_shape, *shape])
        )

        # log q(Raw locations | x)
        raw_locations_log_prob = util.pad_tensor(
            torch.distributions.Normal(loc, scale).log_prob(raw_locations), num_blocks_expanded, 0,
        ).sum(-1)

        return raw_locations_log_prob


if __name__ == "__main__":
    num_samples = 5
    path = "test/samples.png"
    generative_model = GenerativeModel()
    latent, obs = generative_model.sample((num_samples,))
    fig, axs = plt.subplots(1, num_samples, figsize=(num_samples * 4, 4))
    for i in range(num_samples):
        axs[i].imshow(obs[i].cpu().detach()[i])
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    util.save_fig(fig, path)