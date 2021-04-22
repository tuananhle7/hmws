"""
Helper functions for the memory used in memoised wake sleep
"""
from cmws import util
import torch


def init_memory_groups(num_obs, memory_size, generative_model):
    """Initializes a latent variables, making sure it's unique.

    Args
        num_obs (int)
        memory_size (int)
        generative_model -- only used for sampling from p(z_d)

    Returns list of
        [num_obs, memory_size, *event_shapes[i]]
    """
    latent_groupss = []
    for _ in range(num_obs):
        while True:
            latent_groups = generative_model.discrete_latent_sample([memory_size])

            # Force latent_groups into a 1-element list if it is a tensor
            if torch.is_tensor(latent_groups):
                latent_groups = [latent_groups]

            if len(torch.unique(flatten_tensors(latent_groups, 1), dim=0)) == memory_size:
                latent_groupss.append(latent_groups)
                break

    # Extract
    num_groups = len(latent_groupss[0])

    memory_groups = []
    for group_id in range(num_groups):
        memory_groups.append(torch.stack([latent_groupss[i][group_id] for i in range(num_obs)]))
    return memory_groups


class Memory:
    """Creates a memory object used in memoised wake-sleep.

    Args
        num_obs (int)
        size (int)
        generative_model - only used to initialize the memory by sampling from p(z_d)
    """

    def __init__(self, num_obs, size, generative_model):
        self.num_obs = num_obs
        self.size = size
        self.memory_groups = init_memory_groups(num_obs, size, generative_model)
        self.num_groups = len(self.memory_groups)

    @property
    def device(self):
        return self.memory_groups[0].device

    def to(self, device):
        for group_id in range(self.num_groups):
            self.memory_groups[group_id] = self.memory_groups[group_id].to(device)
        return self

    def state_dict(self):
        return [self.memory_groups[group_id] for group_id in range(self.num_groups)]

    def load_state_dict(self, state_dict):
        for group_id in range(self.num_groups):
            self.memory_groups[group_id] = state_dict[group_id]

    def select(self, obs_id):
        """
        Args
            obs_id [batch_size]

        Returns
            [size, batch_size, ...]

            OR

            list of tensors latent_groups
                latent_groups[i]'s shape is [size, batch_size, *event_shape[i]]
        """
        latent = [memory_group[obs_id].transpose(0, 1) for memory_group in self.memory_groups]
        if self.num_groups == 1:
            return latent[0]
        else:
            return latent

    def update(self, obs_id, latent):
        """
        Args
            obs_id [batch_size]
            latent
                [size, batch_size, ...]

                OR

                list of tensors latent_groups
                    latent_groups[i]'s shape is [size, batch_size, *event_shape[i]]
        """
        if self.num_groups == 1:
            self.memory_groups[0][obs_id] = latent.transpose(0, 1)
        else:
            for group_id in range(self.num_groups):
                self.memory_groups[group_id][obs_id] = latent[group_id].transpose(0, 1)

    def is_unique(self, obs_id=None):
        """
        Args
            obs_id None or [batch_size]

        Returns
            [num_obs] or [batch_size]
        """
        if obs_id is None:
            obs_id = torch.arange(self.num_obs, device=self.device)

        latent_groups = self.select(obs_id)
        if self.num_groups == 1:
            latent_groups = [latent_groups]

        latent_groups_flattened = flatten_tensors(latent_groups, 2)
        result = []
        for i in range(len(obs_id)):
            result.append(len(torch.unique(latent_groups_flattened[:, i], dim=0)) == self.size)
        return torch.tensor(result, device=self.device)

    def sample(self, obs_id, obs, generative_model, sample_shape=[]):
        """
        Args
            obs_id [batch_size]
            obs [batch_size, *obs_dims]
            generative_model
            sample_shape (list like)

        Returns
            [*sample_shape, batch_size, size, ...]

            OR

            list of tensors latent_groups
                latent_groups[i]'s shape is
                [*sample_shape, batch_size, size, *event_shape[i]]
        """
        raise NotImplementedError

    def log_prob(self, obs_id, obs, latent, generative_model):
        """
        Args
            obs_id [batch_size]
            obs [batch_size, *obs_dims]
            generative_model
            latent
                [*shape, batch_size, size, ...]

                OR

                list of tensors latent_groups
                    latent_groups[i]'s shape is [*shape, batch_size, size, *event_shape[i]]

        Returns [*shape]
        """
        raise NotImplementedError


def flatten_tensors(x, len_shape):
    """Flatten tensor / list of tensors into a single tensor

    Args
        x
            tensor [*shape, *dims]

            or

            list of N tensors
            [*shape, *dims_1]
            ...
            [*shape, *dims_N]
        len_shape (int) is the length of shape

    Returns
        [*shape, total_ndims]

        OR

        [*shape, total_ndims_1 + ... + total_ndims_N]
    """
    # Flatten
    # --Flatten x
    if torch.is_tensor(x):
        shape = x.shape[:len_shape]

        # [*shape, total_ndims]
        x_flattened = x.view(*[*shape, -1])
    else:
        shape = x[0].shape[:len_shape]

        # [*shape, total_ndims_1 + ... + total_ndims_N]
        x_flattened = torch.cat([single_x.view(*[*shape, -1]) for single_x in x], dim=-1)

    return x_flattened


def concat(x, y):
    """Concatenates latent x and latent y

    Args
        x, y
            tensor [n_x or n_y, batch_size, *dims]

            or

            list of N tensors
            [n_x or n_y, batch_size, *dims_1]
            ...
            [n_x or n_y, batch_size, *dims_N]

    Returns
            tensor [n_x + n_y, batch_size, *dims]

            or

            list of N tensors
            [n_x + n_y, batch_size, *dims_1]
            ...
            [n_x + n_y, batch_size, *dims_N]
    """
    if torch.is_tensor(x):
        return torch.cat([x, y], dim=0)
    else:
        return [torch.cat([x_, y_], dim=0) for x_, y_ in zip(x, y)]


def get_unique_and_top_k(x, scores, k):
    """Removes duplicates in x, sorts them according to their scores and
    takes the top k.

    Args
        x (note that k <= n)
            tensor [n, *shape, *dims]

            or

            list of N tensors
            [n, *shape, *dims_1]
            ...
            [n, *shape, *dims_N]
        scores [n, *shape]
        k (int)

    Returns
        x_selected
            tensor [k, *shape, *dims]

            or

            list of N tensors
            [k, *shape, *dims_1]
            ...
            [k, *shape, *dims_N]
        scores_selected [k, *shape]
    """
    # Extract
    shape = scores.shape[1:]
    n = scores.shape[0]
    num_elements = util.get_num_elements(shape)
    if torch.is_tensor(x):
        dims = x.shape[(len(shape) + 1) :]
    else:
        dimss = [single_x.shape[(len(shape) + 1) :] for single_x in x]
        total_ndimss = [util.get_num_elements(dims) for dims in dimss]

    # Flatten
    # --Flatten x
    x_flattened = flatten_tensors(x, 1 + len(shape)).view(n, num_elements, -1)

    # --Flatten scores
    scores_flattened = scores.view(n, num_elements)

    # Do everything unbatched
    x_top_k = []
    scores_top_k = []
    for element_id in range(num_elements):
        # Take the unique elements
        # [num_unique, total_ndims], [num_unique]
        x_unique, indices_unique = util.unique(x_flattened[:, element_id], dim=0)
        # [num_unique]
        scores_unique = scores_flattened[:, element_id][indices_unique]

        # Take the top k
        # --Sort
        scores_sorted, indices_sorted = scores_unique.sort(dim=0)  # [num_unique]

        # --Take top k
        x_top_k.append(x_unique[indices_sorted[-k:]])
        scores_top_k.append(scores_sorted[-k:])
    # [k, num_elements, total_ndims] or [k, num_elements, total_ndims_1 + ... + total_ndims_N]
    x_top_k = torch.stack(x_top_k, dim=1)
    # [k, num_elements]
    scores_top_k = torch.stack(scores_top_k, dim=1)

    # Check unique
    # for element_id in range(num_elements):
    #     if not len(torch.unique(x_top_k[:, element_id, :], dim=0)) == x_top_k.shape[0]:
    #         raise RuntimeError("elements not unique")

    # Unflatten
    # --Unflatten x
    if torch.is_tensor(x):
        # [k, *shape, *dims]
        x_selected = x_top_k.view(*[k, *shape, *dims])
    else:
        # Split up the xs
        # [k, *shape, *dims_1], ..., [k, *shape, *dims_N]
        x_selected = []
        start = 0
        end = total_ndimss[0]
        for i in range(len(x)):
            if i == 0:
                x_tmp = x_top_k[:, :, : total_ndimss[0]]
            elif i == len(x) - 1:
                x_tmp = x_top_k[:, :, -total_ndimss[-1] :]
            else:
                x_tmp = x_top_k[:, :, start:end]
                start = end
                end = start + total_ndimss[i + 1]
            x_selected.append(x_tmp.view(*[k, *shape, *dimss[i]]))

    # --Unflatten scores
    scores_selected = scores_top_k.view(*[k, *shape])

    return x_selected, scores_selected
