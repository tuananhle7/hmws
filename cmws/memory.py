"""
Helper functions for the memory used in memoised wake sleep
"""
from cmws import util
import torch


def init_memory_group(num_obs, memory_size, event_shape, event_range):
    """Initializes a discrete latent variable, making sure it's unique.

    Args
        num_obs (int)
        memory_size (int)
        event_shape (list) shape of the latent variable group
        event_range
            low (int) inclusive
            high (int) exclusive

    Returns [num_obs, memory_size, *event_shape]"""
    # Extract
    low, high = event_range

    memory = []
    for _ in range(num_obs):
        while True:
            x = torch.randint(low, high, *[memory_size, *event_shape])

            if len(torch.unique(x, dim=0)) == memory_size:
                memory.append(x)
                break
    return torch.stack(memory)


class Memory:
    """Creates a memory object used in memoised wake-sleep.

    Args
        num_obs (int)
        memory_size (int)
        event_shapes
            list
                shape of the latent variable group
                (there is only one group)

            OR

            list of lists
                event_shapes[i] is the shape of the ith latent variable
                group
                (there is more than one group)
        event_ranges (list of lists)
            list
                min (int) inclusive
                max (int) exclusive

            OR

            list of lists
                event_ranges[i] = [min_i (inclusive), max_i (exclusive)]
                    is the range of the ith latent variable group
    """

    def __init__(self, num_obs, memory_size, event_shapes, event_ranges):
        # Determine number of groups
        if isinstance(event_shapes[0], int):
            self.num_groups = 1
            self.event_shapes = [event_shapes]
            self.event_ranges = [event_ranges]
        else:
            self.num_groups = len(event_shapes)
            assert self.num_groups > 1
            self.event_shapes = event_shapes
            self.event_ranges = event_ranges

        self.memory_groups = [
            init_memory_group(num_obs, memory_size, event_shape, event_range)
            for event_shape, event_range in zip(self.event_shapes, self.event_ranges)
        ]

    @property
    def device(self):
        return self.memory_groups[0].device

    def to(self, device):
        for group_id in range(self.num_groups):
            self.memory_groups[group_id] = self.memory_groups[group_id].to(device)

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
            [batch_size, memory_size, ...]

            OR

            list of tensors latent_groups
                latent_groups[i]'s shape is [batch_size, memory_size, *event_shape[i]]
        """
        latent = [memory_group[obs_id] for memory_group in self.memory_groups]
        if self.num_groups == 1:
            return latent[0]
        else:
            return latent

    def update(self, obs_id, latent):
        """
        Args
            obs_id [batch_size]
            latent
                [batch_size, memory_size, ...]

                OR

                list of tensors latent_groups
                    latent_groups[i]'s shape is [batch_size, memory_size, *event_shape[i]]
        """
        if self.num_groups == 1:
            self.memory_groups[0][obs_id] = latent
        else:
            for group_id in range(self.num_groups):
                self.memory_groups[group_id][obs_id] = latent[group_id]

    def sample(self, obs_id, obs, generative_model, sample_shape=[]):
        """
        Args
            obs_id [batch_size]
            obs [batch_size, *obs_dims]
            generative_model
            sample_shape (list like)

        Returns
            [*sample_shape, batch_size, memory_size, ...]

            OR

            list of tensors latent_groups
                latent_groups[i]'s shape is
                [*sample_shape, batch_size, memory_size, *event_shape[i]]
        """
        raise NotImplementedError

    def log_prob(self, obs_id, obs, latent, generative_model):
        """
        Args
            obs_id [batch_size]
            obs [batch_size, *obs_dims]
            generative_model
            latent
                [*shape, batch_size, memory_size, ...]

                OR

                list of tensors latent_groups
                    latent_groups[i]'s shape is [*shape, batch_size, memory_size, *event_shape[i]]

        Returns [*shape]
        """
        raise NotImplementedError


def get_unique_and_top_k(x, scores, k):
    """Removes duplicates in x, sorts them according to their scores and
    takes the top k.

    Args
        x (note that k <= n)
            tensor [*shape, n, *dims]

            or

            list of N tensors
            [*shape, n, *dims_1]
            ...
            [*shape, n, *dims_N]
        scores [*shape, n]
        k (int)

    Returns
        x_selected
            tensor [*shape, k, *dims]

            or

            list of N tensors
            [*shape, k, *dims_1]
            ...
            [*shape, k, *dims_N]
        scores_selected [*shape, k]
    """
    # Extract
    shape = scores.shape[:-1]
    n = scores.shape[-1]
    num_elements = util.get_num_elements(shape)
    if torch.is_tensor(x):
        dims = x.shape[(len(shape) + 1) :]
    else:
        dimss = [single_x.shape[(len(shape) + 1) :] for single_x in x]
        total_ndimss = [util.get_num_elements(dims) for dims in dimss]

    # Flatten
    # --Flatten x
    if torch.is_tensor(x):
        # [num_elements, n, total_ndims]
        x_flattened = x.view(num_elements, n, -1)
    else:
        # [num_elements, n, total_ndims_1 + ... + total_ndims_N]
        x_flattened = torch.cat([single_x.view(num_elements, n, -1) for single_x in x], dim=-1)

    # --Flatten scores
    scores_flattened = scores.view(num_elements, n)

    # Do everything unbatched
    x_top_k = []
    scores_top_k = []
    for element_id in range(num_elements):
        # Take the unique elements
        # [num_unique, total_ndims], [num_unique]
        x_unique, indices_unique = util.unique(x_flattened[element_id], dim=0)
        # [num_unique]
        scores_unique = scores_flattened[element_id][indices_unique]

        # Take the top k
        # --Sort
        scores_sorted, indices_sorted = scores_unique.sort(dim=0)  # [num_unique]

        # --Take top k
        x_top_k.append(x_unique[indices_sorted[-k:]])
        scores_top_k.append(scores_sorted[-k:])
    # [num_elements, k, total_ndims] or [num_elements, k, total_ndims_1 + ... + total_ndims_N]
    x_top_k = torch.stack(x_top_k)
    # [num_elements, k]
    scores_top_k = torch.stack(scores_top_k)

    # Unflatten
    # --Unflatten x
    if torch.is_tensor(x):
        # [*shape, k, *dims]
        x_selected = x_top_k.view(*[*shape, k, *dims])
    else:
        # Split up the xs
        # [*shape, k, *dims_1], ..., [*shape, k, *dims_N]
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
            x_selected.append(x_tmp.view(*[*shape, k, *dimss[i]]))

    # --Unflatten scores
    scores_selected = scores_top_k.view(*[*shape, k])

    return x_selected, scores_selected
