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
            x = torch.randint(low, high, [memory_size] + event_shape)

            if len(torch.unique(x, dim=0)) == memory_size:
                memory.append(x)
                break
    return torch.stack(memory)


class Memory:
    """Creates a memory object used in memoised wake-sleep.

    Args
        num_obs (int)
        size (int)
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

    def __init__(self, num_obs, size, event_shapes, event_ranges):
        self.num_obs = num_obs
        self.size = size

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
            init_memory_group(num_obs, size, event_shape, event_range)
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
    if torch.istensor(x):
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
    if torch.is_tensor(x):
        # [n, num_elements, total_ndims]
        x_flattened = x.view(n, num_elements, -1)
    else:
        # [n, num_elements, total_ndims_1 + ... + total_ndims_N]
        x_flattened = torch.cat([single_x.view(n, num_elements, -1) for single_x in x], dim=-1)

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
