"""
Helper functions for the memory used in memoised wake sleep
"""
from cmws import util
import torch


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
