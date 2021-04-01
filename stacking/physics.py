"""
Implements the stability check of a stack of blocks described in Fig3 of
https://arxiv.org/abs/1804.08018
"""

import util
import torch


def get_stability(num_blocks, bottom_left, sizes, relationships, density=1.0):
    """Determine stability of a stack of square blocks

    Args
        num_blocks [*shape]
        bottom_left [*shape, max_num_blocks, 2]: xy coordinates of the bottom left
        sizes [*shape, max_num_blocks]
        relationships [*shape, max_num_blocks]: 0 means support, 1 means attached
        densities [*shape, max_num_blocks] or float (default: 1.0)

    Returns
        stability [*shape] (bool)
    """
    # Extract
    x, y = bottom_left[..., 0], bottom_left[..., 1]
    max_num_blocks = bottom_left.shape[-2]

    # Compute per block quantities
    # --Mass [*shape, max_num_blocks]
    masses = sizes ** 3 * density
    # --Center of mass [*shape, max_num_blocks, 2]
    centers_of_mass = bottom_left + sizes[..., None] / 2

    # Compute quantities for substacks (starting from a block up to the top block)
    substack_centers_of_mass = []  # [*shape, max_num_blocks, 2]
    substack_contact_intervals = []  # [*shape, max_num_blocks, 2]
    for bottom_block_id in range(max_num_blocks):
        # Center of mass of substack
        substack_centers_of_mass.append(
            get_center_of_mass(
                centers_of_mass[..., bottom_block_id:max_num_blocks, :],
                masses[..., bottom_block_id:max_num_blocks],
            )
        )

        # Contact interval of substack
        if bottom_block_id == 0:
            # The bottom block is fully in contact with the ground
            substack_contact_intervals.append(
                torch.stack(
                    [
                        x[..., bottom_block_id],
                        x[..., bottom_block_id] + sizes[..., bottom_block_id],
                    ],
                    dim=-1,
                )
            )
        else:
            substack_contact_intervals.append(
                get_contact_interval(
                    x[..., bottom_block_id],
                    sizes[..., bottom_block_id],
                    x[..., bottom_block_id - 1],
                    sizes[..., bottom_block_id - 1],
                )
            )
    substack_centers_of_mass = torch.stack(substack_centers_of_mass, dim=-2)
    substack_contact_intervals = torch.stack(substack_contact_intervals, dim=-2)

    # Compute stability of substacks
    # --Raw substack stability
    substack_centers_of_mass_x = substack_centers_of_mass[..., 0]
    # [*shape, max_num_blocks]
    substack_stability = get_is_inside_interval(
        substack_centers_of_mass_x, substack_contact_intervals
    )

    # --Mask stability of substacks that are > num_blocks
    substack_stability = util.pad_tensor(substack_stability, num_blocks, True)

    # --Mask stability of substacks that are attached
    ATTACHED = 1
    substack_stability[relationships == ATTACHED] = True

    return torch.all(substack_stability, dim=-1)


def get_center_of_mass(centers_of_mass, masses):
    """Determine the center of mass of a set of objects

    Args
        centers_of_mass [*shape, num_objects, 2]
        masses [*shape, num_objects]

    Returns [*shape, 2]
    """
    return (centers_of_mass * masses[..., None]).sum(-2) / masses.sum(-1)[..., None]


def get_contact_interval(left_x_1, size_1, left_x_2, size_2):
    """Determine contact interval (in x-coordinates) of two square blocks assuming one is on top of
    the other

    Args
        left_x_1 [*shape]
        size_1 [*shape]
        left_x_2 [*shape]
        size_2 [*shape]

    Returns [*shape, 2]
    """
    return get_intersection(
        torch.stack([left_x_1, left_x_1 + size_1], dim=-1),
        torch.stack([left_x_2, left_x_2 + size_2], dim=-1),
    )


def get_intersection(interval_1, interval_2):
    """Determine the intersection of two intervals

    Args
        interval_1 [*shape, 2]
        interval_2 [*shape, 2]

    Returns
        intersection [*shape, 2]: this will be nan if there is no intersection
    """
    # Extract
    left_1, right_1 = interval_1[..., 0], interval_1[..., 1]
    left_2, right_2 = interval_2[..., 0], interval_2[..., 1]
    shape = interval_1.shape[:-1]

    # Compute the intersection for cases in which there is an intersection
    intersection = torch.stack([torch.max(left_1, left_2), torch.min(right_1, right_2)], dim=-1)

    # Fill with nan if there isn't an interesection
    is_there_an_intersection = (right_1 < left_2) | (left_1 < right_2)
    intersection[is_there_an_intersection[..., None].expand(*[*shape, 2])] = float("nan")

    return intersection


def get_is_inside_interval(x, interval):
    """Is the point x inside the interval?

    Args|
        x [*shape]
        interval [*shape, 2]

    Returns [*shape]
    """
    # Extract
    left, right = interval[..., 0], interval[..., 1]
    return (x >= left) & (x <= right)


if __name__ == "__main__":
    device = "cuda"
    shape = [3]
    max_num_blocks = 5
    num_blocks = torch.randint(1, max_num_blocks + 1, shape, device=device)
    bottom_left = torch.rand(*[*shape, max_num_blocks, 2], device=device)
    sizes = torch.rand(*[*shape, max_num_blocks], device=device)
    relationships = torch.randint(2, [*shape, max_num_blocks], device=device)

    assert list(get_stability(num_blocks, bottom_left, sizes, relationships).shape) == shape
    print("Dims ok")
