import util
from models import rectangles
from models import hearts
from models import heartangles
from models import shape_program
from models import no_rectangle
from models import ldif_representation
from models import hearts_pyro
from models import ldif_representation_pyro
from models import shape_program_pyro
import os
import matplotlib.pyplot as plt
import torch
import seaborn as sns


def xy_to_im_coords(xy, im_size):
    """
    Args:
        xy [batch_size, 2] in [-1, 1] x [-1, 1]
        im_size

    Returns:
        im_coords [batch_size, 2] in [0, im_size] x [0, im_size]
    """
    im_coords = xy.clone()
    im_coords[:, 0] = (xy[:, 0] + 1) / 2 * im_size
    im_coords[:, 1] = im_size - (xy[:, 1] + 1) / 2 * im_size
    return im_coords


def plot_rectangles_posterior(path, guide, obs):
    """
    Args:
        guide
        obs: [num_test_obs, im_size, im_size]
    """
    num_test_obs, im_size, _ = obs.shape
    guide_dist = guide.get_dist(obs)

    # Compute guide params
    # [num_test_obs, 4]
    loc, scale = guide_dist.mean.detach(), guide_dist.stddev.detach()

    # [num_test_obs, 2]
    min_loc, max_loc = loc.chunk(2, dim=-1)
    min_scale, max_scale = scale.chunk(2, dim=-1)

    # Rescale
    min_loc, max_loc = xy_to_im_coords(min_loc, im_size), xy_to_im_coords(max_loc, im_size)
    min_scale = min_scale * im_size / 2
    max_scale = max_scale * im_size / 2

    # Plot
    num_rows = 1
    num_cols = num_test_obs
    fig, axs = plt.subplots(
        num_rows, num_cols, figsize=(2 * num_cols, 2 * num_rows), sharex=True, sharey=True
    )
    for ax in axs:
        ax.set_xlim(0, im_size)
        ax.set_ylim(im_size, 0)
        ax.set_xticks([])
        ax.set_yticks([])

    for sample_id in range(num_test_obs):
        axs[sample_id].imshow(obs[sample_id].cpu(), cmap="Greys", vmin=0, vmax=1)

        # Plot min
        util.plot_normal2d(
            axs[sample_id],
            min_loc[sample_id].cpu().numpy(),
            min_scale[sample_id].diag().cpu().numpy(),
            color="blue",
        )

        # Plot max
        util.plot_normal2d(
            axs[sample_id],
            max_loc[sample_id].cpu().numpy(),
            max_scale[sample_id].diag().cpu().numpy(),
            color="red",
        )

    util.save_fig(fig, path)


def plot_hearts_reconstructions(path, generative_model, guide, obs):
    """
    Args:
        generative_model
        guide
        obs: [num_test_obs, im_size, im_size]
    """
    num_test_obs, im_size, _ = obs.shape

    # Sample latent
    if isinstance(guide, hearts_pyro.Guide):
        latent = guide(obs)
    else:
        latent = guide.sample(obs)

    # Sample reconstructions
    reconstructed_obs = generative_model.get_obs_dist(latent).base_dist.probs

    # Plot
    num_rows = 3
    num_cols = num_test_obs
    fig, axss = plt.subplots(
        num_rows, num_cols, figsize=(2 * num_cols, 2 * num_rows), sharex=True, sharey=True
    )
    for ax in axss.flat:
        ax.set_xlim(0, im_size)
        ax.set_ylim(im_size, 0)
        ax.set_xticks([])
        ax.set_yticks([])

    for sample_id in range(num_test_obs):
        axss[0, sample_id].imshow(obs[sample_id].cpu(), cmap="Greys", vmin=0, vmax=1)
        axss[1, sample_id].imshow(reconstructed_obs[sample_id].cpu(), cmap="Greys", vmin=0, vmax=1)
        axss[2, sample_id].imshow(
            reconstructed_obs[sample_id].cpu() > 0.5, cmap="Greys", vmin=0, vmax=1
        )

    util.save_fig(fig, path)


def plot_heartangles_reconstructions(path, generative_model, guide, obs):
    """
    Args:
        generative_model
        guide
        obs: [num_test_obs, im_size, im_size]
        true_latent
    """
    num_test_obs, im_size, _ = obs.shape

    # Sample latent
    latent = guide.sample(obs)
    is_heart, (raw_position, raw_scale), rectangle_pose = latent

    # Sample reconstructions
    reconstructed_obs_probs = generative_model.get_obs_probs(latent)

    # Plot
    num_rows = 3
    num_cols = num_test_obs
    fig, axss = plt.subplots(
        num_rows, num_cols, figsize=(2 * num_cols, 2 * num_rows), sharex=True, sharey=True
    )
    for ax in axss.flat:
        ax.set_xlim(0, im_size)
        ax.set_ylim(im_size, 0)
        ax.set_xticks([])
        ax.set_yticks([])

    for sample_id in range(num_test_obs):
        # Plot obs
        axss[0, sample_id].imshow(obs[sample_id].cpu(), cmap="Greys", vmin=0, vmax=1)

        # Plot probs
        ax = axss[1, sample_id]
        ax.imshow(reconstructed_obs_probs[sample_id].cpu(), cmap="Greys", vmin=0, vmax=1)
        ax.text(
            0.95,
            0.95,
            heartangles.latent_to_str(
                (
                    is_heart[sample_id],
                    (raw_position[sample_id], raw_scale[sample_id]),
                    rectangle_pose[sample_id],
                )
            ),
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            ha="right",
            color="gray",
        )

        # Plot probs > 0.5
        axss[2, sample_id].imshow(
            reconstructed_obs_probs[sample_id].cpu() > 0.5, cmap="Greys", vmin=0, vmax=1
        )

    util.save_fig(fig, path)


def plot_shape_program_reconstructions(path, generative_model, guide, obs, ground_truth_latent):
    """
    Args:
        generative_model
        guide
        obs: [num_test_obs, im_size, im_size]
    """
    num_test_obs, im_size, _ = obs.shape

    # Deconstruct ground truth latent
    (
        ground_truth_program_id,
        (ground_truth_raw_positions, ground_truth_raw_scales),
        ground_truth_rectangle_poses,
    ) = ground_truth_latent

    # Sample latent
    latent = guide.sample(obs)
    program_id, (raw_positions, raw_scales), rectangle_poses = latent

    # Sample reconstructions
    reconstructed_obs_probs = generative_model.get_obs_probs(latent)

    # Plot
    num_rows = 3
    num_cols = num_test_obs
    fig, axss = plt.subplots(
        num_rows, num_cols, figsize=(2 * num_cols, 2 * num_rows), sharex=True, sharey=True
    )
    for ax in axss.flat:
        ax.set_xlim(0, im_size)
        ax.set_ylim(im_size, 0)
        ax.set_xticks([])
        ax.set_yticks([])

    for sample_id in range(num_test_obs):
        # Plot obs
        ax = axss[0, sample_id]
        ax.imshow(obs[sample_id].cpu(), cmap="Greys", vmin=0, vmax=1)
        ax.text(
            0.95,
            0.95,
            shape_program.latent_to_str(
                (
                    ground_truth_program_id[sample_id],
                    (ground_truth_raw_positions[sample_id], ground_truth_raw_scales[sample_id]),
                    ground_truth_rectangle_poses[sample_id],
                )
            ),
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            ha="right",
            color="gray",
        )

        # Plot probs
        ax = axss[1, sample_id]
        ax.imshow(reconstructed_obs_probs[sample_id].cpu(), cmap="Greys", vmin=0, vmax=1)
        ax.text(
            0.95,
            0.95,
            shape_program.latent_to_str(
                (
                    program_id[sample_id],
                    (raw_positions[sample_id], raw_scales[sample_id]),
                    rectangle_poses[sample_id],
                )
            ),
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            ha="right",
            color="gray",
        )

        # Plot probs > 0.5
        axss[2, sample_id].imshow(
            reconstructed_obs_probs[sample_id].cpu() > 0.5, cmap="Greys", vmin=0, vmax=1
        )

    util.save_fig(fig, path)


def plot_no_rectangle_reconstructions(path, generative_model, guide, obs, ground_truth_latent):
    """
    Args:
        generative_model
        guide
        obs: [num_test_obs, im_size, im_size]
    """
    num_test_obs, im_size, _ = obs.shape

    # Deconstruct ground truth latent
    (
        ground_truth_program_id,
        (ground_truth_raw_positions, ground_truth_raw_scales),
        ground_truth_rectangle_poses,
    ) = ground_truth_latent

    # Sample latent
    latent = guide.sample(obs)
    program_id, (raw_positions_1, raw_scales_1), (raw_positions_2, raw_scales_2) = latent

    # Sample reconstructions
    reconstructed_obs_probs = generative_model.get_obs_probs(latent)

    # Plot
    num_rows = 3
    num_cols = num_test_obs
    fig, axss = plt.subplots(
        num_rows, num_cols, figsize=(2 * num_cols, 2 * num_rows), sharex=True, sharey=True
    )
    for ax in axss.flat:
        ax.set_xlim(0, im_size)
        ax.set_ylim(im_size, 0)
        ax.set_xticks([])
        ax.set_yticks([])

    for sample_id in range(num_test_obs):
        # Plot obs
        ax = axss[0, sample_id]
        ax.imshow(obs[sample_id].cpu(), cmap="Greys", vmin=0, vmax=1)
        ax.text(
            0.95,
            0.95,
            heartangles.latent_to_str(
                (
                    ground_truth_program_id[sample_id],
                    (ground_truth_raw_positions[sample_id], ground_truth_raw_scales[sample_id]),
                    ground_truth_rectangle_poses[sample_id],
                )
            ),
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            ha="right",
            color="gray",
        )

        # Plot probs
        ax = axss[1, sample_id]
        ax.imshow(reconstructed_obs_probs[sample_id].cpu(), cmap="Greys", vmin=0, vmax=1)
        ax.text(
            0.95,
            0.95,
            no_rectangle.latent_to_str(
                (
                    program_id[sample_id],
                    (raw_positions_1[sample_id], raw_scales_1[sample_id]),
                    (raw_positions_2[sample_id], raw_scales_2[sample_id]),
                )
            ),
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            ha="right",
            color="gray",
        )

        # Plot probs > 0.5
        axss[2, sample_id].imshow(
            reconstructed_obs_probs[sample_id].cpu() > 0.5, cmap="Greys", vmin=0, vmax=1
        )

    util.save_fig(fig, path)


def plot_ldif_representation_reconstructions(
    path, generative_model, guide, obs, ground_truth_latent
):
    """
    Args:
        generative_model
        guide
        obs: [num_test_obs, im_size, im_size]
    """
    return plot_no_rectangle_reconstructions(
        path, generative_model, guide, obs, ground_truth_latent
    )


def plot_ldif_representation_pyro_reconstructions(
    path, generative_model, guide, obs, ground_truth_latent
):
    """
    Args:
        generative_model
        guide
        obs: [num_test_obs, im_size, im_size]
    """
    num_test_obs, im_size, _ = obs.shape

    # Deconstruct ground truth latent
    (
        ground_truth_program_id,
        (ground_truth_raw_positions, ground_truth_raw_scales),
        ground_truth_rectangle_poses,
    ) = ground_truth_latent

    # Sample latent
    if isinstance(guide, ldif_representation_pyro.Guide):
        latent = guide(obs)
        program_id, (raw_position, raw_scale) = latent
    else:
        latent = guide.sample(obs)
        program_id, (raw_positions_1, raw_scales_1), (raw_positions_2, raw_scales_2) = latent

    # Sample reconstructions
    reconstructed_obs_probs = generative_model.get_obs_probs(latent)

    # Plot
    num_rows = 3
    num_cols = num_test_obs
    fig, axss = plt.subplots(
        num_rows, num_cols, figsize=(2 * num_cols, 2 * num_rows), sharex=True, sharey=True
    )
    for ax in axss.flat:
        ax.set_xlim(0, im_size)
        ax.set_ylim(im_size, 0)
        ax.set_xticks([])
        ax.set_yticks([])

    for sample_id in range(num_test_obs):
        # Plot obs
        ax = axss[0, sample_id]
        ax.imshow(obs[sample_id].cpu(), cmap="Greys", vmin=0, vmax=1)
        ax.text(
            0.95,
            0.95,
            heartangles.latent_to_str(
                (
                    ground_truth_program_id[sample_id],
                    (ground_truth_raw_positions[sample_id], ground_truth_raw_scales[sample_id]),
                    ground_truth_rectangle_poses[sample_id],
                )
            ),
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            ha="right",
            color="gray",
        )

        # Plot probs
        ax = axss[1, sample_id]
        ax.imshow(reconstructed_obs_probs[sample_id].cpu(), cmap="Greys", vmin=0, vmax=1)
        ax.text(
            0.95,
            0.95,
            no_rectangle.shape_pose_to_str(
                (raw_position[sample_id], raw_scale[sample_id]), program_id[sample_id],
            ),
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            ha="right",
            color="gray",
        )

        # Plot probs > 0.5
        axss[2, sample_id].imshow(
            reconstructed_obs_probs[sample_id].cpu() > 0.5, cmap="Greys", vmin=0, vmax=1
        )

    util.save_fig(fig, path)


def plot_occupancy_network(path, generative_model):
    """
    Args:
        generative_model
    """
    device = generative_model.device
    im_size = generative_model.im_size
    grid_size = 5
    positions = torch.linspace(-0.5, 0.5, grid_size, device=device)

    # Plot
    num_rows, num_cols = grid_size, grid_size
    fig, axss = plt.subplots(
        num_rows, num_cols, figsize=(2 * num_cols, 2 * num_rows), sharex=True, sharey=True
    )
    for ax in axss.flat:
        ax.set_xlim(0, im_size)
        ax.set_ylim(im_size, 0)
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(num_rows):
        for j in range(num_cols):
            position = torch.stack([positions[j], positions[grid_size - 1 - i]])
            scale = torch.tensor(0.5, device=device)
            raw_position = util.logit(position + 0.5)
            raw_scale = util.logit((scale - 0.1) / 0.8)
            if isinstance(generative_model, hearts.GenerativeModel) or isinstance(
                generative_model, hearts_pyro.GenerativeModel
            ):
                obs = generative_model.get_obs_dist((raw_position, raw_scale)).base_dist.probs
            else:
                obs = generative_model.get_heart_obs_dist((raw_position, raw_scale)).base_dist.probs

            # axss[i, j].imshow(obs.cpu() > 0.5, cmap="Greys", vmin=0, vmax=1)
            axss[i, j].imshow(obs.cpu(), cmap="Greys", vmin=0, vmax=1)

    util.save_fig(fig, path)


def plot_occupancy_network_no_rectangle(path, generative_model):
    """
    Args:
        generative_model
    """
    device = generative_model.device
    im_size = generative_model.im_size

    # Shape pose
    position = torch.zeros((2,), device=device)
    scale = torch.tensor(0.5, device=device)
    raw_position = util.logit(position + 0.5)
    raw_scale = util.logit((scale - 0.1) / 0.8)

    # Plot
    num_rows, num_cols = 1, 2
    fig, axs = plt.subplots(
        num_rows, num_cols, figsize=(2 * num_cols, 2 * num_rows), sharex=True, sharey=True
    )
    for ax in axs.flat:
        ax.set_xlim(0, im_size)
        ax.set_ylim(im_size, 0)
        ax.set_xticks([])
        ax.set_yticks([])

    for i, ax in enumerate(axs):
        obs = generative_model.get_shape_obs_dist(i, (raw_position, raw_scale)).base_dist.probs
        ax.imshow(obs.cpu(), cmap="Greys", vmin=0, vmax=1)

    util.save_fig(fig, path)


def plot_occupancy_network_ldif_representation(path, generative_model):
    """
    Args:
        generative_model
    """
    device = generative_model.device
    im_size = generative_model.im_size

    # Shape pose
    position = torch.zeros((2,), device=device)
    scale = torch.tensor(0.5, device=device)
    raw_position = util.logit(position + 0.5)
    raw_scale = util.logit((scale - 0.1) / 0.8)

    # Plot
    num_rows, num_cols = 6, 2
    fig, axss = plt.subplots(
        num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), sharex=True, sharey=True
    )
    for ax in axss.flat:
        ax.set_xlim(0, im_size)
        ax.set_ylim(im_size, 0)
        ax.set_xticks([])
        ax.set_yticks([])

    for i, ax in enumerate(axss[0]):
        analytic_shape_density = generative_model._get_analytic_shape_density(
            i, (raw_position, raw_scale)
        )
        im = ax.imshow(analytic_shape_density.cpu())
        ax.set_title(f"$g_{i + 1}(x, \Sigma)$")
        fig.colorbar(im, ax=ax)
    for i, ax in enumerate(axss[1]):
        deep_shape_density = generative_model._get_deep_shape_density(i, (raw_position, raw_scale))
        im = ax.imshow(deep_shape_density.cpu())
        ax.set_title(f"$f_{i + 1}(x, \\theta)$")
        fig.colorbar(im, ax=ax)
    for i, ax in enumerate(axss[2]):
        deep_shape_density = generative_model._get_deep_shape_density(i, (raw_position, raw_scale))
        im = ax.imshow(deep_shape_density.cpu() + 1)
        ax.set_title(f"$f_{i + 1}(x, \\theta) + 1$")
        fig.colorbar(im, ax=ax)
    for i, ax in enumerate(axss[3]):
        deep_shape_density = generative_model._get_deep_shape_density(i, (raw_position, raw_scale))
        analytic_shape_density = generative_model._get_analytic_shape_density(
            i, (raw_position, raw_scale)
        )
        im = ax.imshow(((deep_shape_density + 1) * analytic_shape_density).cpu())
        ax.set_title(f"$LDIF_{i + 1}(x) = g_{i + 1}(x, \Sigma)(f_{i + 1}(x, \\theta) + 1)$")
        fig.colorbar(im, ax=ax)
    for i, ax in enumerate(axss[4]):
        deep_shape_density = generative_model._get_deep_shape_density(i, (raw_position, raw_scale))
        analytic_shape_density = generative_model._get_analytic_shape_density(
            i, (raw_position, raw_scale)
        )
        im = ax.imshow(
            ((deep_shape_density + 1) * analytic_shape_density).cpu() > 0,
            cmap="Greys",
            vmin=0,
            vmax=1,
        )
        ax.set_title(f"$LDIF_{i + 1}(x) > 0$")
        fig.colorbar(im, ax=ax)
    for i, ax in enumerate(axss[5]):
        obs = generative_model.get_shape_obs_dist(i, (raw_position, raw_scale)).base_dist.probs
        im = ax.imshow(obs.cpu(), cmap="Greys", vmin=0, vmax=1)
        ax.set_title(f"$sigmoid(LDIF_{i + 1}(x))$")
        fig.colorbar(im, ax=ax)

    util.save_fig(fig, path)


def plot_occupancy_network_neural_boundary_pyro(path, generative_model):
    """
    Args:
        generative_model
    """
    device = generative_model.device
    im_size = generative_model.im_size

    # Shape pose
    position = torch.zeros((2,), device=device)
    scale = torch.tensor(0.5, device=device)
    raw_position = util.logit(position + 0.5)
    raw_scale = util.logit((scale - 0.1) / 0.8)

    # Plot
    num_rows, num_cols = 1, generative_model.num_primitives
    fig, axs = plt.subplots(
        num_rows, num_cols, figsize=(2 * num_cols, 2 * num_rows), sharex=True, sharey=True
    )
    for ax in axs.flat:
        ax.set_xlim(0, im_size)
        ax.set_ylim(im_size, 0)
        ax.set_xticks([])
        ax.set_yticks([])

    for i, ax in enumerate(axs):
        obs = generative_model.get_shape_obs_logits(i, (raw_position, raw_scale)).sigmoid()
        ax.imshow(obs.cpu(), cmap="Greys", vmin=0, vmax=1)

    util.save_fig(fig, path)


def plot_neural_boundary_pyro_reconstructions(
    path, generative_model, guide, obs, ground_truth_latent
):
    """
    Args:
        generative_model
        guide
        obs: [num_test_obs, im_size, im_size]
    """
    num_test_obs, im_size, _ = obs.shape

    # Deconstruct ground truth latent
    (
        ground_truth_program_id,
        (ground_truth_raw_positions, ground_truth_raw_scales),
        ground_truth_rectangle_poses,
    ) = ground_truth_latent

    # Sample latent
    latent = guide(obs)
    shape_id, (raw_positions, raw_scales) = latent

    # Sample reconstructions
    reconstructed_obs_probs = torch.stack(
        [
            generative_model.get_shape_obs_logits(
                shape_id[i], (raw_positions[i], raw_scales[i])
            ).sigmoid()
            for i in range(num_test_obs)
        ]
    )

    # Plot
    num_rows = 3
    num_cols = num_test_obs
    fig, axss = plt.subplots(
        num_rows, num_cols, figsize=(2 * num_cols, 2 * num_rows), sharex=True, sharey=True
    )
    for ax in axss.flat:
        ax.set_xlim(0, im_size)
        ax.set_ylim(im_size, 0)
        ax.set_xticks([])
        ax.set_yticks([])

    for sample_id in range(num_test_obs):
        # Plot obs
        ax = axss[0, sample_id]
        ax.imshow(obs[sample_id].cpu(), cmap="Greys", vmin=0, vmax=1)
        ax.text(
            0.95,
            0.95,
            heartangles.latent_to_str(
                (
                    ground_truth_program_id[sample_id],
                    (ground_truth_raw_positions[sample_id], ground_truth_raw_scales[sample_id]),
                    ground_truth_rectangle_poses[sample_id],
                )
            ),
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            ha="right",
            color="gray",
        )

        # Plot probs
        ax = axss[1, sample_id]
        ax.imshow(reconstructed_obs_probs[sample_id].cpu(), cmap="Greys", vmin=0, vmax=1)
        ax.text(
            0.95,
            0.95,
            no_rectangle.shape_pose_to_str(
                (raw_positions[sample_id], raw_scales[sample_id]), shape_id[sample_id]
            ),
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            ha="right",
            color="gray",
        )

        # Plot probs > 0.5
        axss[2, sample_id].imshow(
            reconstructed_obs_probs[sample_id].cpu() > 0.5, cmap="Greys", vmin=0, vmax=1
        )

    util.save_fig(fig, path)


def plot_occupancy_network_shape_program_pyro(path, generative_model):
    """
    Args:
        generative_model
    """
    device = generative_model.device
    im_size = generative_model.im_size

    # Shape pose
    position = torch.zeros((2,), device=device)
    raw_position = util.logit(position + 0.5)

    # Plot
    num_rows, num_cols = 1, generative_model.num_primitives
    fig, axs = plt.subplots(
        num_rows, num_cols, figsize=(2 * num_cols, 2 * num_rows), sharex=True, sharey=True
    )
    for ax in axs.flat:
        ax.set_xlim(0, im_size)
        ax.set_ylim(im_size, 0)
        ax.set_xticks([])
        ax.set_yticks([])

    for i, ax in enumerate(axs):
        obs = generative_model.get_shape_obs_logits(i, raw_position).sigmoid()
        ax.imshow(obs.cpu(), cmap="Greys", vmin=0, vmax=1)

    util.save_fig(fig, path)


def plot_shape_program_pyro_reconstructions(
    path, generative_model, guide, obs, ground_truth_latent
):
    """
    Args:
        generative_model
        guide
        obs: [num_test_obs, im_size, im_size]
    """
    num_test_obs, im_size, _ = obs.shape

    # Deconstruct ground truth latent
    (
        ground_truth_program_id,
        ground_truth_raw_positions,
        ground_truth_rectangle_poses,
    ) = ground_truth_latent

    # Sample latent
    traces = guide(obs)

    # Sample reconstructions
    reconstructed_obs_probs = torch.stack(
        [generative_model.get_obs_probs(*trace) for trace in traces]
    )

    # Plot
    num_rows = 3
    num_cols = num_test_obs
    fig, axss = plt.subplots(
        num_rows, num_cols, figsize=(2 * num_cols, 2 * num_rows), sharex=True, sharey=True
    )
    for ax in axss.flat:
        ax.set_xlim(0, im_size)
        ax.set_ylim(im_size, 0)
        ax.set_xticks([])
        ax.set_yticks([])

    for sample_id in range(num_test_obs):
        # Plot obs
        ax = axss[0, sample_id]
        ax.imshow(obs[sample_id].cpu(), cmap="Greys", vmin=0, vmax=1)
        ax.text(
            0.95,
            0.95,
            shape_program.latent_to_str(
                (
                    ground_truth_program_id[sample_id],
                    ground_truth_raw_positions[sample_id],
                    ground_truth_rectangle_poses[sample_id],
                ),
                fixed_scale=True,
            ),
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            ha="right",
            color="gray",
        )

        # Plot probs
        ax = axss[1, sample_id]
        ax.imshow(reconstructed_obs_probs[sample_id].cpu(), cmap="Greys", vmin=0, vmax=1)
        ax.text(
            0.95,
            0.95,
            shape_program_pyro.trace_to_str(traces[sample_id]),
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            ha="right",
            color="gray",
        )

        # Plot probs > 0.5
        axss[2, sample_id].imshow(
            reconstructed_obs_probs[sample_id].cpu() > 0.5, cmap="Greys", vmin=0, vmax=1
        )

    util.save_fig(fig, path)


def plot_stats(path, stats):
    fig, ax_losses = plt.subplots(1, 1)

    ax_losses.plot(stats.losses)
    ax_losses.set_xlabel("Iteration")
    ax_losses.set_ylabel("Loss")
    sns.despine(ax=ax_losses, trim=True)

    util.save_fig(fig, path)


def main(args):
    # Cuda
    device = util.get_device()

    # Checkpoint paths
    if args.checkpoint_path is None:
        checkpoint_paths = list(util.get_checkpoint_paths())
    else:
        checkpoint_paths = [args.checkpoint_path]

    # Plot for all checkpoints
    for checkpoint_path in checkpoint_paths:
        # Fix seed
        util.set_seed(1)

        if os.path.exists(checkpoint_path):
            # Load checkpoint
            model, optimizer, stats, run_args = util.load_checkpoint(checkpoint_path, device=device)
            generative_model, guide = model
            num_iterations = len(stats.losses)

            # Plot stats
            plot_stats(f"{util.get_save_dir(run_args)}/stats.png", stats)

            # Plot reconstructions and other things
            num_test_obs = 30
            if run_args.model_type == "rectangles":
                # Test data
                generative_model = rectangles.GenerativeModel().to(device)
                latent, obs = generative_model.sample((num_test_obs,))

                # Plot
                plot_rectangles_posterior(
                    f"{util.get_save_dir(run_args)}/posterior/{num_iterations}.png", guide, obs
                )
            elif run_args.model_type == "hearts" or run_args.model_type == "hearts_pyro":
                # Test data
                true_generative_model = hearts.TrueGenerativeModel().to(device)
                latent, obs = true_generative_model.sample((num_test_obs,))

                # Replace generative model by the true generative model if algorithm is sleep
                if run_args.algorithm == "sleep":
                    generative_model = true_generative_model

                # Plot
                plot_hearts_reconstructions(
                    f"{util.get_save_dir(run_args)}/reconstructions/{num_iterations}.png",
                    generative_model,
                    guide,
                    obs,
                )
                plot_occupancy_network(
                    f"{util.get_save_dir(run_args)}/occupancy_network/{num_iterations}.png",
                    generative_model,
                )
            elif run_args.model_type == "heartangles":
                # Test data
                true_generative_model = heartangles.TrueGenerativeModel().to(device)
                latent, obs = true_generative_model.sample((num_test_obs,))

                # Replace generative model by the true generative model if algorithm is sleep
                if run_args.algorithm == "sleep":
                    generative_model = true_generative_model

                # Plot
                plot_heartangles_reconstructions(
                    f"{util.get_save_dir(run_args)}/reconstructions/{num_iterations}.png",
                    generative_model,
                    guide,
                    obs,
                )
                plot_occupancy_network(
                    f"{util.get_save_dir(run_args)}/occupancy_network/{num_iterations}.png",
                    generative_model,
                )
            elif run_args.model_type == "shape_program":
                # Test data
                true_generative_model = shape_program.TrueGenerativeModel().to(device)
                ground_truth_latent, obs = true_generative_model.sample((num_test_obs,))

                # Replace generative model by the true generative model if algorithm is sleep
                if run_args.algorithm == "sleep":
                    generative_model = true_generative_model

                # Plot
                plot_shape_program_reconstructions(
                    f"{util.get_save_dir(run_args)}/reconstructions/{num_iterations}.png",
                    generative_model,
                    guide,
                    obs,
                    ground_truth_latent,
                )
                plot_occupancy_network(
                    f"{util.get_save_dir(run_args)}/occupancy_network/{num_iterations}.png",
                    generative_model,
                )
            elif run_args.model_type == "no_rectangle" or run_args.model_type == "neural_boundary":
                # Test data
                true_generative_model = no_rectangle.TrueGenerativeModel(
                    has_shape_scale=run_args.data_has_shape_scale
                ).to(device)
                ground_truth_latent, obs = true_generative_model.sample((num_test_obs,))

                # Replace generative model by the true generative model if algorithm is sleep
                if run_args.algorithm == "sleep":
                    generative_model = true_generative_model

                # Plot
                plot_no_rectangle_reconstructions(
                    f"{util.get_save_dir(run_args)}/reconstructions/{num_iterations}.png",
                    generative_model,
                    guide,
                    obs,
                    ground_truth_latent,
                )
                plot_occupancy_network_no_rectangle(
                    f"{util.get_save_dir(run_args)}/occupancy_network/{num_iterations}.png",
                    generative_model,
                )
            elif run_args.model_type == "ldif_representation":
                # Test data
                true_generative_model = ldif_representation.TrueGenerativeModel().to(device)
                ground_truth_latent, obs = true_generative_model.sample((num_test_obs,))

                # Replace generative model by the true generative model if algorithm is sleep
                if run_args.algorithm == "sleep":
                    generative_model = true_generative_model

                # Plot
                plot_ldif_representation_reconstructions(
                    f"{util.get_save_dir(run_args)}/reconstructions/{num_iterations}.png",
                    generative_model,
                    guide,
                    obs,
                    ground_truth_latent,
                )
                plot_occupancy_network_ldif_representation(
                    f"{util.get_save_dir(run_args)}/occupancy_network/{num_iterations}.png",
                    generative_model,
                )
            elif run_args.model_type == "ldif_representation_pyro":
                # Test data
                true_generative_model = ldif_representation.TrueGenerativeModel().to(device)
                ground_truth_latent, obs = true_generative_model.sample((num_test_obs,))

                # Replace generative model by the true generative model if algorithm is sleep
                if run_args.algorithm == "sleep":
                    generative_model = true_generative_model

                # Plot
                plot_ldif_representation_pyro_reconstructions(
                    f"{util.get_save_dir(run_args)}/reconstructions/{num_iterations}.png",
                    generative_model,
                    guide,
                    obs,
                    ground_truth_latent,
                )
                plot_occupancy_network_ldif_representation(
                    f"{util.get_save_dir(run_args)}/occupancy_network/{num_iterations}.png",
                    generative_model,
                )
            elif run_args.model_type == "neural_boundary_pyro":
                # Test data
                true_generative_model = no_rectangle.TrueGenerativeModel(
                    has_shape_scale=run_args.data_has_shape_scale
                ).to(device)
                ground_truth_latent, obs = true_generative_model.sample((num_test_obs,))

                # Plot
                plot_neural_boundary_pyro_reconstructions(
                    f"{util.get_save_dir(run_args)}/reconstructions/{num_iterations}.png",
                    generative_model,
                    guide,
                    obs,
                    ground_truth_latent,
                )
                plot_occupancy_network_neural_boundary_pyro(
                    f"{util.get_save_dir(run_args)}/occupancy_network/{num_iterations}.png",
                    generative_model,
                )
            elif run_args.model_type == "shape_program_pyro":
                # Test data
                true_generative_model = shape_program.TrueGenerativeModelFixedScale().to(device)
                ground_truth_latent, obs = true_generative_model.sample((num_test_obs,))

                # Plot
                plot_shape_program_pyro_reconstructions(
                    f"{util.get_save_dir(run_args)}/reconstructions/{num_iterations}.png",
                    generative_model,
                    guide,
                    obs,
                    ground_truth_latent,
                )
                plot_occupancy_network_shape_program_pyro(
                    f"{util.get_save_dir(run_args)}/occupancy_network/{num_iterations}.png",
                    generative_model,
                )
        else:
            # Checkpoint doesn't exist
            util.logging.info(f"No checkpoint in {checkpoint_path}")


def get_parser():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--repeat", action="store_true", help="")
    parser.add_argument("--checkpoint-path", type=str, default=None, help=" ")

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    with torch.no_grad():
        if args.repeat:
            while True:
                main(args)
        else:
            main(args)
