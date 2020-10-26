import models
import os
import util
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


def plot_posterior(path, guide, obs):
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
        if os.path.exists(checkpoint_path):
            # Load checkpoint
            model, optimizer, stats, run_args = util.load_checkpoint(checkpoint_path, device=device)
            generative_model, guide = model

            # Test data
            device = "cuda"
            generative_model = models.GenerativeModel().to(device)
            num_test_obs = 10
            _, obs = generative_model.sample((num_test_obs,))

            # Plot
            plot_stats(f"{util.get_save_dir(run_args)}/stats.pdf", stats)
            plot_posterior(f"{util.get_save_dir(run_args)}/posterior.pdf", guide, obs)
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
