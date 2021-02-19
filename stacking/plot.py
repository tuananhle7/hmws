import util
import models
import os
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import render


def plot_stats(path, stats):
    fig, ax_losses = plt.subplots(1, 1)

    ax_losses.plot(stats.losses)
    ax_losses.set_xlabel("Iteration")
    ax_losses.set_ylabel("Loss")
    sns.despine(ax=ax_losses, trim=True)

    util.save_fig(fig, path)


def plot_reconstructions(path, generative_model, guide, obs):
    """
    Args:
        path (str)
        generative_model
        guide
        obs: [num_test_obs, num_channels, im_size, im_size]
    """
    num_test_obs, num_channels, im_size, _ = obs.shape

    # Sample latent
    traces = guide(obs)

    # Sample reconstructions
    # --Soft renders
    reconstructed_obs_soft = torch.stack(
        [
            render.soft_render(
                generative_model.primitives,
                trace["stacking_program"],
                trace["raw_locations"],
                generative_model.raw_color_sharpness,
                generative_model.raw_blur,
                num_channels=generative_model.num_channels,
                num_rows=generative_model.num_rows,
                num_cols=generative_model.num_cols,
            )
            for trace in traces
        ]
    )
    # --Hard renders
    reconstructed_obs_hard = torch.stack(
        [
            render.render(
                generative_model.primitives,
                trace["stacking_program"],
                trace["raw_locations"],
                num_channels=generative_model.num_channels,
                num_rows=generative_model.num_rows,
                num_cols=generative_model.num_cols,
            )
            for trace in traces
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
        ax.imshow(obs[sample_id].cpu().permute(1, 2, 0))

        # Plot probs
        ax = axss[1, sample_id]
        ax.imshow(reconstructed_obs_soft[sample_id].cpu().permute(1, 2, 0))
        ax.text(
            0.95,
            0.95,
            f"Inferred program: {traces[sample_id]['stacking_program']}"
            if sample_id == 0
            else f"{traces[sample_id]['stacking_program']}",
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            ha="right",
            color="gray",
        )

        # Plot probs > 0.5
        axss[2, sample_id].imshow(reconstructed_obs_hard[sample_id].cpu().permute(1, 2, 0))

    # Set labels
    axss[0, 0].set_ylabel("Observed image")
    axss[1, 0].set_ylabel("Soft reconstruction")
    axss[2, 0].set_ylabel("Hard reconstruction")

    util.save_fig(fig, path)


def plot_primitives(path, generative_model):
    device = generative_model.device
    im_size = generative_model.num_rows

    # Init
    location = torch.zeros((2,), device=device)
    blank_canvas = render.init_canvas(
        device,
        num_channels=generative_model.num_channels,
        num_rows=generative_model.num_rows,
        num_cols=generative_model.num_cols,
    )

    # Plot
    num_rows, num_cols = 2, generative_model.num_primitives
    fig, axss = plt.subplots(
        num_rows, num_cols, figsize=(2 * num_cols, 2 * num_rows), sharex=True, sharey=True
    )
    for ax in axss.flat:
        ax.set_xlim(0, im_size)
        ax.set_ylim(im_size, 0)
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(generative_model.num_primitives):
        obs_soft = render.soft_render_square(
            generative_model.primitives[i],
            location,
            blank_canvas,
            color_sharpness=render.get_color_sharpness(generative_model.raw_color_sharpness),
            blur=render.get_blur(generative_model.raw_blur),
        )
        obs_hard = render.render_square(generative_model.primitives[i], location, blank_canvas,)
        axss[0, i].imshow(obs_soft.cpu().permute(1, 2, 0))
        axss[1, i].imshow(obs_hard.cpu().permute(1, 2, 0))

    # Labels
    axss[0, 0].set_ylabel("Soft render")
    axss[1, 0].set_ylabel("Hard render")

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
            num_test_obs = 10
            # Test data
            obs = models.generate_from_true_generative_model(
                num_test_obs, num_primitives=run_args.data_num_primitives, device=device
            )

            # Plot
            plot_reconstructions(
                f"{util.get_save_dir(run_args)}/reconstructions/{num_iterations}.png",
                generative_model,
                guide,
                obs,
            )
            plot_primitives(
                f"{util.get_save_dir(run_args)}/primitives/{num_iterations}.png", generative_model,
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
