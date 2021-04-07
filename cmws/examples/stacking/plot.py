import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from cmws import util
from cmws.examples.stacking import data, render
from cmws.examples.stacking import util as stacking_util


def plot_stats(path, stats):
    if len(stats.sleep_pretraining_losses) > 0:
        fig, (ax_sleep_pretraining_losses, ax_losses, ax_logp, ax_kl) = plt.subplots(
            1, 4, figsize=(24, 4)
        )

        # Sleep pretraining Loss
        ax = ax_sleep_pretraining_losses
        ax.plot(stats.sleep_pretraining_losses)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Sleep Pretraining Loss")
        sns.despine(ax=ax, trim=True)
    else:
        fig, (ax_losses, ax_logp, ax_kl) = plt.subplots(1, 3, figsize=(18, 4))

    # Loss
    ax = ax_losses
    ax.plot(stats.losses)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    sns.despine(ax=ax, trim=True)

    # Logp
    ax = ax_logp
    ax.plot([x[0] for x in stats.log_ps], [x[1] for x in stats.log_ps])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Log p")
    sns.despine(ax=ax, trim=True)

    # KL
    ax = ax_kl
    ax.plot([x[0] for x in stats.kls], [x[1] for x in stats.kls])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("KL")
    sns.despine(ax=ax, trim=True)

    util.save_fig(fig, path)


def plot_reconstructions_stacking_pyro(path, generative_model, guide, obs):
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


def plot_primitives_stacking_pyro(path, generative_model):
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
        num_rows,
        num_cols,
        figsize=(2 * num_cols, 2 * num_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for ax in axss.flat:
        ax.set_xlim(0, im_size)
        ax.set_ylim(im_size, 0)
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(generative_model.num_primitives):
        util.logging.info(f"Primitive {i} = {generative_model.primitives[i]}")
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


def plot_reconstructions_one_primitive(path, generative_model, guide, obs):
    """
    Args:
        path (str)
        generative_model
        guide
        obs: [num_test_obs, num_channels, im_size, im_size]
    """
    num_test_obs, num_channels, im_size, _ = obs.shape

    # Sample latent
    latent = guide.sample(obs)

    # Sample reconstructions
    # --Soft renders
    reconstructed_obs_soft = generative_model.get_obs_loc(latent)

    # --Hard renders
    reconstructed_obs_hard = generative_model.get_obs_loc_hard(latent)

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

        # Plot probs > 0.5
        axss[2, sample_id].imshow(reconstructed_obs_hard[sample_id].cpu().permute(1, 2, 0))

    # Set labels
    axss[0, 0].set_ylabel("Observed image")
    axss[1, 0].set_ylabel("Soft reconstruction")
    axss[2, 0].set_ylabel("Hard reconstruction")

    util.save_fig(fig, path)


def plot_reconstructions_two_primitives(path, generative_model, guide, obs):
    """
    Args:
        path (str)
        generative_model
        guide
        obs: [num_test_obs, num_channels, im_size, im_size]
    """
    plot_reconstructions_one_primitive(path, generative_model, guide, obs)


def plot_primitives_two_primitives(path, generative_model):
    plot_primitives_stacking_pyro(path, generative_model)


def plot_reconstructions_stacking(path, generative_model, guide, obs):
    """
    Args:
        path (str)
        generative_model
        guide
        obs: [num_test_obs, num_channels, im_size, im_size]
    """
    plot_reconstructions_one_primitive(path, generative_model, guide, obs)


def plot_primitives_stacking(path, generative_model):
    plot_primitives_stacking_pyro(path, generative_model)


def plot_reconstructions_stacking_top_down(path, generative_model, guide, obs):
    """
    Args:
        path (str)
        generative_model
        guide
        obs: [num_test_obs, num_channels, im_size, im_size]
    """
    num_test_obs, num_channels, im_size, _ = obs.shape
    num_reconstructions = 5

    # Sample latent
    latent = guide.sample(obs, (num_reconstructions,))

    # Sample reconstructions
    # --Soft renders
    reconstructed_obs_soft = generative_model.get_obs_loc(latent)

    # --Hard renders
    # reconstructed_obs_hard = generative_model.get_obs_loc_hard(latent)
    reconstructed_obs_hard_front = generative_model.get_obs_loc_hard_front(latent)

    # Log prob
    generative_model_log_prob = generative_model.log_prob(
        latent,
        obs[None].expand(*[num_reconstructions, num_test_obs, num_channels, im_size, im_size]),
    )

    # Plot
    num_rows = 1 + 2 * num_reconstructions
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
        ax.patch.set_facecolor("lightgray")

        for reconstruction_id in range(num_reconstructions):
            # Plot probs
            ax = axss[1 + reconstruction_id * 2, sample_id]
            ax.imshow(reconstructed_obs_soft[reconstruction_id, sample_id].cpu().permute(1, 2, 0))
            ax.text(
                0.95,
                0.95,
                f"log p(z, x) = {generative_model_log_prob[reconstruction_id, sample_id]:.0f}",
                transform=ax.transAxes,
                fontsize=7,
                va="top",
                ha="right",
                color="gray",
            )

            # Plot probs > 0.5
            axss[2 + reconstruction_id * 2, sample_id].imshow(
                reconstructed_obs_hard_front[reconstruction_id, sample_id].cpu().permute(1, 2, 0)
            )

    # Set labels
    axss[0, 0].set_ylabel("Observed image")
    for reconstruction_id in range(num_reconstructions):
        axss[1 + reconstruction_id * 2, 0].set_ylabel(f"FRONT VIEW {reconstruction_id}")
        axss[2 + reconstruction_id * 2, 0].set_ylabel(f"TOP VIEW {reconstruction_id}")
    # axss[3, 0].set_ylabel("Front view")

    util.save_fig(fig, path)


def plot_primitives_stacking_top_down(path, generative_model):
    plot_primitives_stacking_pyro(path, generative_model)


def plot_reconstructions_stacking_with_attachment(path, generative_model, guide, obs):
    """
    Args:
        path (str)
        generative_model
        guide
        obs: [num_test_obs, num_channels, im_size, im_size]
    """
    num_test_obs, num_channels, im_size, _ = obs.shape

    # Sample latent
    latent = guide.sample(obs)
    num_blocks, (stacking_order, attachment), raw_locations = latent

    # Sample reconstructions
    # --Soft renders
    reconstructed_obs_soft = generative_model.get_obs_loc(latent)

    # --Hard renders
    reconstructed_obs_hard = generative_model.get_obs_loc_hard(latent)

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
        text = (
            "Stacking order: "
            + f"{list(stacking_order[sample_id, :num_blocks[sample_id]].detach().cpu().numpy())}"
            + f"\nAttachments: "
            f"{list(attachment[sample_id, :num_blocks[sample_id]].detach().cpu().numpy())}"
        )
        ax.text(
            0.95,
            0.95,
            text,
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


def plot_primitives_stacking_with_attachment(path, generative_model):
    plot_primitives_stacking_pyro(path, generative_model)


def main(args):
    # Cuda
    device = util.get_device()

    # Checkpoint paths
    if args.checkpoint_path is None:
        checkpoint_paths = list(util.get_checkpoint_paths())
    else:
        checkpoint_paths = [args.checkpoint_path]

    # # Plot log p for all checkpoints
    # fig, ax = plt.subplots(1, 1)

    # colors = {0.0: "blue", 0.25: "C1", 0.5: "C2", 0.75: "C4", 1.0: "C5"}
    # for checkpoint_path in checkpoint_paths:
    #     # Fix seed
    #     util.set_seed(1)

    #     if os.path.exists(checkpoint_path):
    #         # Load checkpoint
    #         model, optimizer, stats, run_args = util.load_checkpoint(checkpoint_path, device=device)
    #         generative_model, guide = model["generative_model"], model["guide"]
    #         num_iterations = len(stats.losses)

    #         # Logp
    #         ax.plot(
    #             [x[0] for x in stats.log_ps],
    #             [x[1] for x in stats.log_ps],
    #             label=util.get_path_base_from_args(run_args),
    #             linestyle="dashed" if run_args.num_sleep_pretraining_iterations > 0 else "solid",
    #             color=colors[run_args.insomnia],
    #         )
    # ax.set_xlim(0, 50000)
    # ax.legend()
    # ax.set_xlabel("Iteration")
    # ax.set_ylabel("Log p")
    # sns.despine(ax=ax, trim=True)
    # util.save_fig(fig, "losses.png", dpi=200)
    # return

    # Plot for all checkpoints
    for checkpoint_path in checkpoint_paths:
        # Fix seed
        util.set_seed(1)

        if os.path.exists(checkpoint_path):
            # Load checkpoint
            model, optimizer, stats, run_args = stacking_util.load_checkpoint(
                checkpoint_path, device=device
            )
            generative_model, guide = model["generative_model"], model["guide"]
            num_iterations = len(stats.losses)

            # Plot stats
            plot_stats(f"{util.get_save_dir(run_args)}/stats.png", stats)

            # Plot reconstructions and other things
            # Test data
            obs = data.generate_test_obs(run_args, device)

            # Plot
            if run_args.model_type == "stacking_pyro":
                plot_reconstructions_stacking_pyro(
                    f"{util.get_save_dir(run_args)}/reconstructions/{num_iterations}.png",
                    generative_model,
                    guide,
                    obs,
                )
                plot_primitives_stacking_pyro(
                    f"{util.get_save_dir(run_args)}/primitives/{num_iterations}.png",
                    generative_model,
                )
            elif run_args.model_type == "one_primitive":
                plot_reconstructions_one_primitive(
                    f"{util.get_save_dir(run_args)}/reconstructions/{num_iterations}.png",
                    generative_model,
                    guide,
                    obs,
                )
            elif run_args.model_type == "two_primitives":
                plot_reconstructions_two_primitives(
                    f"{util.get_save_dir(run_args)}/reconstructions/{num_iterations}.png",
                    generative_model,
                    guide,
                    obs,
                )
                plot_primitives_two_primitives(
                    f"{util.get_save_dir(run_args)}/primitives/{num_iterations}.png",
                    generative_model,
                )
            elif run_args.model_type == "stacking":
                plot_reconstructions_stacking(
                    f"{util.get_save_dir(run_args)}/reconstructions/{num_iterations}.png",
                    generative_model,
                    guide,
                    obs,
                )
                plot_primitives_stacking(
                    f"{util.get_save_dir(run_args)}/primitives/{num_iterations}.png",
                    generative_model,
                )
            elif run_args.model_type == "stacking_top_down":
                plot_reconstructions_stacking_top_down(
                    f"{util.get_save_dir(run_args)}/reconstructions/{num_iterations}.png",
                    generative_model,
                    guide,
                    obs,
                )
                plot_primitives_stacking_top_down(
                    f"{util.get_save_dir(run_args)}/primitives/{num_iterations}.png",
                    generative_model,
                )
            elif run_args.model_type == "stacking_with_attachment":
                plot_reconstructions_stacking_with_attachment(
                    f"{util.get_save_dir(run_args)}/reconstructions/{num_iterations}.png",
                    generative_model,
                    guide,
                    obs,
                )
                plot_primitives_stacking_with_attachment(
                    f"{util.get_save_dir(run_args)}/primitives/{num_iterations}.png",
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
