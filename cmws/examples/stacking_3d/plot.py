import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from cmws import util
from cmws.examples.stacking_3d import data, render
from cmws.examples.stacking_3d import util as stacking_3d_util


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


def plot_reconstructions_stacking(path, generative_model, guide, obs):
    """
    Args:
        path (str)
        generative_model
        guide
        obs: [num_test_obs, num_channels, im_size, im_size]
    """
    num_test_obs, num_channels, im_size, _ = obs.shape
    hi_res_im_size = 256

    # Sample latent
    latent = guide.sample(obs)

    # Sample reconstructions
    # --Renders
    reconstructed_obs = generative_model.get_obs_loc(latent)

    # --Hi res renders
    generative_model.im_size = hi_res_im_size
    reconstructed_obs_hi_res = generative_model.get_obs_loc(latent)
    generative_model.im_size = im_size

    # Plot
    num_rows = 3
    num_cols = num_test_obs
    fig, axss = plt.subplots(num_rows, num_cols, figsize=(2 * num_cols, 2 * num_rows))
    for ax in axss.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    for sample_id in range(num_test_obs):
        # Plot obs
        ax = axss[0, sample_id]
        ax.imshow(obs[sample_id].cpu().permute(1, 2, 0))

        # Plot probs
        ax = axss[1, sample_id]
        ax.imshow(reconstructed_obs[sample_id].cpu().permute(1, 2, 0))

        # Plot probs > 0.5
        axss[2, sample_id].imshow(reconstructed_obs_hi_res[sample_id].cpu().permute(1, 2, 0))

    # Set labels
    axss[0, 0].set_ylabel("Observed image")
    axss[1, 0].set_ylabel("Reconstruction")
    axss[2, 0].set_ylabel("Hi-res reconstruction")

    util.save_fig(fig, path)


def plot_primitives_stacking(path, generative_model):
    device = generative_model.device
    im_size = generative_model.im_size
    hi_res_im_size = 256

    # Init
    location = torch.tensor([0, 0, -1], device=device).float()

    # Plot
    num_rows, num_cols = 2, generative_model.num_primitives
    fig, axss = plt.subplots(
        num_rows, num_cols, figsize=(2 * num_cols, 2 * num_rows), squeeze=False,
    )
    for ax in axss.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(generative_model.num_primitives):
        util.logging.info(f"Primitive {i} = {generative_model.primitives[i]}")
        obs = render.render_cube(
            generative_model.primitives[i].size,
            generative_model.primitives[i].color,
            location,
            im_size=im_size,
        )
        obs_high_res = render.render_cube(
            generative_model.primitives[i].size,
            generative_model.primitives[i].color,
            location,
            im_size=hi_res_im_size,
        )
        axss[0, i].imshow(obs.cpu())
        axss[1, i].imshow(obs_high_res.cpu())

    # Labels
    axss[0, 0].set_ylabel("Primitives")
    axss[1, 0].set_ylabel("Hi-res primitives")

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
            model, optimizer, stats, run_args = stacking_3d_util.load_checkpoint(
                checkpoint_path, device=device
            )
            generative_model, guide = model["generative_model"], model["guide"]
            num_iterations = len(stats.losses)

            # Plot stats
            plot_stats(f"{util.get_save_dir(run_args)}/stats.png", stats)

            # Plot reconstructions and other things
            # Test data
            obs = data.generate_test_obs(device)

            # Plot
            if run_args.model_type == "stacking":
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
