import os

import cmws
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from cmws import util
from cmws.examples.timeseries import data, run
from cmws.examples.timeseries import util as timeseries_util
from cmws.examples.timeseries import lstm_util


def plot_obs(ax, obs):
    """
    Args
        ax
        obs [num_timesteps]
    """
    ax.plot(obs.cpu().numpy(), color="C0")
    ax.set_ylim(-4, 4)
    ax.set_xticks([])
    ax.set_yticks([-4, 4])
    ax.tick_params(axis="y", direction="in")


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


def plot_reconstructions_timeseries(path, generative_model, guide, obs):
    """
    Args:
        path (str)
        generative_model
        guide
        obs: [num_test_obs, num_timesteps]
    """
    num_samples = 5
    num_test_obs, num_timesteps = obs.shape

    # Sample latent
    latent = guide.sample(obs, [num_samples])
    x, eos, _ = latent
    num_chars = lstm_util.get_num_timesteps(eos)

    # Sample reconstructions
    reconstructed_obs = generative_model.sample_obs(latent)

    # Plot
    num_rows = 1 + num_samples
    num_cols = num_test_obs
    fig, axss = plt.subplots(
        num_rows, num_cols, figsize=(3 * num_cols, 2 * num_rows), sharex=True, sharey=True
    )

    for test_obs_id in range(num_test_obs):
        # Plot obs
        ax = axss[0, test_obs_id]
        plot_obs(ax, obs[test_obs_id])

        for sample_id in range(num_samples):
            # Plot probs
            ax = axss[1 + sample_id, test_obs_id]
            plot_obs(ax, reconstructed_obs[sample_id, test_obs_id])
            expression = timeseries_util.get_expression(
                x[sample_id, test_obs_id][: num_chars[sample_id, test_obs_id]]
            )
            ax.text(
                0.95,
                0.95,
                f"Inferred kernel: {expression}",
                transform=ax.transAxes,
                fontsize=7,
                va="top",
                ha="right",
                color="gray",
            )

    # Set labels
    axss[0, 0].set_ylabel("Observed signal")
    for sample_id in range(num_samples):
        axss[1 + sample_id, 0].set_ylabel(f"Reconstruction {sample_id}")

    util.save_fig(fig, path)


def plot_predictions_timeseries(path, generative_model, guide, obs):
    """
    Args:
        path (str)
        generative_model
        guide
        obs: [num_test_obs, num_timesteps]
    """
    num_samples = 5
    num_test_obs, num_timesteps = obs.shape

    # Sample latent
    latent = guide.sample(obs, [])
    x, eos, _ = latent
    num_chars = lstm_util.get_num_timesteps(eos)

    # Sample reconstructions
    obs_predictions = generative_model.sample_obs_predictions(latent, obs, [num_samples])

    # Plot
    num_rows = 1
    num_cols = num_test_obs
    fig, axss = plt.subplots(
        num_rows,
        num_cols,
        figsize=(3 * num_cols, 2 * num_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for test_obs_id in range(num_test_obs):
        # Plot obs
        ax = axss[0, test_obs_id]
        plot_obs(ax, obs[test_obs_id])

        expression = timeseries_util.get_expression(x[test_obs_id][: num_chars[test_obs_id]])
        ax.text(
            0.95,
            0.95,
            f"Inferred kernel: {expression}",
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            ha="right",
            color="gray",
        )
        for sample_id in range(num_samples):
            ax.plot(
                torch.arange(data.num_timesteps, 2 * data.num_timesteps).float(),
                obs_predictions[sample_id, test_obs_id].cpu().detach(),
                color="C1",
                alpha=0.5,
            )

    util.save_fig(fig, path)


def main(args):
    # Cuda
    device = util.get_device()

    # Checkpoint paths
    if args.checkpoint_path is None:
        checkpoint_paths = list(util.get_checkpoint_paths(args.experiment_name))
    else:
        checkpoint_paths = [args.checkpoint_path]

    # Plot log p and KL for all checkpoints
    util.logging.info(f"Plotting stats for all runs in the experiment: {checkpoint_paths}")
    fig, axs = plt.subplots(1, 2, figsize=(2 * 6, 1 * 4))

    colors = {"cmws": "C0", "rws": "C1"}
    for checkpoint_path in checkpoint_paths:
        # Fix seed
        util.set_seed(1)

        util.logging.info(f"Start {checkpoint_path}")

        if os.path.exists(checkpoint_path):
            # Load checkpoint
            model, optimizer, stats, run_args = timeseries_util.load_checkpoint(
                checkpoint_path, device=device
            )
            generative_model, guide = model["generative_model"], model["guide"]
            num_iterations = len(stats.losses)

            label = run_args.algorithm if run_args.seed == 1 else None
            color = colors[run_args.algorithm]
            plot_kwargs = {"label": label, "color": color, "alpha": 0.8, "linewidth": 1.5}

            # Logp
            ax = axs[0]
            ax.plot([x[0] for x in stats.log_ps], [x[1] for x in stats.log_ps], **plot_kwargs)

            # KL
            ax = axs[1]
            ax.plot([x[0] for x in stats.kls], [x[1] for x in stats.kls], **plot_kwargs)
        util.logging.info(f"End {checkpoint_path}")
    ax = axs[0]
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Log p")
    ax.set_ylim(-45000, -10000)

    ax = axs[1]
    ax.set_xlabel("Iteration")
    ax.set_ylabel("KL")
    ax.legend()
    for ax in axs:
        # ax.set_xlim(0, 20000)
        sns.despine(ax=ax, trim=True)
    util.save_fig(fig, f"save/{args.experiment_name}/stats.png", dpi=200)
    return

    # Plot for all checkpoints
    for checkpoint_path in checkpoint_paths:
        # Fix seed
        util.set_seed(1)

        if os.path.exists(checkpoint_path):
            # Load checkpoint
            model, optimizer, stats, run_args = timeseries_util.load_checkpoint(
                checkpoint_path, device=device
            )
            generative_model, guide, memory = (
                model["generative_model"],
                model["guide"],
                model["memory"],
            )
            num_iterations = len(stats.losses)
            save_dir = util.get_save_dir(run_args.experiment_name, run.get_config_name(run_args))

            # Plot stats
            plot_stats(f"{save_dir}/stats.png", stats)

            # Plot reconstructions and other things
            # Test data
            # num_test_data = 50
            timeseries_dataset = data.TimeseriesDataset(device, test=True)
            obs, _ = timeseries_dataset[700:750]

            # Plot
            if run_args.model_type == "timeseries":
                plot_reconstructions_timeseries(
                    f"{save_dir}/reconstructions/{num_iterations}.pdf",
                    generative_model,
                    guide,
                    obs,
                )
                plot_predictions_timeseries(
                    f"{save_dir}/predictions/{num_iterations}.pdf", generative_model, guide, obs,
                )

        else:
            # Checkpoint doesn't exist
            util.logging.info(f"No checkpoint in {checkpoint_path}")


def get_parser():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--experiment-name", type=str, default="", help=" ")
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
