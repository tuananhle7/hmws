import os

import time
import cmws
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
import torch
import numpy as np
from cmws import util
from cmws.examples.timeseries import data, run
from cmws.examples.timeseries import util as timeseries_util
from cmws.examples.timeseries import lstm_util
import cmws.examples.timeseries.inference


def plot_obs(ax, obs):
    """
    Args
        ax
        obs [num_timesteps]
    """
    if torch.is_tensor(obs):
        obs = obs.cpu().numpy()
    ax.plot(obs, color="C0")
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
    losses = np.array(stats.losses)
    losses[losses > 1000] = np.nan
    ax.plot(losses)
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


def plot_predictions_timeseries(path, generative_model, guide, obs, memory=None, obs_id=None):
    """
    Args:
        path (str)
        generative_model
        guide
        obs: [num_test_obs, num_timesteps]
        memory
        obs_id
    """
    num_samples = 1
    num_test_obs, num_timesteps = obs.shape

    # Sample latent
    num_svi_iterations = 10
    if memory is None:
        num_particles = 10
        latent, log_weight = cmws.examples.timeseries.inference.svi_importance_sampling(
            num_particles, num_svi_iterations, obs, generative_model, guide
        )
    else:
        assert obs_id is not None
        num_particles = 10
        # latent, log_weight = cmws.examples.timeseries.inference.svi_memory(
        #     num_svi_iterations, obs, obs_id, generative_model, guide, memory
        # )
        latent, log_weight = cmws.examples.timeseries.inference.importance_sample_memory(
            num_particles, num_svi_iterations, obs, obs_id, generative_model, guide, memory
        )
    x, eos, _ = latent
    num_chars = lstm_util.get_num_timesteps(eos)

    # Sort by log weight
    # [num_test_obs, num_particles], [num_test_obs, num_particles]
    _, sorted_indices = torch.sort(log_weight.T, descending=True)

    # Sample predictions
    # -- Expand obs
    obs_expanded = obs[None].expand(num_particles, num_test_obs, num_timesteps)

    # -- Sample predictions
    obs_predictions = generative_model.sample_obs_predictions(latent, obs_expanded, [num_samples])

    # Plot
    num_rows = num_particles
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
        for particle_id in range(num_particles):
            # Plot obs
            ax = axss[particle_id, test_obs_id]
            plot_obs(ax, obs[test_obs_id])

            # Compute sorted particle id
            sorted_particle_id = sorted_indices[test_obs_id, particle_id]

            long_expression = timeseries_util.get_long_expression(
                timeseries_util.get_expression(
                    x[sorted_particle_id, test_obs_id][: num_chars[sorted_particle_id, test_obs_id]]
                )
            )
            ax.text(
                0.05,
                0.95,
                "\n".join(textwrap.wrap(long_expression, 20)),
                transform=ax.transAxes,
                fontsize=7,
                va="top",
                ha="left",
                color="black",
            )
            ax.text(
                0.95,
                0.95,
                f"{log_weight[sorted_particle_id, test_obs_id].item():.0f}",
                transform=ax.transAxes,
                fontsize=7,
                va="top",
                ha="right",
                color="black",
            )
            for sample_id in range(num_samples):
                ax.plot(
                    torch.arange(data.num_timesteps, 2 * data.num_timesteps).float(),
                    obs_predictions[sample_id, sorted_particle_id, test_obs_id].cpu().detach(),
                    color="C1",
                    alpha=0.5,
                )

    for particle_id in range(num_particles):
        axss[particle_id, 0].set_ylabel(f"Particle {particle_id}")

    util.save_fig(fig, path)


def plot_prior_timeseries(path, generative_model, num_samples):
    """
    Args:
        path (str)
        generative_model
        num_samples
    """
    latent, obs = generative_model.sample([num_samples])
    raw_expression, eos, _ = latent
    num_chars = lstm_util.get_num_timesteps(eos)

    # Plot
    num_rows = 1
    num_cols = num_samples
    fig, axss = plt.subplots(
        num_rows,
        num_cols,
        figsize=(3 * num_cols, 2 * num_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for sample_id in range(num_samples):
        # Plot obs
        ax = axss[0, sample_id]
        plot_obs(ax, obs[sample_id])
        long_expression = timeseries_util.get_long_expression(
            timeseries_util.get_expression(raw_expression[sample_id][: num_chars[sample_id]])
        )
        ax.text(
            0.95,
            0.95,
            "\n".join(textwrap.wrap(long_expression, 20)),
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            ha="right",
            color="black",
        )

    util.save_fig(fig, path)


def plot_with_error_bars(ax, x, data, **plot_kwargs):
    mid = np.nanmedian(data, axis=0)
    low = np.nanpercentile(data, 25, axis=0)
    high = np.nanpercentile(data, 75, axis=0)

    num_not_nan = np.count_nonzero(~np.isnan(mid))

    if num_not_nan > 0:
        ax.plot(x[:num_not_nan], mid[:num_not_nan], **plot_kwargs)
        ax.fill_between(
            x[:num_not_nan], low[:num_not_nan], high[:num_not_nan], alpha=0.2, **plot_kwargs
        )


def plot_comparison(path, checkpoint_paths):
    device = util.get_device()

    util.logging.info(f"Plotting stats for all runs in the experiment: {checkpoint_paths}")

    # Load
    x = []
    log_ps = {"cmws": [], "rws": []}
    kls = {"cmws": [], "rws": []}
    colors = {"cmws": "C0", "rws": "C1"}
    for checkpoint_path in checkpoint_paths:
        if os.path.exists(checkpoint_path):
            # Load checkpoint
            model, optimizer, stats, run_args = timeseries_util.load_checkpoint(
                checkpoint_path, device=device
            )
            x_new = [x[0] for x in stats.log_ps]
            if len(x_new) > len(x):
                x = x_new
            log_ps[run_args.algorithm].append([x[1] for x in stats.log_ps])
            kls[run_args.algorithm].append([x[1] for x in stats.kls])

    # Make numpy arrays
    max_len = len(x)
    num_seeds = 5
    algorithms = ["cmws", "rws"]
    log_ps_np = dict(
        [[algorithm, np.full((num_seeds, max_len), np.nan)] for algorithm in algorithms]
    )
    kls_np = dict([[algorithm, np.full((num_seeds, max_len), np.nan)] for algorithm in algorithms])
    for algorithm in algorithms:
        for seed in range(num_seeds):
            log_p = log_ps[algorithm][seed]
            kl = kls[algorithm][seed]
            log_ps_np[algorithm][seed][: len(log_p)] = log_p
            kls_np[algorithm][seed][: len(kl)] = kl

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(2 * 6, 1 * 4))
    colors = {"cmws": "C0", "rws": "C1"}
    for algorithm in algorithms:
        label = algorithm
        linestyle = "solid"
        color = colors[algorithm]
        plot_kwargs = {"color": color, "linestyle": linestyle, "label": label}

        # Logp
        log_p = log_ps_np[algorithm]
        ax = axs[0]
        plot_with_error_bars(ax, x, log_p, **plot_kwargs)

        # KL
        kl = kls_np[algorithm]
        ax = axs[1]
        plot_with_error_bars(ax, x, kl, **plot_kwargs)

    ax = axs[0]
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Log p")
    ax.set_ylim(-500, 600)
    ax.set_xlim(0, 5000)
    ax.legend()

    ax = axs[1]
    ax.set_xlabel("Iteration")
    ax.set_ylabel("KL")
    ax.set_ylim(0, 1000000)
    ax.set_xlim(0, 5000)
    ax.legend()
    for ax in axs:
        # ax.set_xlim(0, 20000)
        sns.despine(ax=ax, trim=True)
    util.save_fig(fig, path, dpi=200)


def main(args):
    # Cuda
    device = util.get_device()

    # Checkpoint paths
    if args.checkpoint_path is None:
        checkpoint_paths = list(util.get_checkpoint_paths(args.experiment_name))
    else:
        checkpoint_paths = [args.checkpoint_path]

    # Plot log p and KL for all checkpoints
    # plot_comparison(f"save/{args.experiment_name}/stats.png", checkpoint_paths)
    # return
    util.logging.info(
        f"Max GPU memory allocated = {util.get_max_gpu_memory_allocated_MB(device):.0f} MB"
    )

    # Plot for all checkpoints
    plotted_something = False
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
            # Load data
            obs = {}

            # -- Test
            test_timeseries_dataset = data.TimeseriesDataset(device, test=True)
            obs["test"], _ = test_timeseries_dataset[:]

            # -- Train
            train_timeseries_dataset = data.TimeseriesDataset(
                device, test=False, full_data=run_args.full_training_data
            )
            if run_args.full_training_data:
                obs["train"], obs_id = train_timeseries_dataset[
                    [62, 188, 269, 510, 711, 1262, 1790]
                ]
            else:
                obs["train"], obs_id = train_timeseries_dataset[:]

            # Plot
            if run_args.model_type == "timeseries":
                # Plot prior
                plot_prior_timeseries(
                    f"{save_dir}/prior/{num_iterations}.png", generative_model, num_samples=100
                )
                # Plot predictions
                if memory is not None:
                    plot_predictions_timeseries(
                        f"{save_dir}/predictions/train/memory/{num_iterations}.png",
                        generative_model,
                        guide,
                        obs["train"],
                        memory,
                        obs_id,
                    )
                for mode in ["train", "test"]:
                    plot_predictions_timeseries(
                        f"{save_dir}/predictions/{mode}/guide/{num_iterations}.png",
                        generative_model,
                        guide,
                        obs[mode],
                    )

            plotted_something = True
        else:
            # Checkpoint doesn't exist
            util.logging.info(f"No checkpoint in {checkpoint_path}")

    util.logging.info(
        f"Max GPU memory allocated = {util.get_max_gpu_memory_allocated_MB(device):.0f} MB"
    )
    return plotted_something


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
                plotted_something = main(args)
                if not plotted_something:
                    util.logging.info("Didn't plot anything ... waiting 30 seconds")
                    time.sleep(30)
        else:
            main(args)
