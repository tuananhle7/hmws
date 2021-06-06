import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from cmws import util
from cmws.examples.scene_understanding import data, render, run
from cmws.examples.scene_understanding import util as scene_understanding_util
import time
import numpy as np


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

# from: https://github.com/tuananhle7/continuous_mws/blob/master/cmws/examples/timeseries/plot.py#L244
def plot_with_error_bars(ax, x, data, **plot_kwargs):
    mid = np.nanmedian(data, axis=0)
    low = np.nanpercentile(data, 25, axis=0)
    high = np.nanpercentile(data, 75, axis=0)

    num_not_nan = np.count_nonzero(~np.isnan(mid))

    if num_not_nan > 0:
        lines = ax.plot(x[:num_not_nan], mid[:num_not_nan], **plot_kwargs)
        ax.fill_between(
            x[:num_not_nan],
            low[:num_not_nan],
            high[:num_not_nan],
            alpha=0.2,
            color=lines[0].get_color(),
        )

def get_analysis_plots(experiment_name, grid_sizes=[2, 3], cmws_version="cmws_2"):
    save_dir = f"../save/{experiment_name}"
    checkpoint_paths = []
    for config_name in sorted(os.listdir(save_dir)):
        checkpoint_paths.append(util.get_checkpoint_path(experiment_name, config_name, -1))

    for grid_size in grid_sizes:  # grid size
        fig, axs = plt.subplots(1, 2, figsize=(2 * 6, 1 * 4))

        colors = {cmws_version: "C0", "rws": "C1"}
        added_label = {k: False for k in colors.keys()}  # keep track of whether we've used alg as label
        for checkpoint_path in checkpoint_paths:
            checkpoint_path = f"../{checkpoint_path}"

            # Fix seed
            util.set_seed(1)

            if os.path.exists(checkpoint_path):
                # Load checkpoint
                try:
                    model, optimizer, stats, run_args = scene_understanding_util.load_checkpoint(
                        checkpoint_path, device="cpu"
                    )
                except:
                    continue

                if run_args.num_grid_cols != grid_size: continue

                generative_model, guide = model["generative_model"], model["guide"]
                num_iterations = len(stats.losses)

                if not added_label[run_args.algorithm]:
                    label = run_args.algorithm
                    added_label[run_args.algorithm] = True
                else:
                    label = None
                color = colors[run_args.algorithm]
                plot_kwargs = {"label": label, "color": color, "alpha": 0.8, "linewidth": 1.5}

                # Logp
                ax = axs[0]
                ax.plot([x[0] for x in stats.log_ps], [x[1] for x in stats.log_ps], **plot_kwargs)

                # KL
                ax = axs[1]
                ax.plot([x[0] for x in stats.kls], [x[1] for x in stats.kls], **plot_kwargs)
        ax = axs[0]
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Log p")

        ax = axs[1]
        ax.set_xlabel("Iteration")
        ax.set_ylabel("KL")
        ax.legend()
        for ax in axs:
            sns.despine(ax=ax, trim=True)
        util.save_fig(fig, f"{save_dir}/losses_{grid_size}.png", dpi=200)

    for grid_size in grid_sizes:  # grid size
        # Load
        x = []
        log_ps = {cmws_version: [], "cmws": [], "rws": []}
        kls = {cmws_version: [], "cmws": [], "rws": []}
        colors = {cmws_version: "C0", "rws": "C1"}
        for checkpoint_path in checkpoint_paths:
            checkpoint_path = f"../{checkpoint_path}"
            if os.path.exists(checkpoint_path):
                # Load checkpoint
                try:
                    model, optimizer, stats, run_args = scene_understanding_util.load_checkpoint(
                        checkpoint_path, device="cpu"
                    )
                except:
                    continue
                if run_args.num_grid_cols != grid_size: continue
                x_new = [x[0] for x in stats.log_ps]
                if len(x_new) > len(x):
                    x = x_new
                log_ps[run_args.algorithm].append([x[1] for x in stats.log_ps])
                kls[run_args.algorithm].append([x[1] for x in stats.kls])
        # Make numpy arrays
        max_len = len(x)
        if grid_size == 2:
            num_seeds = 5  # 10
        else:
            num_seeds = 5
        algorithms = [cmws_version, "rws"]
        log_ps_np = dict(
            [[algorithm, np.full((num_seeds, max_len), np.nan)] for algorithm in algorithms]
        )
        kls_np = dict([[algorithm, np.full((num_seeds, max_len), np.nan)] for algorithm in algorithms])
        for algorithm in algorithms:
            for seed in range(num_seeds):
                try:
                    log_p = log_ps[algorithm][seed]
                    kl = kls[algorithm][seed]
                except Exception:
                    log_p = []
                    kl = []
                log_ps_np[algorithm][seed][: len(log_p)] = log_p
                kls_np[algorithm][seed][: len(kl)] = kl

        # Plot
        fig, axs = plt.subplots(1, 2, figsize=(2 * 6, 1 * 4))
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
        ax.legend()

        ax = axs[1]
        ax.set_xlabel("Iteration")
        ax.set_ylabel("KL")
        ax.legend()
        for ax in axs:
            sns.despine(ax=ax, trim=True)
        util.save_fig(fig, f"{save_dir}/losses_{grid_size}_overlay.png", dpi=200)


def latent_to_str(latent):
    num_blocks, stacking_program, raw_locations = latent
    num_grid_rows, num_grid_cols, max_num_blocks = stacking_program.shape
    latent_str = []
    for row in reversed(range(num_grid_rows)):
        tmp = []
        for col in reversed(range(num_grid_cols)):
            s = ["." for _ in range(max_num_blocks)]
            s[: num_blocks[row, col]] = "".join(
                [str(x.item()) for x in stacking_program[row, col, : num_blocks[row, col]]]
            )
            tmp.append("".join(s))
        latent_str.append("|".join(tmp))

    return "\n".join(latent_str)

def plot_memory_scene_understanding(path, generative_model, guide, memory, obs, obs_id):
    # modified from: https://github.com/tuananhle7/continuous_mws/blob/master/cmws/examples/timeseries/plot.py#L87-L102
    obs = obs.squeeze(1)
    num_test_obs, num_channels, im_size, _ = obs.shape
    im_size = 256

    num_particles = memory.size
    latent, log_weight = util.importance_sample_memory(
        num_particles, obs, obs_id, generative_model, guide, memory, im_size
    )

    num_blocks, stacking_program, raw_locations = latent

    # Sort by log weight
    # [num_test_obs, num_particles], [num_test_obs, num_particles]
    _, sorted_indices = torch.sort(log_weight.T, descending=True)

    # Sample predictions
    # -- Expand obs
    obs_expanded = obs[None].expand(num_particles, num_test_obs, 3, im_size, im_size)

    num_rows = 4
    num_cols = 5  # number to show
    fig, axss = plt.subplots(
        num_rows,
        num_cols,
        figsize=(3 * num_cols, 2 * num_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for row in range(num_rows):
        for test_obs_id in range(num_cols):
            ax = axss[row][test_obs_id]
            if row == 0:  # pull from observations
                img = obs_expanded[0][test_obs_id].permute(1, 2, 0)
            else:
                particle_id = row - 1
                sorted_particle_id = sorted_indices[test_obs_id, particle_id]

                num_blocks_selected = num_blocks[sorted_particle_id, test_obs_id]
                stacking_program_selected = stacking_program[sorted_particle_id, test_obs_id]
                raw_locations_selected = raw_locations[sorted_particle_id, test_obs_id]

                sampled_latent = (num_blocks_selected, stacking_program_selected, raw_locations_selected)

                camera_elevation = 30
                camera_azimuth = -40  # 40

                sampled_obs = generative_model.get_obs_loc(sampled_latent, (camera_elevation, camera_azimuth))

                img = sampled_obs.permute(1, 2, 0).detach().numpy()
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
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
    util.save_fig(fig, path)


def plot_reconstructions_scene_understanding(path, generative_model, guide, obs):
    """
    Args:
        path (str)
        generative_model
        guide
        obs: [num_test_obs, num_channels, im_size, im_size]
    """
    obs = obs.squeeze(1)
    num_test_obs, num_channels, im_size, _ = obs.shape
    hi_res_im_size = 256

    # Sample latent
    latent = guide.sample(obs)
    num_blocks, stacking_program, raw_locations = latent

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
    fig, axss = plt.subplots(
        num_rows, num_cols, figsize=(2 * num_cols, 2 * num_rows), squeeze=False
    )
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
        ax.text(
            0.95,
            0.95,
            latent_to_str(
                (num_blocks[sample_id], stacking_program[sample_id], raw_locations[sample_id])
            ),
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            ha="right",
            color="black",
        )

        # Plot probs > 0.5
        axss[2, sample_id].imshow(reconstructed_obs_hi_res[sample_id].cpu().permute(1, 2, 0))

    # Set labels
    axss[0, 0].set_ylabel("Observed image")
    axss[1, 0].set_ylabel("Reconstruction")
    axss[2, 0].set_ylabel("Hi-res reconstruction")

    util.save_fig(fig, path, dpi=300)


def plot_primitives_scene_understanding(path, generative_model, remove_color=False, mode="cube",
                                        camera_elevation=0.1, camera_azimuth=0):
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
        obs = render.render_block(
            generative_model.primitives[i].size,
            generative_model.primitives[i].color,
            location,
            im_size=im_size,
            remove_color=remove_color,
            mode=mode,
            camera_elevation=camera_elevation,
            camera_azimuth=camera_azimuth
        )
        obs_high_res = render.render_block(
            generative_model.primitives[i].size,
            generative_model.primitives[i].color,
            location,
            im_size=hi_res_im_size,
            remove_color=remove_color,
            mode=mode,
            camera_elevation=camera_elevation,
            camera_azimuth=camera_azimuth
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
        checkpoint_paths = list(util.get_checkpoint_paths(args.experiment_name))
    else:
        checkpoint_paths = [args.checkpoint_path]

    # Plot for all checkpoints
    for checkpoint_path in checkpoint_paths:
        # Fix seed
        util.set_seed(1)

        if os.path.exists(checkpoint_path):
            # Load checkpoint
            try:
                model, optimizer, stats, run_args = scene_understanding_util.load_checkpoint(
                    checkpoint_path, device=device
                )
            except: continue
            generative_model, guide = model["generative_model"], model["guide"]
            num_iterations = len(stats.losses)
            save_dir = util.get_save_dir(run_args.experiment_name, run.get_config_name(run_args))

            # Plot stats
            plot_stats(f"{save_dir}/stats.png", stats)

            # Plot reconstructions and other things
            # Test data
            # NOTE: Plotting the train dataset only
            train_dataset = data.SceneUnderstandingDataset(
                device, run_args.num_grid_rows, run_args.num_grid_cols, test=False,
                remove_color=run_args.remove_color,
                mode=run_args.mode
            )
            obs, obs_id = train_dataset[:10]

            # Plot
            if run_args.model_type == "scene_understanding":
                plot_reconstructions_scene_understanding(
                    f"{save_dir}/reconstructions/{num_iterations}.png",
                    generative_model,
                    guide,
                    obs
                )
                plot_primitives_scene_understanding(
                    f"{save_dir}/primitives/{num_iterations}.png", generative_model, run_args.remove_color,
                    run_args.mode
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
    parser.add_argument(
        "--delay", action="store_true", help="Whether to delay the start of execution"
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    with torch.no_grad():
        if args.delay:
            # delay start to ensure checkpoints exist before plotting
            time.sleep(5 * 60)  # units of seconds

        if args.repeat:
            while True:
                main(args)
        else:
            main(args)
