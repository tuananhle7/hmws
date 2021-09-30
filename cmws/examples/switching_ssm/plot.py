import os

import time
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from cmws import util
from cmws.examples.switching_ssm import util as switching_ssm_util
import ssm
from ssm.util import random_rotation, find_permutation
from cmws.examples.switching_ssm.data import SLDSDataset


def append_to_dict(k, x, dict_):
    if k in dict_:
        dict_[k].append(x)
    else:
        dict_[k] = [x]


def get_ground_truth_elbo(obs_dim, num_states, continuous_dim, obs):
    print("Fitting SLDS with Laplace-EM")

    # Create the model and initialize its parameters
    slds = ssm.SLDS(obs_dim, num_states, continuous_dim, emissions="gaussian_orthog")

    # Fit the model using Laplace-EM with a structured variational posterior
    q_lem_elbos, q_lem = slds.fit(
        obs.cpu().numpy(),
        method="laplace_em",
        variational_posterior="structured_meanfield",
        num_iters=100,
        alpha=0.0,
    )

    return q_lem_elbos[-1]


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


def plot_comparison(path, paper_plot_path, checkpoint_paths):
    device = util.get_device()

    # COMPUTE GROUND TRUTH ELBO
    # Load checkpoint
    _, _, _, run_args = switching_ssm_util.load_checkpoint(checkpoint_paths[0], device=device)
    slds_dataset = SLDSDataset(device)
    obs, obs_id = slds_dataset[0]
    ground_truth_elbo = get_ground_truth_elbo(
        run_args.obs_dim, run_args.num_states, run_args.continuous_dim, obs
    )

    algorithms = ["cmws_5", "rws", "vimco_2", "reinforce"]
    lrss = [[1e-3] for _ in range(len(algorithms))]

    util.logging.info(f"Plotting stats for all runs in the experiment: {checkpoint_paths}")

    # Load
    x = []
    log_ps = {}
    kls = {}
    for checkpoint_path in checkpoint_paths:
        if os.path.exists(checkpoint_path):
            # Load checkpoint
            model, optimizer, stats, run_args = switching_ssm_util.load_checkpoint(
                checkpoint_path, device=device
            )
            x_new = [x[0] for x in stats.log_ps]
            if len(x_new) > len(x):
                x = x_new
            append_to_dict(
                (run_args.algorithm, run_args.continuous_guide_lr),
                [x[1] for x in stats.log_ps],
                log_ps,
            )
            append_to_dict(
                (run_args.algorithm, run_args.continuous_guide_lr), [x[1] for x in stats.kls], kls
            )

    # Make numpy arrays
    max_len = len(x)
    num_seeds = 10
    # algorithms = ["cmws_5", "rws"]
    # log_ps_np = dict(
    #     [[algorithm, np.full((num_seeds, max_len), np.nan)] for algorithm in algorithms]
    # )
    # kls_np = dict([[algorithm, np.full((num_seeds, max_len), np.nan)] for algorithm in algorithms])
    log_ps_np, kls_np = {}, {}
    for algorithm, continuous_guide_lrs in zip(algorithms, lrss):
        for continuous_guide_lr in continuous_guide_lrs:
            log_ps_np[(algorithm, continuous_guide_lr)] = np.full((num_seeds, max_len), np.nan)
            kls_np[(algorithm, continuous_guide_lr)] = np.full((num_seeds, max_len), np.nan)
            for seed in range(num_seeds):
                try:
                    log_p = log_ps[(algorithm, continuous_guide_lr)][seed]
                    kl = kls[(algorithm, continuous_guide_lr)][seed]
                except Exception:
                    log_p = []
                    kl = []
                log_ps_np[(algorithm, continuous_guide_lr)][seed][: len(log_p)] = log_p
                kls_np[(algorithm, continuous_guide_lr)][seed][: len(kl)] = kl

    # Plot
    colors = {"cmws_5": "C0", "rws": "C1", "vimco": "C2", "reinforce": "C3", "vimco_2": "C4"}
    linestyles = {5e-3: "dotted", 1e-3: "solid", 5e-4: "dashed"}
    fig, axs = plt.subplots(1, 2, figsize=(2 * 6, 1 * 4))

    actual_fig, actual_ax = plt.subplots(1, 1, figsize=(6, 4))
    # for algorithm in algorithms:
    for algorithm, continuous_guide_lrs in zip(algorithms, lrss):
        for continuous_guide_lr in continuous_guide_lrs:
            if "cmws" in algorithm:
                label = "HMWS"
            elif "vimco" in algorithm:
                label = "VIMCO"
            else:
                label = algorithm.upper()
            # label = f"{algorithm} {continuous_guide_lr}"
            linestyle = linestyles[continuous_guide_lr]
            color = colors[algorithm]
            plot_kwargs = {"color": color, "linestyle": linestyle, "label": label}

            # Logp
            log_p = log_ps_np[(algorithm, continuous_guide_lr)]
            ax = axs[0]
            plot_with_error_bars(ax, x, log_p, **plot_kwargs)
            plot_with_error_bars(actual_ax, x, log_p, **plot_kwargs)

            # KL
            kl = kls_np[(algorithm, continuous_guide_lr)]
            ax = axs[1]
            plot_with_error_bars(ax, x, kl, **plot_kwargs)

    ax = axs[0]
    ax.axhline(ground_truth_elbo, color="black", label="Laplace EM")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("log p(x)")
    ax.set_ylim(-5000, 0)
    # ax.set_xlim(0, 2000)
    ax.legend()

    ax = axs[1]
    ax.set_xlabel("Iteration")
    ax.set_ylabel("KL")
    # ax.set_ylim(0, 1000000)
    # ax.set_xlim(0, 2000)
    ax.legend()
    for ax in axs:
        # ax.set_xlim(0, 20000)
        sns.despine(ax=ax, trim=True)
    util.save_fig(fig, path, dpi=200)

    actual_ax.set_ylim(0, 35)
    actual_ax.set_xticks([0, 10000])
    actual_ax.set_xlabel("Iteration", labelpad=-10)
    actual_ax.set_ylabel(f"$\\log p_\\theta(x)$")
    actual_ax.legend()
    sns.despine(ax=actual_ax, trim=True)
    util.save_fig(actual_fig, paper_plot_path, dpi=200)


def main(args):
    # Checkpoint paths
    if args.checkpoint_path is None:
        checkpoint_paths = list(util.get_checkpoint_paths(args.experiment_name))
    else:
        checkpoint_paths = [args.checkpoint_path]

    plot_comparison(
        f"save/{args.experiment_name}/stats.png",
        f"save/{args.experiment_name}/stats.pdf",
        checkpoint_paths,
    )


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
