import util
import sweep
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_lo_mid_hi(x):
    mid = np.mean(x)
    std = np.std(x)
    lo = mid - std
    hi = mid + std
    return lo, mid, hi


def main():
    sweep_argss = list(sweep.get_sweep_argss())
    sweep_args = sweep_argss[0]
    device = "cpu"
    seeds = range(5)
    num_particless = [2, 4, 6, 8, 10]
    num_cmws_iterationss = num_particless

    statss = {}
    for i, sweep_args in enumerate(sweep_argss):
        checkpoint_path = util.get_checkpoint_path(sweep_args)
        (generative_model, guide, optimizer, memory, stats, run_args) = util.load_checkpoint(
            checkpoint_path, device=device
        )
        statss[(sweep_args.seed, sweep_args.num_particles, sweep_args.num_cmws_iterations)] = stats

    # IS CMWS Estimator
    is_inference_errors, is_cmws_memory_errors, is_locs_errors = [
        np.zeros((len(num_particless), 3)) for _ in range(3)
    ]
    for i, num_particles in enumerate(num_particless):
        is_inference_errors[i] = get_lo_mid_hi(
            [statss[(seed, num_particles, None)].inference_errors[-1] for seed in seeds]
        )
        is_cmws_memory_errors[i] = get_lo_mid_hi(
            [statss[(seed, num_particles, None)].cmws_memory_errors[-1] for seed in seeds]
        )
        is_locs_errors[i] = get_lo_mid_hi(
            [statss[(seed, num_particles, None)].locs_errors[-1] for seed in seeds]
        )

    # SGD CMWS Estimator
    sgd_inference_errors, sgd_cmws_memory_errors, sgd_locs_errors = [
        np.zeros((len(num_cmws_iterationss), 3)) for _ in range(3)
    ]
    for i, num_cmws_iterations in enumerate(num_cmws_iterationss):
        sgd_inference_errors[i] = get_lo_mid_hi(
            [statss[(seed, None, num_cmws_iterations)].inference_errors[-1] for seed in seeds]
        )
        sgd_cmws_memory_errors[i] = get_lo_mid_hi(
            [statss[(seed, None, num_cmws_iterations)].cmws_memory_errors[-1] for seed in seeds]
        )
        sgd_locs_errors[i] = get_lo_mid_hi(
            [statss[(seed, None, num_cmws_iterations)].locs_errors[-1] for seed in seeds]
        )

    # Exact CMWS Estimator
    exact_inference_errors = get_lo_mid_hi(
        [statss[(seed, None, None)].inference_errors[-1] for seed in seeds]
    )
    exact_cmws_memory_errors = get_lo_mid_hi(
        [statss[(seed, None, None)].cmws_memory_errors[-1] for seed in seeds]
    )
    exact_locs_errors = get_lo_mid_hi(
        [statss[(seed, None, None)].locs_errors[-1] for seed in seeds]
    )

    fig, axs = plt.subplots(3, 1, figsize=(6 * 1, 4 * 3), sharex=True)

    for algorithm, color, errorss in zip(
        ["IS", "SGD"],
        ["C0", "C1"],
        [
            [is_inference_errors, is_cmws_memory_errors, is_locs_errors],
            [sgd_inference_errors, sgd_cmws_memory_errors, sgd_locs_errors],
        ],
    ):
        # if algorithm == "SGD":
        #     continue
        for ax, errors in zip(axs, errorss):
            ax.plot(num_particless, errors[:, 1], color=color, label=algorithm)
            ax.fill_between(num_particless, errors[:, 0], errors[:, 2], alpha=0.5, color=color)

    color = "C2"
    algorithm = "Exact"
    for ax, errors in zip(
        axs, [exact_inference_errors, exact_cmws_memory_errors, exact_locs_errors]
    ):
        ax.axhline(errors[1], color=color, label=algorithm)
        ax.axhspan(errors[0], errors[2], facecolor=color, alpha=0.5)

    for ax, ylabel in zip(axs, ["q(d, c) error", "q(d) error", "q(c | d) error"]):
        ax.set_ylabel(ylabel)
        sns.despine(ax=ax, trim=True)

    axs[0].legend()
    axs[-1].set_xlabel("Number of particles / iterations")

    fig.suptitle(f"Inference error ({len(seeds)} runs; $\pm$ 1 s.d.)")
    util.save_fig(fig, "save/cmws_comparison.pdf", tight_layout_kwargs={"rect": [0, 0.03, 1, 0.97]})


if __name__ == "__main__":
    main()
