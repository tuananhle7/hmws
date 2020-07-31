import util
import sweep
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


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
    seeds = range(3)
    num_cmws_mc_sampless = [10, 50, 100, 500, 1000]
    num_particless = num_cmws_mc_sampless

    statss = {}
    for i, sweep_args in enumerate(sweep_argss):
        checkpoint_path = util.get_checkpoint_path(sweep_args)
        (generative_model, guide, optimizer, memory, stats, run_args) = util.load_checkpoint(
            checkpoint_path, device=device
        )
        statss[(sweep_args.seed, sweep_args.num_cmws_mc_samples, sweep_args.num_particles)] = stats

    cmws_memory_errors, locs_errors, inference_errors = [
        np.zeros((len(num_cmws_mc_sampless), len(num_particless), 3)) for _ in range(3)
    ]
    for i, num_cmws_mc_samples in enumerate(num_cmws_mc_sampless):
        for j, num_particles in enumerate(num_particless):
            inference_errors[i, j] = get_lo_mid_hi(
                [
                    statss[seed, num_cmws_mc_samples, num_particles].inference_errors[-1]
                    for seed in seeds
                ]
            )
            locs_errors[i, j] = get_lo_mid_hi(
                [statss[seed, num_cmws_mc_samples, num_particles].locs_errors[-1] for seed in seeds]
            )
            cmws_memory_errors[i, j] = get_lo_mid_hi(
                [
                    statss[seed, num_cmws_mc_samples, num_particles].cmws_memory_errors[-1]
                    for seed in seeds
                ]
            )

    print(cmws_memory_errors)
    print(locs_errors)
    print(inference_errors)

    norm = matplotlib.colors.Normalize(vmin=-1, vmax=len(num_cmws_mc_sampless) - 1)
    cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.Blues)
    cmap.set_array([])

    fig, axs = plt.subplots(3, 1, figsize=(6 * 1, 4 * 3), sharex=True)
    ax = axs[0]
    for ax, data, ylabel in zip(
        axs,
        [inference_errors, cmws_memory_errors, locs_errors],
        ["q(d, c) error", "q(d) error", "q(c | d) error"],
    ):
        for i, num_cmws_mc_samples in enumerate(num_cmws_mc_sampless):
            ax.plot(
                num_particless,
                data[i, :, 1],
                color=cmap.to_rgba(i),
                label=f"MC={num_cmws_mc_samples}",
            )
            ax.fill_between(
                num_particless, data[i, :, 0], data[i, :, 2], alpha=0.5, color=cmap.to_rgba(i),
            )
            ax.set_ylabel(ylabel)
    axs[-1].set_xlabel("Number of particles")
    axs[-1].legend()
    util.save_fig(fig, "save/cmws_comparison.pdf")


if __name__ == "__main__":
    main()
