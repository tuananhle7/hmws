"""
Compare SGD and IS, assuming *memory elements are correct*.
"""

import run
import util
import torch
import matplotlib.pyplot as plt
import numpy as np


def get_lo_mid_hi(x):
    mid = np.mean(x)
    std = np.std(x)
    lo = mid - std
    hi = mid + std
    return lo, mid, hi


def init(seed):
    util.set_seed(seed)
    device = "cpu"
    args = run.get_args_parser().parse_args([])
    args.algorithm = "cmws"
    generative_model, guide, optimizer, memory, stats = util.init(args, device=device)
    memory = torch.tensor([2, 3, 4], device=device).long()
    return generative_model, guide, memory


def main():
    seeds = range(20)
    num_particles_or_iterationss = [10, 50, 100, 500, 1000, 5000]
    is_errors, sgd_errors = [np.zeros((len(num_particles_or_iterationss), 3)) for _ in range(2)]
    for i, num_particles_or_iterations in enumerate(num_particles_or_iterationss):
        is_error = []
        sgd_error = []
        for seed in seeds:
            generative_model, guide, memory = init(seed)

            # IS
            is_error.append(
                util.get_cmws_memory_error(
                    generative_model,
                    guide,
                    memory,
                    num_particles=num_particles_or_iterations,
                    num_iterations=None,
                )
            )

            # SGD
            sgd_error.append(
                util.get_cmws_memory_error(
                    generative_model,
                    guide,
                    memory,
                    num_particles=None,
                    num_iterations=num_particles_or_iterations,
                )
            )
        is_errors[i] = get_lo_mid_hi(is_error)
        sgd_errors[i] = get_lo_mid_hi(sgd_error)

    print(f"is_errors = {is_errors}")
    print(f"sgd_errors = {sgd_errors}")

    fig, ax = plt.subplots(1, 1)
    for data, label, color in zip([is_errors, sgd_errors], ["IS", "SGD"], ["C0", "C1"]):
        ax.plot(
            num_particles_or_iterationss, data[:, 1], color=color, label=label,
        )
        ax.fill_between(
            num_particles_or_iterationss, data[:, 0], data[:, 2], alpha=0.5, color=color
        )
    ax.set_ylabel("q(d) error")

    ax.legend()
    ax.set_xlabel("Number of particles (blue) / iterations (red)")
    util.save_fig(fig, "save/marginalization_comparison.pdf")


if __name__ == "__main__":
    main()
